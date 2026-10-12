// Operator/GEMM-TC/V5_annotated.cu
// ============================================================================
// V5.cu 的逐行注释版 —— Step 5: cp.async 多级流水线
// 逻辑与 V5.cu 完全一致，只是在代码行上方补了对应说明。
// 编译: nvcc -O3 -arch=native -Xptxas -v V5_annotated.cu -o V5_annotated.exe
// ============================================================================
//
// ---- 要解决的问题 ----
// V3/V4 每轮 K 是: 搬 tile(global->寄存器->shared) -> __syncthreads -> 算 -> __syncthreads。
// 搬运那段时间 Tensor Core 完全闲着。global 延迟是几百 cycle，而一个 K 迭代的计算只有
// 几十 cycle，所以 SM 大部分时间在等内存。
//
// ---- cp.async 怎么解决 ----
// 1. 让 global->shared 的搬运不经过寄存器，一条指令搞定(省指令、省寄存器)；
// 2. 搬运是异步的: 发出指令后立刻返回，SM 可以继续算；
// 3. 开 NSTAGE 级 buffer 做流水: 算第 i 轮的时候，第 i+1..i+NSTAGE-1 轮已经在路上。
//
// cp.async 的三个配套指令:
//   cp.async.cg.shared.global [dst], [src], 16;   // 搬 16 字节(8 个 half)
//   cp.async.commit_group;                        // 把"刚才发出的那批"打成一个组
//   cp.async.wait_group N;                        // 等到"未完成的组数 <= N"
//
// ---- 流水线的骨架 ----
//   序言: 先把最前面的 NSTAGE-1 轮搬进来(每轮一个 commit_group)
//   循环第 i 轮:
//       wait_group<NSTAGE-2>   // 让"本轮要算的那级"就绪，还允许 NSTAGE-2 组在飞
//       __syncthreads()        // 让全 block 都看到新数据
//       计算 buffer[i % NSTAGE]
//       把第 i+NSTAGE-1 轮搬进 buffer[(i+NSTAGE-1) % NSTAGE]  <- 正好是"上一轮算完的"那块
//       commit_group           // 即使越界没搬，也要 commit，否则组数就对不上了
//
// 为什么只需要一个 __syncthreads:
//   循环顶部这个 barrier 在程序顺序上晚于上一轮的计算，所以它同时保证了
//   "上一轮算完的那块 buffer 现在没人读了"，可以直接覆盖。
//
// ---- 关键点: 搬运的粒度必须和 shared 布局对齐 ----
// cp.async 一次搬 16 字节 = 8 个 half，所以搬 tile 的循环按"16B chunk"来数，不是按元素数。
// A tile 的 chunk 数 = BM * (BK/8)，B tile = BK * (BN/8)。
// ============================================================================

// 引入 CUDA runtime 基础 API(cudaMalloc / cudaMemcpy / cudaEvent 等)
#include <cuda_runtime.h>
// 引入 __half 类型与 __float2half / __half2float 等转换函数
#include <cuda_fp16.h>
// 定长整型 uint32_t 以及 size_t
#include <cstdint>
// printf
#include <cstdio>
// exit
#include <cstdlib>
// fabs
#include <cmath>
// host 侧的 std::vector
#include <vector>
// host 侧的 std::sort / std::max
#include <algorithm>

// CK(x): 执行一句 CUDA runtime 调用并检查返回值；出错就打印 文件:行号 与错误串后 exit(1)。
// 它是一个跨两行的宏(第一行末尾有行连接符 '\')，所以宏体内不能再插 '//' 注释。
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

// ---- 可调配置: 实验表就是改这两个宏(用 -DBKV=.. / -DSTAGES=.. 覆盖) ----
// 如果编译时没有定义 BKV...
#ifndef BKV
// ...就默认 K 块大小 BK = 64
#define BKV    64        // K 块大小
// 结束 #ifndef BKV
#endif
// 如果编译时没有定义 STAGES...
#ifndef STAGES
// ...就默认流水线 2 级
#define STAGES 2         // 流水线级数
// 结束 #ifndef STAGES
#endif

// block tile 大小: 128 x 128；K 方向一次吃 BK 列
constexpr int BM = 128, BN = 128, BK = BKV;
// A 在 shared 中的行距: 每行尾部多垫 8 个 half，用来消 ldmatrix 的 bank conflict
constexpr int AST = BK + 8;      // A padding 后的行距
// B 在 shared 中的行距: 同理垫 8 个 half
constexpr int BST = BN + 8;      // B padding 后的行距
// warp tile 大小: 64 x 32
constexpr int WM = 64,  WN = 32;
// block 在 M 方向排几个 warp(2)、在 N 方向排几个 warp(4)
constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
// 每个 block 的线程数 = warp 数 x 32(= 8 x 32 = 256)
constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
// MF: 一个 warp 在 M 方向要几个 16；NF: N 方向要几个 8；KK: 每个 BK 里有多少个 16
constexpr int MF = WM / 16, NF = WN / 8, KK = BK / 16;
// 流水线级数(给 STAGES 起个更语义化的名字)
constexpr int NSTAGE = STAGES;
// A tile 里的 16B chunk 总数 = 每行 BK/8 个 chunk x BM 行
constexpr int ACH = BM * (BK/8);   // A tile 里有几个 16B chunk
// B tile 里的 16B chunk 总数 = 每行 BN/8 个 chunk x BK 行
constexpr int BCH = BK * (BN/8);   // B tile 里有几个 16B chunk

// 把通用指针换成 32 位 shared 地址(ldmatrix / cp.async 要的都是后者，不是通用指针)
__device__ __forceinline__ uint32_t smem_addr(const void* p){
    // __cvta_generic_to_shared 负责地址空间转换
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 发出一条 cp.async: 从 global 的 src 异步搬 16 字节到 shared 的 dst
__device__ __forceinline__ void cp_async16(void* dst, const void* src){
    // .cg = 绕过 L1 只走 L2(16B 大粒度的标准写法)；%0 是目的 shared 地址(32位)，%1 是源指针(64位)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr(dst)), "l"(src));
}
// 提交: 把"自上一次 commit 以来发出的所有 cp.async"打包成一个 group
__device__ __forceinline__ void cp_commit(){
    // 只有 commit 过的搬运才会被 wait_group 计数/等待
    asm volatile("cp.async.commit_group;\n");
}
// 等待: 允许最多 N 个 group 尚未完成(即等到"未完成组数 <= N")
// N 必须是编译期常量，所以做成模板参数(PTX 要求立即数)
template<int N> __device__ __forceinline__ void cp_wait(){
    // "n"(N) 把 N 当成立即数写进指令
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
// 一条 m16n8k16 张量核 mma: D = A*B + C(f16 输入、f32 累加)
__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4], const uint32_t b[2]){
    // 下面 5 行是同一个 asm 语句: 操作数顺序为 {d},{a},{b},{c}，且 d 与 c 同寄存器(in-out)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// ldmatrix x4(非转置): 一次装 4 个 8x8 的 b16 子块 = 一个 16x16，产出 4 个 32 位寄存器
__device__ __forceinline__ void ldm_x4(uint32_t r[4], uint32_t addr){
    // 每个 lane 提供的地址是"自己那一行的行首(16B)"
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// ldmatrix x2(转置): 一次装 2 个 8x8 b16 子块并转置，产出 2 个 32 位寄存器 —— 专给 B 用
__device__ __forceinline__ void ldm_x2_trans(uint32_t r[2], uint32_t addr){
    // mma 要的 B fragment 两个元素在内存里隔一整行，只有 .trans 才装得出来
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// 主 kernel 声明；__launch_bounds__(NTHREADS) 告诉编译器本 kernel 固定用 NTHREADS 个线程，便于它控寄存器
__global__ void __launch_bounds__(NTHREADS)
// 参数: A(M x K 行主序)、B(K x N 行主序)、C(M x N 行主序)、以及三个维度
gemm_tc(const __half* __restrict__ A, const __half* __restrict__ B,
        float* __restrict__ C, int M, int N, int K){
    // 多级 buffer 的大小随 STAGES 变，静态数组写不出来，所以用动态 shared memory
    extern __shared__ __align__(16) __half smem[];
    // A 的所有 stage 排在动态 shared 最前面: NSTAGE 块，每块 BM*AST 个 half
    __half* As = smem;                              // NSTAGE * BM * AST
    // B 的所有 stage 紧跟其后: NSTAGE 块，每块 BK*BST 个 half
    __half* Bs = smem + (size_t)NSTAGE * BM * AST;  // NSTAGE * BK * BST

    // 本线程在 warp 内的编号 0..31
    const int lane  = threadIdx.x & 31;
    // 本线程属于 block 内的第几个 warp
    const int warp  = threadIdx.x >> 5;
    // lane>>2 (0..7): 控制 mma fragment 的"行"
    const int group = lane >> 2;
    // lane&3  (0..3): 控制 mma fragment 的"列"
    const int tid   = lane & 3;
    // 本 warp 负责 block 的哪 64 行
    const int warp_m = warp % WARPS_M;
    // 本 warp 负责 block 的哪 32 列
    const int warp_n = warp / WARPS_M;

    // 寄存器累加器: MF x NF 个 fragment，每个 fragment 4 个 float(= 共 64 个寄存器)
    float acc[MF][NF][4];
    // 三重循环把累加器清零
    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf)
        for (int i=0;i<4;++i) acc[mf][nf][i] = 0.f;

    // 本 block 在 A 中的行起点: 第 blockIdx.y 个 128 行块
    const __half* Ag = A + (size_t)blockIdx.y * BM * K;
    // 本 block 在 B 中的列起点: 第 blockIdx.x 个 128 列块
    const __half* Bg = B + (size_t)blockIdx.x * BN;

    // issue(st, k0): 把第 k0 个 K 块(共 BK 列)的 A/B 异步搬进第 st 级 buffer。
    // 注意循环按 16B chunk 计数，且行内偏移要用 padding 后的行距 AST / BST。
    auto issue = [&](int st, int k0){
        // 本 stage 的 A 基址
        __half* Asb = As + (size_t)st * BM * AST;
        // 本 stage 的 B 基址
        __half* Bsb = Bs + (size_t)st * BK * BST;
        // 全 block 协作搬 A 的 ACH 个 chunk
        for (int i = threadIdx.x; i < ACH; i += NTHREADS){
            // chunk i 落在第 r 行、行内第 c 列(c 是 8 的倍数)
            const int r = i / (BK/8), c = (i % (BK/8)) * 8;
            // 从 global 异步搬到 shared(行距用 AST)
            cp_async16(&Asb[r*AST + c], Ag + (size_t)r*K + k0 + c);
        }
        // 全 block 协作搬 B 的 BCH 个 chunk
        for (int i = threadIdx.x; i < BCH; i += NTHREADS){
            // chunk i 落在第 r 行、行内第 c 列
            const int r = i / (BN/8), c = (i % (BN/8)) * 8;
            // 从 global 异步搬到 shared(行距用 BST)
            cp_async16(&Bsb[r*BST + c], Bg + (size_t)(k0 + r)*N + c);
        }
    };

    // ===== TODO 1: 序言要预取几级? =====
    // 循环里每轮只用掉一级、同时补进一级，所以"在飞的"永远有 NSTAGE-1 级；
    // 序言就是把最前面的这 NSTAGE-1 级先填满。
    // 逐级预取第 0 .. NSTAGE-2 个 K 块
    for (int s = 0; s < NSTAGE - 1; ++s){
        // 该 K 块没越界才真正发出搬运
        if (s*BK < K) issue(s, s*BK);
        // 即使越界没搬，也必须 commit，否则后面的 wait_group 组数会错位
        cp_commit();          // 即使越界没搬，也要 commit，否则后面的 wait_group 计数会错
    }

    // 当前正在计算(即将计算)的 buffer 级号，从 0 开始
    int st = 0;
    // K 方向主循环: 每轮推进 BK 列
    for (int k0 = 0; k0 < K; k0 += BK){
        // ===== TODO 2: 这里该 wait_group 几? =====
        // 我们只要求"轮到要算的那一级已经就绪"，允许还有几级在路上。
        // 在飞的一共 NSTAGE-1 级: 最新的一级是"下一轮"，最老的一级是"本轮"，
        // 所以允许还在飞的组数 = NSTAGE-2。
        // 等到"未完成组数 <= NSTAGE-2"，本轮 buffer 即已就绪
        cp_wait<NSTAGE - 2>();
        // 让全 block 都看到刚搬进来的数据(同时保证上一轮算完的 buffer 已无人读)
        __syncthreads();

        // 本轮要计算的 A stage 基址
        const __half* Asb = As + (size_t)st * BM * AST;
        // 本轮要计算的 B stage 基址
        const __half* Bsb = Bs + (size_t)st * BK * BST;
        // 在 BK 内按 16 列一组，共做 KK 轮
        for (int kk = 0; kk < KK; ++kk){
            // A 的 fragment: MF 个，每个 4 个 uint32
            uint32_t a[MF][4];
            // 逐个 M 方向 fragment 装 A
            for (int mf=0; mf<MF; ++mf){
                // ldmatrix 的地址角色: 子块首行 + 象限行偏移 + 子块内行号
                const int a_row = warp_m * WM + mf * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                // 列: 先定位到本次 kk 对应的 16 列，再按 lane>>4 选左/右 8 列
                const int a_col = kk * 16 + (lane >> 4) * 8;
                // x4 非转置: 一次装 4 个 8x8 子块 = 这个 16x16 的四个象限(行距 AST)
                ldm_x4(a[mf], smem_addr(&Asb[a_row * AST + a_col]));
            }
            // B 的 fragment: NF 个，每个 2 个 uint32
            uint32_t b[NF][2];
            // 逐个 N 方向 fragment 装 B
            for (int nf=0; nf<NF; ++nf){
                // B 的行: 本次 kk 的 16 行 + lane&15(x2 只用 lane0-15，16-31 折回后重复)
                const int b_row = kk * 16 + (lane & 15);
                // B 的列: 本 warp 负责的 32 列里第 nf 个 8 列块
                const int b_col = warp_n * WN + nf * 8;
                // x2 转置: mma 要的 B fragment 两元素隔一整行，必须用 .trans(行距 BST)
                ldm_x2_trans(b[nf], smem_addr(&Bsb[b_row * BST + b_col]));
            }
            // MF x NF 条 mma，全部累加进寄存器 acc
            for (int mf=0; mf<MF; ++mf)
              for (int nf=0; nf<NF; ++nf)
                mma_m16n8k16(acc[mf][nf], a[mf], b[nf]);
        }

        // ===== TODO 3: 预取写回哪一级 buffer? =====
        // 本轮算的是第 st 级；要预取的是"本轮 + NSTAGE-1"那一轮，
        // 而那一轮该用的 buffer = 上一轮算完的那块 = 第 (st+NSTAGE-1)%NSTAGE 级。
        // 要预取的 K 块起点(本轮往后数 NSTAGE-1 个 BK)
        const int nk  = k0 + (NSTAGE - 1) * BK;
        // 要写回的 buffer 级号(环形)
        const int nst = (st + NSTAGE - 1) % NSTAGE;
        // 该块没越界才真正发出搬运
        if (nk < K) issue(nst, nk);
        // 提交本轮的 group(同样: 越界没搬也要 commit)
        cp_commit();
        // 级号前进一格(环形回绕)
        st = (st + 1) % NSTAGE;
    }

    // 写回 C: 遍历本 warp 的每个 16x8 小块
    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf){
        // 该小块在 C 中的全局行首(fragment 内行偏移是 group；+8 表示后半)
        const int grow = blockIdx.y*BM + warp_m*WM + mf*16 + group;
        // 该小块在 C 中的全局列首(fragment 内列偏移是 2*tid；+1 表示后半)
        const int gcol = blockIdx.x*BN + warp_n*WN + nf*8 + 2*tid;
        // acc[0] -> (grow, gcol)
        C[(size_t)grow*N + gcol]         = acc[mf][nf][0];
        // acc[1] -> (grow, gcol+1)
        C[(size_t)grow*N + gcol + 1]     = acc[mf][nf][1];
        // acc[2] -> (grow+8, gcol)
        C[(size_t)(grow+8)*N + gcol]     = acc[mf][nf][2];
        // acc[3] -> (grow+8, gcol+1)
        C[(size_t)(grow+8)*N + gcol + 1] = acc[mf][nf][3];
      }
}

// ================== host 侧：多了"动态 shared 要显式申请上限"这一步 ==================

// 每个 block 需要多少动态 shared 字节 = NSTAGE 份 (A 的 BM*AST + B 的 BK*BST) 个 half
static const size_t SMEM = (size_t)NSTAGE * (BM*AST + BK*BST) * 2;

// CPU 参考实现: 用 double 累加(输入是同一份 fp16)，所以差异只应来自累加顺序
static void cpu_ref(const std::vector<__half>& hA, const std::vector<__half>& hB,
                    std::vector<float>& ref, int M, int N, int K){
    // 遍历输出的每一行 i 与每一列 n
    for (int i=0;i<M;i++) for (int n=0;n<N;n++){
        // double 累加器
        double s = 0;
        // K 方向累加
        for (int k=0;k<K;k++)
            // 两个 fp16 先转 float 再转 double 相乘，保精度
            s += (double)__half2float(hA[(size_t)i*K+k]) * (double)__half2float(hB[(size_t)k*N+n]);
        // 结果存回 fp32
        ref[(size_t)i*N+n] = (float)s;
    }
}

// 正确性校验: 造随机数据，与 cpu_ref 对拍
static int check(int M, int N, int K){
    // host 侧 A / B
    std::vector<__half> hA((size_t)M*K), hB((size_t)K*N);
    // 固定随机种子，保证每次可复现
    srand(1234);
    // A 填 [-1,1) 的随机 fp16
    for (auto& x : hA) x = __float2half((float)(rand()/(double)RAND_MAX*2.0-1.0));
    // B 填 [-1,1) 的随机 fp16
    for (auto& x : hB) x = __float2half((float)(rand()/(double)RAND_MAX*2.0-1.0));
    // ref = CPU 参考结果, got = GPU 结果
    std::vector<float> ref((size_t)M*N), got((size_t)M*N, 0.f);
    // 先算参考
    cpu_ref(hA,hB,ref,M,N,K);

    // 动态 shared 超过 48KB 时必须显式申请，否则 launch 直接报 invalid argument
    CK(cudaFuncSetAttribute(gemm_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));

    // device 侧指针
    __half *dA,*dB; float *dC;
    // 分配 A
    CK(cudaMalloc(&dA,(size_t)M*K*sizeof(__half)));
    // 分配 B
    CK(cudaMalloc(&dB,(size_t)K*N*sizeof(__half)));
    // 分配 C
    CK(cudaMalloc(&dC,(size_t)M*N*sizeof(float)));
    // 拷 A 到 device
    CK(cudaMemcpy(dA,hA.data(),(size_t)M*K*sizeof(__half),cudaMemcpyHostToDevice));
    // 拷 B 到 device
    CK(cudaMemcpy(dB,hB.data(),(size_t)K*N*sizeof(__half),cudaMemcpyHostToDevice));
    // grid: x 方向 N/BN 个、y 方向 M/BM 个 block
    dim3 grid(N/BN, M/BM);
    // 启动 kernel(第 3 个参数 = 动态 shared 字节数)
    gemm_tc<<<grid, NTHREADS, SMEM>>>(dA,dB,dC,M,N,K);
    // 检查启动是否报错
    CK(cudaGetLastError());
    // 等 kernel 跑完
    CK(cudaDeviceSynchronize());
    // 把结果拷回 host
    CK(cudaMemcpy(got.data(),dC,(size_t)M*N*sizeof(float),cudaMemcpyDeviceToHost));

    // 最大绝对误差、失败计数
    double maxAbs=0; int bad=0;
    // 逐元素比对
    for (size_t i=0;i<(size_t)M*N;i++){
        // 更新最大绝对误差
        maxAbs = std::max(maxAbs, (double)fabs(got[i]-ref[i]));
        // 超出容差(绝对 1e-3 + 相对 1e-3)就记一次失败
        if (fabs(got[i]-ref[i]) > 1e-3 + 1e-3*fabs(ref[i])) bad++;
    }
    // 打印校验结果
    printf("[check] %dx%dx%d  max|abs diff| = %.3g   mismatched = %d / %d  -> %s\n",
           M,N,K, maxAbs, bad, M*N, bad==0?"PASS":"FAIL");
    // 释放 device 内存
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    // 返回码: 全对返回 0
    return bad==0?0:1;
}

// 计时: warmup 次预热后，用 cudaEvent 测 iters 次，报告 median
static void bench(int M, int N, int K, int warmup, int iters){
    // 同样要先申请动态 shared 上限
    CK(cudaFuncSetAttribute(gemm_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
    // device 指针
    __half *dA,*dB; float *dC;
    // 分配 A / B / C
    CK(cudaMalloc(&dA,(size_t)M*K*sizeof(__half)));
    CK(cudaMalloc(&dB,(size_t)K*N*sizeof(__half)));
    CK(cudaMalloc(&dC,(size_t)M*N*sizeof(float)));
    // 输入清零(只测性能，不校验正确性)
    CK(cudaMemset(dA,0,(size_t)M*K*sizeof(__half)));
    CK(cudaMemset(dB,0,(size_t)K*N*sizeof(__half)));
    // grid
    dim3 grid(N/BN, M/BM);
    // 预热若干次
    for (int i=0;i<warmup;i++) gemm_tc<<<grid,NTHREADS,SMEM>>>(dA,dB,dC,M,N,K);
    // 等预热结束
    CK(cudaDeviceSynchronize());
    // 计时用的 event 与耗时记录
    std::vector<double> t; cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // 逐次计时
    for (int i=0;i<iters;i++){
        // 记录起点
        cudaEventRecord(e0);
        // 跑一次 kernel
        gemm_tc<<<grid,NTHREADS,SMEM>>>(dA,dB,dC,M,N,K);
        // 记录终点并同步
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        // 取本次耗时(ms)存入
        float ms; cudaEventElapsedTime(&ms,e0,e1); t.push_back(ms);
    }
    // 排序
    std::sort(t.begin(), t.end());
    // 取中位数
    double med = t[t.size()/2];
    // 计算浮点运算量
    double flops = 2.0*M*N*K;
    // 打印结果(含 TFLOPS)
    printf("[bench] BK=%d stages=%d smem=%zu B  %dx%dx%d  median %.3f ms  ->  %.2f TFLOPS\n",
           BK, NSTAGE, SMEM, M,N,K, med, flops/(med*1e-3)/1e12);
    // 释放
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// 程序入口
int main(){
    // 打印本次编译使用的配置(BK / 级数 / shared 字节数)
    printf("BK=%d  stages=%d  smem=%zu B (%zu KB)\n", BK, NSTAGE, SMEM, SMEM/1024);
    // 校验 256^3
    int rc = check(256,256,256);
    // 再校验 512^3
    rc |= check(512,512,512);
    // 2048^3 计时
    bench(2048,2048,2048, 10, 50);
    // 返回(任一校验失败则非 0)
    return rc;
}

// ---- Step 5 的实验表（用 -DBKV=.. -DSTAGES=.. 切换） ----
//   nvcc -O3 -arch=native -DBKV=16 -DSTAGES=3 V5.cu -o V5.exe
// 参考解实测（2048^3）:
//   BK=16 stages=2 -> 68.5     BK=16 stages=3 -> 77.5   BK=16 stages=4 -> 76.7
//   BK=16 stages=5 -> 68.0     BK=16 stages=6 -> 66.9
//   BK=32 stages=3 -> 68.4     BK=32 stages=4 -> 68.1
//   BK=64 stages=2 -> 60.9     BK=64 stages=3 -> 超 smem 上限，跑不起来
// 注意 stages 一多反而变慢（shared 变大 -> 每个 SM 能放的 block 变少 -> 并行度下降）。
