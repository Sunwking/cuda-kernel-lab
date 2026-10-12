// Operator/GEMM-TC/V5.cu
// Step 5: cp.async 多级流水线 —— 把"等 global load"的时间藏到计算后面
//
// 前置: V3.cu / V4.cu(59.2 TFLOPS)。前面每一步都在优化 shared memory 的**布局**，
//       这一步第一次优化**时序**。
//    
// 编译: nvcc -O3 -arch=native -Xptxas -v V5.cu -o V5.exe
//       顶部的 BKV / STAGES 可以改，用来跑实验表。改动后要重新编译。
//
// ---- 现在的问题 ----
// V3/V4 每轮 K 是: 搬 tile(global->寄存器->shared) -> __syncthreads -> 算 -> __syncthreads
// 搬的那段时间，Tensor Core 完全闲着。global 的延迟是几百个 cycle，而一个 K 迭代的计算
// 只有几十个 cycle，所以 SM 大部分时间在等内存。
//
// ---- cp.async 怎么解决 ----
// 1. 让 global->shared 的搬运**不经过寄存器**，一条指令搞定(省指令、省寄存器)；
// 2. 搬运是**异步**的: 发出指令后立刻返回，SM 可以继续算；
// 3. 开 NSTAGE 级 buffer 做流水: 算第 i 轮的时候，第 i+1..i+NSTAGE-1 轮已经在路上了。
//
// cp.async 的三个配套指令:
//   cp.async.cg.shared.global [dst], [src], 16;   // 搬 16 字节(8 个 half)
//   cp.async.commit_group;                        // 把"刚才发出的那批"打成一个组
//   cp.async.wait_group N;                        // 等到"未完成的组数 <= N"
//
// ---- 流水线的骨架(本步要理解的核心) ----
//   序言: 先把最前面的 NSTAGE-1 轮搬进来(每轮一个 commit_group)
//   循环第 i 轮:
//       wait_group<NSTAGE-2>   // 最老的那一级已经就绪(还允许 NSTAGE-2 组在飞)
//       __syncthreads()        // 让全 block 都看到新数据
//       计算 buffer[i % NSTAGE]
//       把第 i+NSTAGE-1 轮搬进 buffer[(i+NSTAGE-1) % NSTAGE]  <- 注意这正好是"上一轮算完的"那块
//       commit_group           // 即使这一轮没有实际搬运(越界)，也要 commit，否则组数就对不上了
//
// 为什么只需要**一个** __syncthreads 而不是两个:
//   循环顶部的这个 barrier 在程序顺序上已经晚于上一轮的计算，所以它同时保证了
//   "上一轮算完的那块 buffer 现在没人读了"，可以直接覆盖。
//
// ---- 关键点: 搬运的粒度必须和 shared 布局对齐 ----
// cp.async 一次搬 16 字节 = 8 个 half。所以搬 tile 的循环按"16B chunk"来数，
// 不是按元素来数。A tile 的 chunk 数 = BM * (BK/8)，B tile = BK * (BN/8)。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

// ---- 可调配置(实验表就是改这两个) ----
#ifndef BKV
#define BKV    16        // K 块大小
#endif
#ifndef STAGES
#define STAGES 3         // 流水线级数
#endif

constexpr int BM = 128, BN = 128, BK = BKV;
constexpr int AST = BK + 8;      // A padding 后的行距
constexpr int BST = BN + 8;      // B padding 后的行距
constexpr int WM = 64,  WN = 32;
constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
constexpr int MF = WM / 16, NF = WN / 8, KK = BK / 16;
constexpr int NSTAGE = STAGES;
constexpr int ACH = BM * (BK/8);   // A tile 里有几个 16B chunk
constexpr int BCH = BK * (BN/8);   // B tile 里有几个 16B chunk

__device__ __forceinline__ uint32_t smem_addr(const void* p){
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src){
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit(){
    asm volatile("cp.async.commit_group;\n");
}
template<int N> __device__ __forceinline__ void cp_wait(){
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4], const uint32_t b[2]){
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldm_x4(uint32_t r[4], uint32_t addr){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldm_x2_trans(uint32_t r[2], uint32_t addr){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__global__ void __launch_bounds__(NTHREADS)
gemm_tc(const __half* __restrict__ A, const __half* __restrict__ B,
        float* __restrict__ C, int M, int N, int K){
    // 多级 buffer 改成了动态 shared memory: 级数一变，大小就变，静态数组写不出来。
    extern __shared__ __align__(16) __half smem[];
    __half* As = smem;                              // NSTAGE * BM * AST
    __half* Bs = smem + (size_t)NSTAGE * BM * AST;  // NSTAGE * BK * BST

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int group = lane >> 2;
    const int tid   = lane & 3;
    const int warp_m = warp % WARPS_M;
    const int warp_n = warp / WARPS_M;

    float acc[MF][NF][4];
    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf)
        for (int i=0;i<4;++i) acc[mf][nf][i] = 0.f;

    const __half* Ag = A + (size_t)blockIdx.y * BM * K;
    const __half* Bg = B + (size_t)blockIdx.x * BN;

    // 把第 k0 个 K 块的 A/B 异步搬进第 st 级 buffer。
    // 注意循环按 16B chunk 数，且行内偏移按 padding 后的行距走。
    auto issue = [&](int st, int k0){
        __half* Asb = As + (size_t)st * BM * AST;
        __half* Bsb = Bs + (size_t)st * BK * BST;
        for (int i = threadIdx.x; i < ACH; i += NTHREADS){
            const int r = i / (BK/8), c = (i % (BK/8)) * 8;
            cp_async16(&Asb[r*AST + c], Ag + (size_t)r*K + k0 + c);
        }
        for (int i = threadIdx.x; i < BCH; i += NTHREADS){
            const int r = i / (BN/8), c = (i % (BN/8)) * 8;
            cp_async16(&Bsb[r*BST + c], Bg + (size_t)(k0 + r)*N + c);
        }
    };

    // ===== TODO 1: 序言要预取几级?(1 处标红) =====
    // 循环里每轮只用掉一级 buffer、同时补进一级，所以"在飞的"永远有 NSTAGE-1 级。
    // 序言就是把这 NSTAGE-1 级先填满。
    for (int s = 0; s < NSTAGE - 1; ++s){
        if (s*BK < K) issue(s, s*BK);
        cp_commit();          // 即使越界没搬，也要 commit，否则后面的 wait_group 计数会错
    }

    int st = 0;
    for (int k0 = 0; k0 < K; k0 += BK){
        // ===== TODO 2: 这里该 wait_group 几?(1 处标红) =====
        // 我们只要求"轮到要算的那一级已经就绪"，允许还有几级在路上。
        // 在飞的一共 NSTAGE-1 级，最新的一级是"下一轮",最老的一级是"本轮"。
        // 所以允许在飞的组数 = ?
        cp_wait<NSTAGE - 2>();
        __syncthreads();

        const __half* Asb = As + (size_t)st * BM * AST;
        const __half* Bsb = Bs + (size_t)st * BK * BST;
        for (int kk = 0; kk < KK; ++kk){
            uint32_t a[MF][4];
            for (int mf=0; mf<MF; ++mf){
                const int a_row = warp_m * WM + mf * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int a_col = kk * 16 + (lane >> 4) * 8;
                ldm_x4(a[mf], smem_addr(&Asb[a_row * AST + a_col]));
            }
            uint32_t b[NF][2];
            for (int nf=0; nf<NF; ++nf){
                const int b_row = kk * 16 + (lane & 15);
                const int b_col = warp_n * WN + nf * 8;
                ldm_x2_trans(b[nf], smem_addr(&Bsb[b_row * BST + b_col]));
            }
            for (int mf=0; mf<MF; ++mf)
              for (int nf=0; nf<NF; ++nf)
                mma_m16n8k16(acc[mf][nf], a[mf], b[nf]);
        }

        // ===== TODO 3: 预取写回哪一级 buffer?(1 处标红) =====
        // 本轮算的是第 st 级；要预取的是"本轮 + NSTAGE-1"那一轮。
        // 而那一轮该用的 buffer = 上一轮算完的那块 = 第 (st-1) 级(模 NSTAGE)。
        // 提示: (st + NSTAGE - 1) % NSTAGE 和 (st - 1 + NSTAGE) % NSTAGE 是同一个东西。
        const int nk  = k0 + (NSTAGE - 1) * BK;
        const int nst = (st + NSTAGE - 1) % NSTAGE;
        if (nk < K) issue(nst, nk);
        cp_commit();
        st = (st + 1) % NSTAGE;
    }

    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf){
        const int grow = blockIdx.y*BM + warp_m*WM + mf*16 + group;
        const int gcol = blockIdx.x*BN + warp_n*WN + nf*8 + 2*tid;
        C[(size_t)grow*N + gcol]         = acc[mf][nf][0];
        C[(size_t)grow*N + gcol + 1]     = acc[mf][nf][1];
        C[(size_t)(grow+8)*N + gcol]     = acc[mf][nf][2];
        C[(size_t)(grow+8)*N + gcol + 1] = acc[mf][nf][3];
      }
}

// ================== host 侧：多了"动态 shared 要显式申请上限"这一步 ==================

static const size_t SMEM = (size_t)NSTAGE * (BM*AST + BK*BST) * 2;

static void cpu_ref(const std::vector<__half>& hA, const std::vector<__half>& hB,
                    std::vector<float>& ref, int M, int N, int K){
    for (int i=0;i<M;i++) for (int n=0;n<N;n++){
        double s = 0;
        for (int k=0;k<K;k++)
            s += (double)__half2float(hA[(size_t)i*K+k]) * (double)__half2float(hB[(size_t)k*N+n]);
        ref[(size_t)i*N+n] = (float)s;
    }
}

static int check(int M, int N, int K){
    std::vector<__half> hA((size_t)M*K), hB((size_t)K*N);
    srand(1234);
    for (auto& x : hA) x = __float2half((float)(rand()/(double)RAND_MAX*2.0-1.0));
    for (auto& x : hB) x = __float2half((float)(rand()/(double)RAND_MAX*2.0-1.0));
    std::vector<float> ref((size_t)M*N), got((size_t)M*N, 0.f);
    cpu_ref(hA,hB,ref,M,N,K);

    // 动态 shared 超过 48KB 时必须显式申请，否则 launch 直接报 invalid argument
    CK(cudaFuncSetAttribute(gemm_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));

    __half *dA,*dB; float *dC;
    CK(cudaMalloc(&dA,(size_t)M*K*sizeof(__half)));
    CK(cudaMalloc(&dB,(size_t)K*N*sizeof(__half)));
    CK(cudaMalloc(&dC,(size_t)M*N*sizeof(float)));
    CK(cudaMemcpy(dA,hA.data(),(size_t)M*K*sizeof(__half),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB.data(),(size_t)K*N*sizeof(__half),cudaMemcpyHostToDevice));
    dim3 grid(N/BN, M/BM);
    gemm_tc<<<grid, NTHREADS, SMEM>>>(dA,dB,dC,M,N,K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(),dC,(size_t)M*N*sizeof(float),cudaMemcpyDeviceToHost));

    double maxAbs=0; int bad=0;
    for (size_t i=0;i<(size_t)M*N;i++){
        maxAbs = std::max(maxAbs, (double)fabs(got[i]-ref[i]));
        if (fabs(got[i]-ref[i]) > 1e-3 + 1e-3*fabs(ref[i])) bad++;
    }
    printf("[check] %dx%dx%d  max|abs diff| = %.3g   mismatched = %d / %d  -> %s\n",
           M,N,K, maxAbs, bad, M*N, bad==0?"PASS":"FAIL");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return bad==0?0:1;
}

static void bench(int M, int N, int K, int warmup, int iters){
    CK(cudaFuncSetAttribute(gemm_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
    __half *dA,*dB; float *dC;
    CK(cudaMalloc(&dA,(size_t)M*K*sizeof(__half)));
    CK(cudaMalloc(&dB,(size_t)K*N*sizeof(__half)));
    CK(cudaMalloc(&dC,(size_t)M*N*sizeof(float)));
    CK(cudaMemset(dA,0,(size_t)M*K*sizeof(__half)));
    CK(cudaMemset(dB,0,(size_t)K*N*sizeof(__half)));
    dim3 grid(N/BN, M/BM);
    for (int i=0;i<warmup;i++) gemm_tc<<<grid,NTHREADS,SMEM>>>(dA,dB,dC,M,N,K);
    CK(cudaDeviceSynchronize());
    std::vector<double> t; cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i=0;i<iters;i++){
        cudaEventRecord(e0);
        gemm_tc<<<grid,NTHREADS,SMEM>>>(dA,dB,dC,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms,e0,e1); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    double med = t[t.size()/2];
    double flops = 2.0*M*N*K;
    printf("[bench] BK=%d stages=%d smem=%zu B  %dx%dx%d  median %.3f ms  ->  %.2f TFLOPS\n",
           BK, NSTAGE, SMEM, M,N,K, med, flops/(med*1e-3)/1e12);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(){
    printf("BK=%d  stages=%d  smem=%zu B (%zu KB)\n", BK, NSTAGE, SMEM, SMEM/1024);
    int rc = check(256,256,256);
    rc |= check(512,512,512);
    bench(2048,2048,2048, 10, 50);
    bench(4096,4096,4096, 10, 50);
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