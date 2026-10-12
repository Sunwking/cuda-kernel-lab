// Operator/GEMM-TC/V6.cu
// Step 6: 任意 shape 的边界处理 + split-K
//
// 前置: V5b.cu(BK=16 / stages=3 -> 78.0 TFLOPS @2048^3)
//
// 编译: nvcc -O3 -arch=native -Xptxas -v V6.cu -o V6.exe
//       V6.exe 0  -> 边界正确性 (129x130x131 / 257x385x1000 / 128x128x2048)
//       V6.exe 1  -> 2048^3            (已经喂饱机器的 shape)
//       V6.exe 2  -> 256x256x32768     (瘦长 shape, split-K 的主场)
//       加 -DSPLITKV=2/4/8/16 切换 split-K 段数(默认 1 = 不开)
//
// ---- 第一件事: 边界 ----
// 前 5(b) 步都假设 M/N 是 128 的倍数、K 是 16 的倍数。这一步把假设拆掉。
// 越界的位置必须**当作 0** 参与计算(补零 zfill)：A 补在 M 方向, B 补在 K 和 N 两个方向。
// 补零之后所有 tile 都是满的, mma 那部分代码一行都不用改。
//
// 边界带来的第二个坑在**搬运**上: cp.async 要求 global 地址 16 字节对齐。
//   A[grow][gk] 的字节地址 = (grow*K + gk) * 2
// K 不是 8 的倍数时(比如 K=131), 行首 grow*K 往往不是 8 的倍数 -> 只有 2B 对齐
// -> cp.async 直接报 "misaligned address"。所以必须给搬运留一条逐元素拷贝的退路。
//
// ---- 第二件事: split-K ----
// grid 只有 ceil(M/128) * ceil(N/128) 个 block。M/N 很小时(256x256 -> 4 个 block)
// 70 个 SM 大部分在睡觉, 这时候算力是"块数"限制的, 跟 kernel 写得好不好没关系。
// 解法: 把 K 方向也切开, grid.z = SPLITK, 每个 block 算一段部分和, 最后累加。
// 代价: 累加要用 atomicAdd(多个 block 写同一个 C 元素), C 是 fp32 且要 read-modify-write,
//       于是 C 的访存流量被放大 SPLITK 倍。瘦长 shape 赚, 已喂饱的 shape 赔。

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

#ifndef BKV
#define BKV    16
#endif
#ifndef STAGES
#define STAGES 3
#endif
#ifndef SPLITKV
#define SPLITKV 1        // K 方向切几段。1 = 不切
#endif
#ifndef MINB
// __launch_bounds__ 的第二个参数 = 承诺「每个 SM 至少驻留几个 block」。
// 它只能压寄存器(65536/(MINB*256)), 管不了 smem。stages<=4 时 smem 允许 2 个 -> 2;
// stages>=5 时 smem 只允许 1 个, 这时承诺 2 只会白丢寄存器(见 README Step 5b)。
#if STAGES < 5
#define MINB   2
#else
#define MINB   1
#endif
#endif

constexpr int BM = 128, BN = 128, BK = BKV;
constexpr int AST = BK + 8, BST = BN + 8;
constexpr int WM = 64,  WN = 32;
constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
constexpr int MF = WM / 16, NF = WN / 8, KK = BK / 16;
constexpr int NSTAGE = STAGES;
constexpr int SPLITK = SPLITKV;
constexpr int ACH = BM * (BK/8);
constexpr int BCH = BK * (BN/8);

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

__global__ void __launch_bounds__(NTHREADS, MINB)
gemm_tc(const __half* __restrict__ A, const __half* __restrict__ B,
        float* __restrict__ C, int M, int N, int K){
    extern __shared__ __align__(16) __half smem[];
    __half* As = smem;                              // NSTAGE * BM * AST
    __half* Bs = smem + (size_t)NSTAGE * BM * AST;  // NSTAGE * BK * BST
    const __half Z = __float2half(0.f);              // 补零用的 0

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int group = lane >> 2;
    const int tid   = lane & 3;
    const int warp_m = warp % WARPS_M;
    const int warp_n = warp / WARPS_M;

    const int row0 = blockIdx.y * BM;   // 本 block 负责的 C 行起点
    const int col0 = blockIdx.x * BN;   // 本 block 负责的 C 列起点

    // ===== TODO 1: split-K 的区间划分 (2 处标红) =====
    // 把 ktiles 个 K-tile 均分给 SPLITK 段, 本 block 是第 blockIdx.z 段, 负责 [tbeg, tend) 号 tile。
    // 两个约束:
    //   (1) 每段的起点必须是 **tile 的整数倍**(这样 kbeg 才是 BK 的倍数, tile 不会跨段);
    //   (2) SPLITK 不整除 ktiles 时, 各段长度最多差 1 个 tile。
    // 提示: 一行整数除法 (ktiles * z) / SPLITK 就能同时满足这两点; 注意先把 z 转成 int。
    const int ktiles = (K + BK - 1) / BK;
    const int tbeg = (ktiles * (int)blockIdx.z) / SPLITK;
    const int tend = (ktiles * (int)(blockIdx.z + 1)) / SPLITK;
    const int kbeg = tbeg * BK;
    const int kend = tend * BK;

    float acc[MF][NF][4];
    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf)
        for (int i=0;i<4;++i) acc[mf][nf][i] = 0.f;

    auto issue = [&](int st, int k0){
        __half* Asb = As + (size_t)st * BM * AST;
        __half* Bsb = Bs + (size_t)st * BK * BST;
        for (int i = threadIdx.x; i < ACH; i += NTHREADS){
            const int r = i / (BK/8), c = (i % (BK/8)) * 8;
            const int grow = row0 + r, gk = k0 + c;
            __half* dst = &Asb[r*AST + c];
            // ===== TODO 3: A 的边界装载 (2 处标红) =====
            // 两种情况:
            //   (a) 这一行的这 8 个 half 全在界内, 且 global 地址 **16 字节对齐** -> cp_async16
            //   (b) 其余 -> 逐元素拷 8 个 half, 越界的位置写 Z
            // 对齐判据: cp.async 要 16B = 8 个 half, 也就是元素下标 (grow*K + gk) 必须是 8 的倍数。
            // 为什么要判它: K=131 这类 shape 下行首 grow*K 常常不是 8 的倍数 -> 只有 2B 对齐
            //              -> cp.async 报 misaligned address。
            if (grow < M && gk + 8 <= K && ((grow * K + gk) & 7) == 0) cp_async16(dst, A + (size_t)grow*K + gk);
            else for (int t=0;t<8;++t)
                dst[t] = (grow < M && gk + t < K) ? A[(size_t)grow * K + gk + t] : Z;
        }
        for (int i = threadIdx.x; i < BCH; i += NTHREADS){
            const int r = i / (BN/8), c = (i % (BN/8)) * 8;
            const int gk = k0 + r, gn = col0 + c;
            __half* dst = &Bsb[r*BST + c];
            // ===== TODO 4: B 的边界装载 (2 处标红) =====
            // B 的越界有**两个**方向: 行(gk 超过 K)和列(gn 超过 N)。对齐判据同上。
            if (gk < K && gn + 8 <= N && ((gk * N + gn) & 7) == 0) cp_async16(dst, B + (size_t)gk*N + gn);
            else for (int t=0;t<8;++t) dst[t] = (gk < K && gn + t < N) ? B[(size_t)gk * N + gn + t] : Z;
        }
    };

    // ---- 序言 ----
    for (int s = 0; s < NSTAGE-1; ++s){
        if (kbeg + s*BK < kend) issue(s, kbeg + s*BK);   // ===== TODO 2 ====
        cp_commit();
    }

    // ---- 主循环 ----
    int st = 0;
    for (int k0 = kbeg; k0 < kend; k0 += BK){            // ===== TODO 2 ====
        // ===== TODO 2: 这三处比较的右边界 (3 处标红) =====
        // 流水线的结构一个字都不用改, 只要把"全局的 K"换成"本段负责的 kend"。
        // 开头用 kbeg 而不是 0, 是上面那个 for 的初值。
        cp_wait<NSTAGE-2>();
        __syncthreads();

        const __half* Asb = As + (size_t)st * BM * AST;
        const __half* Bsb = Bs + (size_t)st * BK * BST;
        for (int kk=0; kk<KK; ++kk){
            uint32_t a[MF][4];
            for (int mf=0; mf<MF; ++mf){
                const int a_row = warp_m*WM + mf*16 + ((lane>>3)&1)*8 + (lane&7);
                const int a_col = kk*16 + (lane>>4)*8;
                ldm_x4(a[mf], smem_addr(&Asb[a_row*AST + a_col]));
            }
            uint32_t b[NF][2];
            for (int nf=0; nf<NF; ++nf){
                const int b_row = kk*16 + (lane & 15);
                const int b_col = warp_n*WN + nf*8;
                ldm_x2_trans(b[nf], smem_addr(&Bsb[b_row*BST + b_col]));
            }
            for (int mf=0; mf<MF; ++mf)
              for (int nf=0; nf<NF; ++nf)
                mma_m16n8k16(acc[mf][nf], a[mf], b[nf]);
        }

        const int nk  = k0 + (NSTAGE-1)*BK;
        const int nst = (st + NSTAGE - 1) % NSTAGE;
        if (nk < kend) issue(nst, nk);                   // ===== TODO 2 ====
        cp_commit();
        st = (st + 1) % NSTAGE;
    }

    // ---- 写回: 边界判据 + split-K 累加 ----
    // 四个累积器的 (row, col) 和 A 的 fragment 规则一致: group 管行(+8 是后半), 2*tid 管列。
    // 越界的行/列直接丢掉(不写)。
    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf){
        const int grow = row0 + warp_m*WM + mf*16 + group;
        const int gcol = col0 + warp_n*WN + nf*8 + 2*tid;
        const bool ok0 = (grow   < M) && (gcol   < N);
        const bool ok1 = (grow   < M) && (gcol+1 < N);
        const bool ok2 = (grow+8 < M) && (gcol   < N);
        const bool ok3 = (grow+8 < M) && (gcol+1 < N);
        if (SPLITK == 1){
            if (ok0) C[(size_t)grow*N + gcol]         = acc[mf][nf][0];
            if (ok1) C[(size_t)grow*N + gcol + 1]     = acc[mf][nf][1];
            if (ok2) C[(size_t)(grow+8)*N + gcol]     = acc[mf][nf][2];
            if (ok3) C[(size_t)(grow+8)*N + gcol + 1] = acc[mf][nf][3];
        } else {
            // 为什么这里必须 atomicAdd 而不能直接写:
            //   SPLITK 段 block 各自算出一份"部分和", 它们写的是**同一个** C 元素。
            //   直接写 = 互相覆盖, 最后只剩某一段的结果。
            // 它的代价: C 是 fp32 且 read-modify-write -> C 的访存流量 ×SPLITK, 而且是原子操作。
            if (ok0) atomicAdd(&C[(size_t)grow*N + gcol],         acc[mf][nf][0]);
            if (ok1) atomicAdd(&C[(size_t)grow*N + gcol + 1],     acc[mf][nf][1]);
            if (ok2) atomicAdd(&C[(size_t)(grow+8)*N + gcol],     acc[mf][nf][2]);
            if (ok3) atomicAdd(&C[(size_t)(grow+8)*N + gcol + 1], acc[mf][nf][3]);
        }
      }
}

// ==================== host 侧(基本沿用 V5b, 多了 grid.z 和 C 的边界) ====================

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

    CK(cudaFuncSetAttribute(gemm_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
    __half *dA,*dB; float *dC;
    CK(cudaMalloc(&dA,(size_t)M*K*2));
    CK(cudaMalloc(&dB,(size_t)K*N*2));
    CK(cudaMalloc(&dC,(size_t)M*N*4));
    CK(cudaMemcpy(dA,hA.data(),(size_t)M*K*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB.data(),(size_t)K*N*2,cudaMemcpyHostToDevice));
    // split-K 走 atomicAdd 累加, 所以 C 必须先清零
    CK(cudaMemset(dC,0,(size_t)M*N*4));

    // 边界向下取整变向上取整; grid.z = split-K 的段数
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM, SPLITK);
    gemm_tc<<<grid, NTHREADS, SMEM>>>(dA,dB,dC,M,N,K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(),dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));

    double maxAbs=0; int bad=0;
    for (size_t i=0;i<(size_t)M*N;i++){
        maxAbs = std::max(maxAbs, (double)fabs(got[i]-ref[i]));
        if (fabs(got[i]-ref[i]) > 1e-3 + 1e-3*fabs(ref[i])) bad++;
    }
    printf("[check] M=%d N=%d K=%d splitK=%d  max|abs diff| = %.3g  mismatched = %d / %d -> %s\n",
           M,N,K,SPLITK, maxAbs, bad, M*N, bad==0?"PASS":"FAIL");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return bad==0?0:1;
}

static void bench(int M, int N, int K, int warmup, int iters){
    CK(cudaFuncSetAttribute(gemm_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
    __half *dA,*dB; float *dC;
    CK(cudaMalloc(&dA,(size_t)M*K*2));
    CK(cudaMalloc(&dB,(size_t)K*N*2));
    CK(cudaMalloc(&dC,(size_t)M*N*4));
    CK(cudaMemset(dA,0,(size_t)M*K*2));
    CK(cudaMemset(dB,0,(size_t)K*N*2));
    CK(cudaMemset(dC,0,(size_t)M*N*4));
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM, SPLITK);
    for (int i=0;i<warmup;i++) gemm_tc<<<grid,NTHREADS,SMEM>>>(dA,dB,dC,M,N,K);
    CK(cudaDeviceSynchronize());
    std::vector<double> t; cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i=0;i<iters;i++){
        // split-K 是累加, 每次计时前要把 C 清零(不计入计时)
        if (SPLITK > 1) CK(cudaMemset(dC,0,(size_t)M*N*4));
        cudaEventRecord(e0);
        gemm_tc<<<grid,NTHREADS,SMEM>>>(dA,dB,dC,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms,e0,e1); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    double med = t[t.size()/2];
    double flops = 2.0*M*N*K;
    printf("[bench] %dx%dx%d grid=(%d,%d,%d)=%d blocks splitK=%d  median %.3f ms -> %.2f TFLOPS\n",
           M,N,K, grid.x,grid.y,grid.z, grid.x*grid.y*grid.z, SPLITK, med, flops/(med*1e-3)/1e12);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char** argv){
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    printf("V6 arbitrary shape + split-K  BK=%d stages=%d splitK=%d MINB=%d smem=%zu B\n",
           BK, NSTAGE, SPLITK, MINB, SMEM);
    if (mode == 0){
        int rc = 0;
        rc |= check(129,130,131);
        rc |= check(257,385,1000);
        rc |= check(128,128,2048);
        rc |= check(512, 512, 2048);
    } 
    bench(2048,2048,2048, 10, 50);
    bench(4096,4096,4096, 10, 50);
    bench(256,256,32768, 5, 30);
    
    return 0;
}

// ================== 验收 / 参考解实测 ==================
// 边界(全部 PASS, splitK=1/2/3/4 都过):
//   129x130x131 / 257x385x1000 / 128x128x2048
//
// 参考解实测(median):
//   256x256x32768   grid.x*grid.y = 4 个 block, K 方向再切:
//     splitK= 1 ->   4 blocks ->  4.23 TFLOPS
//     splitK= 2 ->   8 blocks ->  8.79
//     splitK= 4 ->  16 blocks -> 17.30
//     splitK= 8 ->  32 blocks -> 33.10
//     splitK=16 ->  64 blocks -> 59.39      <- 14.04 倍
//   2048x2048x2048  grid.x*grid.y = 256 个 block(已经 1.83 波, 不饿):
//     splitK= 1 -> 78.51 / 2 -> 76.64 / 4 -> 69.15 / 8 -> 55.04 / 16 -> 37.62   <- 只赔不赚
//
// 所以 split-K 不是"优化", 是一个**权衡**: 它拿 C 的访存流量(dp32 原子累加, 放大 SPLITK 倍)
// 去换 grid 的并行度。判断标准就一句话: 基础 grid 的 block 数够不够填满 SM。
//
// 另外注意寄存器: 加了边界判断之后 ptxas 会涨到 141 个(-> 1 block/SM, 2048^3 掉到 62.6),
// 所以这个文件里的 __launch_bounds__(NTHREADS, MINB) 不是装饰, 是必须的。