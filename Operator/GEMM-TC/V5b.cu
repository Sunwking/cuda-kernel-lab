// Operator/GEMM-TC/V5b.cu
// Step 5b: 寄存器双缓冲 —— 把 LDSM 提前一级发射，让 HMMA 的时间盖住它的延迟
//
// 前置: V5.cu(BK=16 / 3 stage -> 77.4 TFLOPS, 2 block/SM)
//
// 编译: nvcc -O3 -arch=native -Xptxas -v V5b.cu -o V5b.exe
//       默认 BK=16 / STAGES=3; 改 -DBKV=.. -DSTAGES=.. 跑实验表。
//
// ---- V5 还剩什么没藏起来 ----
// V5 把 global -> shared 的搬运藏到计算后面了, 但 shared -> 寄存器 (LDSM) 没藏。
// V5 每个 K 迭代的程序顺序是:
//     cp_wait -> __syncthreads -> LDSM(第 i 级) -> HMMA(第 i 级) -> issue -> commit
// 屏障后面所有 warp 一起冲出来做 LDSM, 而 HMMA 必须等 LDSM 的结果回来才能发射。
//
// ---- 这一步怎么修 ----
// 让 LDSM 提前一级: 算第 i 级的时候, 顺手把第 i+1 级的 fragment 装进**另一组**寄存器。
// 这样 HMMA(第 i 级) 用的是"一级之前就装好"的寄存器, LDSM 的延迟被 HMMA 的执行时间盖住。
//
//   V5 :  LDSM(i)  -> HMMA(i)  -> LDSM(i+1) -> HMMA(i+1)    每个 LDSM 都挡在 HMMA 前面
//   V5b:  LDSM(i+1)-> HMMA(i)  -> LDSM(i+2) -> HMMA(i+1)    HMMA 一发出去, LDSM 就在后台跑
//
// ---- 代价: 24 个寄存器, 而且必须自己钉住 occupancy ----
// fragment 要两套: a[MF][4] + b[NF][2] = 16+8 = 24, 加倍就是 +24。
// 实测 104 -> 151, 而 65536/(151*256) = 1.69 -> **只剩 1 block/SM**, 比不加还慢
// (77.4 -> 71.0)。所以 stage=3/4 上必须用 __launch_bounds__(NTHREADS, 2) 把它钉回 128 个(见文末)。
//
// ---- 顺序上的关键点(本步唯一需要"想"的地方) ----
// 要预取第 i+1 级, 前提是第 i+1 级已经搬完。而 cp.async.wait_group 数的是"还没完成的组",
// 所以必须把"发射下一级"的 issue/commit 挪到 cp_wait **之前** —— commit 组的编号整体前移
// 一位, 于是同样一个等待参数就从"保证第 i 级就绪"变成"保证第 i+1 级就绪"。
// 照抄 V5 的顺序(issue 放在计算之后)去预取下一级, 会读到还没搬完的 buffer。

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
#define BKV    16        // 本步只支持 BK=16(KK=1), 见文末说明
#endif
#ifndef STAGES
#define STAGES 3
#endif
#ifndef MINB
// __launch_bounds__ 的第二个参数 = 承诺「每个 SM 至少驻留几个 block」。
// 它只能压寄存器 (65536/(MINB*256))，管不了 smem。所以要填 smem 真正允许的那个数:
//   STAGES<=4: smem 允许 2 个 -> MINB=2
//   STAGES>=5: smem 只允许 1 个 -> 必须 MINB=1，否则白丢 22 个寄存器 (见 README)
#if STAGES < 5
#define MINB   2
#else
#define MINB   1
#endif
#endif

constexpr int BM = 128, BN = 128, BK = BKV;
constexpr int AST = BK + 8;
constexpr int BST = BN + 8;
constexpr int WM = 64,  WN = 32;
constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
constexpr int MF = WM / 16, NF = WN / 8;
constexpr int NSTAGE = STAGES;
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

// ===== 唯一和 V5 不同的地方: 第二个参数 2 = "至少要能驻留 2 个 block/SM" =====
// 少了它 ptxas 会放开用到 151 个寄存器, occupancy 掉到 1 block/SM。
__global__ void __launch_bounds__(NTHREADS, MINB)
gemm_tc(const __half* __restrict__ A, const __half* __restrict__ B,
        float* __restrict__ C, int M, int N, int K){
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

    // 把第 st 级(该级的 kk 子块)的两个 fragment 装进寄存器。
    // a 用 x4: 行 = warp 的 16 行子块 + lane 的 8 行偏移; 列 = kk*16 + (lane>>4)*8
    // b 用 x2.trans: 行 = kk*16 + (lane&15); 列 = warp 的 32 列起始 + nf*8
    auto ldsm_a = [&](uint32_t (&out)[MF][4], const __half* Asb, int kk){
        for (int mf=0; mf<MF; ++mf){
            const int a_row = warp_m*WM + mf*16 + ((lane>>3)&1)*8 + (lane&7);
            const int a_col = kk*16 + (lane>>4)*8;
            ldm_x4(out[mf], smem_addr(&Asb[a_row*AST + a_col]));
        }
    };
    auto ldsm_b = [&](uint32_t (&out)[NF][2], const __half* Bsb, int kk){
        for (int nf=0; nf<NF; ++nf){
            const int b_row = kk*16 + (lane & 15);
            const int b_col = warp_n*WN + nf*8;
            ldm_x2_trans(out[nf], smem_addr(&Bsb[b_row*BST + b_col]));
        }
    };

    // ---- 序言: 还是先填满 NSTAGE-1 级(和 V5 一样) ----
    for (int s = 0; s < NSTAGE-1; ++s){
        if (s*BK < K) issue(s, s*BK);
        cp_commit();
    }
    const int NIT = (K + BK - 1) / BK;

    // a/b  = 第 i   级的 fragment(正在算的)
    // an/bn= 第 i+1 级的 fragment(正在预取的)
    uint32_t a[MF][4], b[NF][2], an[MF][4], bn[NF][2];

    // ===== TODO 1: 进循环之前, 让"第 0 级"就绪并把它的 fragment 装进来 (1 处标红) =====
    // 循环体里我们假设 a/b 已经拿着"上一级"了, 所以第 0 级得在循环外单独装一次。
    // 装之前必须等第 0 级真的到岸。此时一共 commit 了 NSTAGE-1 组, 第 0 级是最老的一组,
    // 所以 cp_wait 的参数 = ?
    cp_wait<NSTAGE - 2>();
    __syncthreads();
    ldsm_a(a, As, 0);
    ldsm_b(b, Bs, 0);

    for (int i = 0; i < NIT; ++i){
        // ---- (a) 先发射"最远处"那一级: 本轮 + NSTAGE-1 ----
        // 注意它和 V5 一样, 但**位置提前了**(V5 里这段在计算之后)。
        const int nk  = (i + NSTAGE - 1) * BK;
        const int nst = (i + NSTAGE - 1) % NSTAGE;
        if (nk < K) issue(nst, nk);
        cp_commit();

        if (i + 1 < NIT){
            // ===== TODO 2: 这里 cp_wait 几? (1 处标红) =====
            // 现在是"先决定要预取第 i+1 级, 后 wait"。因为 (a) 刚刚多 commit 了一组,
            // 待完成的组里最老的那一组就是第 i+1 级。想让第 i+1 级完成, 允许几组在飞?
            // (提示: 别急着照抄 V5 —— 先数清楚现在"待完成"的组里, 最老的那一组是哪一级。)
            cp_wait<NSTAGE - 2>();
            __syncthreads();

            // ===== TODO 3: 预取哪一级 buffer? (2 处标红) =====
            // 本轮算的是第 i 级, 要预取的是第 i+1 级, 它在 buffer 数组里的下标 = ?
            ldsm_a(an, As + (size_t)((i + 1) % NSTAGE) * BM * AST, 0);
            ldsm_b(bn, Bs + (size_t)((i + 1) % NSTAGE) * BK * BST, 0);
        }

        // ---- (b) 算第 i 级: 用的是**上一轮**预取回来的 a/b ----
        for (int mf=0; mf<MF; ++mf)
          for (int nf=0; nf<NF; ++nf)
            mma_m16n8k16(acc[mf][nf], a[mf], b[nf]);

        // ---- (c) ping-pong: 把刚预取回来的 an/bn 扶正 ----
        #pragma unroll
        for (int mf=0; mf<MF; ++mf)
          for (int j=0;j<4;++j) a[mf][j] = an[mf][j];
        #pragma unroll
        for (int nf=0; nf<NF; ++nf)
          for (int j=0;j<2;++j) b[nf][j] = bn[nf][j];
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

// ==================== host 侧(和 V5 一样, 不用改) ====================

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
    printf("V5b reg-double-buffer BK=%d stages=%d smem=%zu B (%zu KB)\n",
           BK, NSTAGE, SMEM, SMEM/1024);
    int rc = check(256,256,256);
    rc |= check(512,512,512);
    bench(2048,2048,2048, 10, 50);
    bench(4096,4096,4096, 10, 50);
    return rc;
}

// ================== 验收 / 实验表 ==================
//    nvcc -O3 -arch=native -DSTAGES=3 V5b.cu -o V5b.exe
//    nvcc -O3 -arch=native -DSTAGES=5 -DMINB=1 V5b.cu -o V5b.exe
// 参考解实测(2048^3, median of 50)。三列 = V5 / V5b(MINB=2) / V5b(MINB=1):
//   stage=3  104 / 128 / 151 regs   2 / 2 / 1 block/SM   77.2 / 78.0 / 70.9
//   stage=4  104 / 127 / 152 regs   2 / 2 / 1 block/SM   76.7 / 77.7 / 66.8
//   stage=5  104 / 128 / 150 regs   1 / 1 / 1 block/SM   68.1 / 68.4 / 70.3
//
// 结论: 2 block/SM(16 个 warp/SM)的时候, LDSM 延迟**已经**被别的 warp 盖住了, 寄存器双缓冲
// 几乎没空间(+0.9%)。只有 occupancy 掉到 1 block/SM 时, warp 级并行藏不住延迟, 指令级并行
// 才有价值 —— 但前提是**同时放开寄存器**: MINB=1 用 150 个寄存器 -> 70.3 (+3.3%),
// MINB=2 把它钉在 128 个 -> 只剩 68.4 (+0.5%)。
//
// MINB 是"承诺"不是"愿望": stage=5 时 smem 本来就只能放 1 个 block, 你还承诺 2 个, 于是白丢
// 22 个寄存器; 反过来 stage=3/4 上写 MINB=1, 151 个寄存器直接把 occupancy 打到 1 -> 70.9。
//
// 为什么这个变体只支持 BK=16: KK=1 时"跨 K 块预取"就是"跨 kk 预取", 一份代码两用。
// BK>=32 时一块里有 KK>=2 个 kk 子块, fragment 需要 KK*MF*4 个寄存器, 那样应该改成
// **内层 kk 双缓冲**(算 kk 的时候预取 kk+1), 而不是在 K 块之间来回倒 —— 留给后面。

/*
V5b reg-double-buffer BK=16 stages=3 smem=31488 B (30 KB)
[check] 1024x1024x1024  max|abs diff| = 9.54e-05   mismatched = 0 / 1048576  -> PASS
[check] 512x512x512  max|abs diff| = 3.05e-05   mismatched = 0 / 262144  -> PASS
[bench] BK=16 stages=3 smem=31488 B  2048x2048x2048  median 0.221 ms  ->  77.73 TFLOPS
[bench] BK=16 stages=3 smem=31488 B  4096x4096x4096  median 1.509 ms  ->  91.06 TFLOPS
*/