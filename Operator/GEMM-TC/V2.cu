// Operator/GEMM-TC/V2.cu
// Step 3: 真正的 tiling 框架 —— block 128x128, warp 64x32, K 方向按 16 分块
//
// 前置: V0(单 warap mma 的 fragment layout) + V1(ldmatrix 装 fragment)。本步把它们接起来。
//
// 编译: nvcc -O3 -arch=native V2.cu -o V2.exe
//       加 -Xptxas -v 可以看寄存器/共享内存；加 -Xcompiler "/utf-8" 可免去 BOM 的麻烦。
//
// 结构:
//   grid  = (N/BN, M/BM)   —— 每个 block 算 C 的一个 128x128 小块
//   block = 8 个 warp (256 线程)，排成 2(行) x 4(列)，每个 warp 算 64x32
//   K 方向不切 grid，在 block 内部循环，每轮吃 16 个 k
//
// 每轮 K 循环:
//   1. 全 block 协作把 A 的 128x16 和 B 的 16x128 搬进 shared memory
//   2. __syncthreads()
//   3. 每个 warp 用 ldmatrix 装自己的 fragment: 4 个 A fragment(x4) + 4 个 B fragment(x2.trans)
//   4. 4x4 = 16 条 mma 累加进寄存器
//   5. __syncthreads()  —— 防止下一轮覆盖 shared memory 时还有 warp 在读
//
// 一个 warp 为什么要 4x4 个 fragment:
//   warp tile 64x32，而一条 mma 算 16x8 —— 64/16 = 4 个 M 方向、32/8 = 4 个 N 方向。

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

// ---- tile 尺寸 ----
constexpr int BM = 128, BN = 128, BK = 16;
constexpr int WM = 64,  WN = 32;
constexpr int WARPS_M = BM / WM;              // 2
constexpr int WARPS_N = BN / WN;              // 4
constexpr int NTHREADS = WARPS_M * WARPS_N * 32;   // 256
constexpr int MF = WM / 16;                   // 4 个 M 方向 fragment
constexpr int NF = WN / 8;                    // 4 个 N 方向 fragment

__device__ __forceinline__ uint32_t smem_addr(const void* p){
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
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
gemm_tc(const __half* __restrict__ A,   // M x K, row-major
        const __half* __restrict__ B,   // K x N, row-major
        float* __restrict__ C,          // M x N, row-major
        int M, int N, int K){
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int group = lane >> 2;   // 0..7
    const int tid   = lane & 3;    // 0..3

    // 8 个 warp 排成 2(行) x 4(列)。拆法是"先列后行"(warp 0..3 是第一行的四列)，
    // 这样同一个 block 里连续的 warp 拿的 B fragment 相邻，shared 里读的位置也更近。
    const int warp_m = warp % WARPS_M;    // 0..1 -> 本 warp 负责 block 的哪 64 行
    const int warp_n = warp / WARPS_M;    // 0..3 -> 本 warp 负责 block 的哪 32 列

    // A tile: 128 x 16，行距 BK=16 half=32B；B tile: 16 x 128，行距 BN=128 half=256B。
    // 行首地址都是 16B 的整数倍，满足 ldmatrix 的对齐要求。
    __shared__ __align__(16) __half As[BM * BK];
    __shared__ __align__(16) __half Bs[BK * BN];

    // 每个线程的累加器: MF x NF 个 fragment，每个 fragment 4 个 float = 64 个寄存器。
    float acc[MF][NF][4];
    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf)
        for (int i=0;i<4;++i) acc[mf][nf][i] = 0.f;

    const __half* Ag = A + (size_t)blockIdx.y * BM * K;
    const __half* Bg = B + (size_t)blockIdx.x * BN;

    for (int k0 = 0; k0 < K; k0 += BK){
        // ---- 1) 搬 tile 到 shared memory（每个线程搬 BM*BK/NTHREADS = 8 个 A 元素、8 个 B 元素）----
        for (int i = threadIdx.x; i < BM*BK; i += NTHREADS){
            const int r = i / BK, c = i % BK;
            As[i] = Ag[(size_t)r*K + k0 + c];
        }
        for (int i = threadIdx.x; i < BK*BN; i += NTHREADS){
            const int r = i / BN, c = i % BN;
            Bs[i] = Bg[(size_t)(k0 + r)*N + c];
        }
        __syncthreads();

        // ===== TODO 1: A 的 ldmatrix 地址（2 处标红）=====
        // As 是 128x16。本 warp 的第 mf 个 16x16 子块，左上角在 As 的第
        //   warp_m*WM + mf*16  行、第 0 列。
        // 再套 V1 里已经验证过的 lane 规则:
        //   行 = 子块首行 + ((lane>>3)&1)*8 + (lane&7)
        //   列 = (lane>>4)*8
        // 注意: 行距是 BK(不是 16 以外的数)，因为 As 的每行就是 16 个 half。
        uint32_t a[MF][4];
        for (int mf=0; mf<MF; ++mf){
            const int a_row = warp_m * WM + mf * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
            const int a_col = (lane >> 4) * 8;
            ldm_x4(a[mf], smem_addr(&As[a_row*BK + a_col]));
        }

        // ===== TODO 2: B 的 ldmatrix 地址（1 处标红）=====
        // Bs 是 16x128，x2.trans 的两个 8x8 子块 = Bs 的第 0..7 行和第 8..15 行。
        // b_row 已经给好（lane & 15，lane 16-31 的地址被忽略但必须是合法地址）。
        // 你要填的只有"列": 本 warp 的第 nf 个 8x8 子块，左上角在 Bs 的第几列？
        //   提示: warp_n 负责的 32 列里，第 nf 个 8 列块。
        uint32_t b[NF][2];
        for (int nf=0; nf<NF; ++nf){
            const int b_row = lane & 15;
            const int b_col = warp_n * WN + nf * 8;
            ldm_x2_trans(b[nf], smem_addr(&Bs[b_row*BN + b_col]));
        }

        // ---- 3) 16 条 mma ----
        for (int mf=0; mf<MF; ++mf)
          for (int nf=0; nf<NF; ++nf)
            mma_m16n8k16(acc[mf][nf], a[mf], b[nf]);

        __syncthreads();
    }

    // ===== TODO 3: 写回 C 的全局下标（2 处标红）=====
    // acc[mf][nf] 是一个 16x8 的小块，它在 C 里的左上角是:
    //   行 = blockIdx.y*BM + warp_m*WM + mf*16
    //   列 = blockIdx.x*BN + warp_n*WN + nf*8
    // 而 fragment 内部的行列偏移和 V0 一模一样: group 管行(+8 是后半)、2*tid 管列(+1 是后半)。
    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf){
        const int grow = blockIdx.y * BM + warp_m * WM + mf * 16 + group;
        const int gcol = blockIdx.x * BN + warp_n * WN + nf * 8 + 2 * tid;
        C[(size_t)grow*N + gcol]         = acc[mf][nf][0];
        C[(size_t)grow*N + gcol + 1]     = acc[mf][nf][1];
        C[(size_t)(grow+8)*N + gcol]     = acc[mf][nf][2];
        C[(size_t)(grow+8)*N + gcol + 1] = acc[mf][nf][3];
      }
}

// ================== 以下都是机械部分（与 Step 1/2 同一套思路，只是形状变大） ==================

static void cpu_ref(const std::vector<__half>& hA, const std::vector<__half>& hB,
                    std::vector<float>& ref, int M, int N, int K){
    // 用 double 累加、输入用同一份 fp16，所以差异只来自累加顺序 -> 期望 ~1e-5
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

    __half *dA,*dB; float *dC;
    CK(cudaMalloc(&dA,(size_t)M*K*sizeof(__half)));
    CK(cudaMalloc(&dB,(size_t)K*N*sizeof(__half)));
    CK(cudaMalloc(&dC,(size_t)M*N*sizeof(float)));
    CK(cudaMemcpy(dA,hA.data(),(size_t)M*K*sizeof(__half),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB.data(),(size_t)K*N*sizeof(__half),cudaMemcpyHostToDevice));
    dim3 grid(N/BN, M/BM);
    gemm_tc<<<grid, NTHREADS>>>(dA,dB,dC,M,N,K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(),dC,(size_t)M*N*sizeof(float),cudaMemcpyDeviceToHost));

    double maxAbs=0, maxRef=0; int bad=0;
    for (size_t i=0;i<(size_t)M*N;i++){
        maxAbs = std::max(maxAbs, (double)fabs(got[i]-ref[i]));
        maxRef = std::max(maxRef, (double)fabs(ref[i]));
        if (fabs(got[i]-ref[i]) > 1e-3 + 1e-3*fabs(ref[i])) bad++;
    }
    printf("[check] %dx%dx%d  max|abs diff| = %.3g   max|ref| = %.3g   mismatched = %d / %d  -> %s\n",
           M,N,K, maxAbs, maxRef, bad, M*N, bad==0?"PASS":"FAIL");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return bad==0?0:1;
}

static double bench(int M, int N, int K, int warmup, int iters){
    __half *dA,*dB; float *dC;
    CK(cudaMalloc(&dA,(size_t)M*K*sizeof(__half)));
    CK(cudaMalloc(&dB,(size_t)K*N*sizeof(__half)));
    CK(cudaMalloc(&dC,(size_t)M*N*sizeof(float)));
    CK(cudaMemset(dA,0,(size_t)M*K*sizeof(__half)));
    CK(cudaMemset(dB,0,(size_t)K*N*sizeof(__half)));
    dim3 grid(N/BN, M/BM);
    for (int i=0;i<warmup;i++) gemm_tc<<<grid,NTHREADS>>>(dA,dB,dC,M,N,K);
    CK(cudaDeviceSynchronize());
    std::vector<double> t; cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i=0;i<iters;i++){
        cudaEventRecord(e0);
        gemm_tc<<<grid,NTHREADS>>>(dA,dB,dC,M,N,K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms,e0,e1); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    double med = t[t.size()/2];
    double flops = 2.0*M*N*K;
    printf("[bench] %dx%dx%d  median %.3f ms  ->  %.2f TFLOPS\n", M,N,K, med, flops/(med*1e-3)/1e12);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return flops/(med*1e-3)/1e12;
}

int main(){
    int rc = check(256,256,256);
    rc |= check(512,512,512);
    bench(2048,2048,2048, 10, 50);   // 参考解 0.372 ms / 46.17 TFLOPS
    bench(4096,4096,4096, 10, 50);   // 参考解 2.97 ms / 45.00 TFLOPS  
    return rc;
}