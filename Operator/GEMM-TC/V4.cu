// Operator/GEMM-TC/V4.cu
// Step 4b: 用 XOR swizzle 消 bank conflict —— 不浪费 shared memory 的那条路
//
// 前置: V3.cu(padding 版)。padding 的代价是浪费存储，swizzle 是 CUTLASS 实际走的路:
//       不改变 tile 大小，而是把"第几列"和"第几行"做异或，让 8 个 lane 的 16B 窗口错开。
//
// 编译: nvcc -O3 -arch=native -Xptxas -v V4.cu -o V4.exe
//
// ---- 为什么 A 用 padding、B 用 swizzle ----
// swizzle 的本质是"把同一个 16B chunk 的行号信息混进它的列位置"，所以它需要
// **一行里有足够多的 16B chunk** 才能错开:
//   A tile: 一行 BK=16 个 half = 32B = **2 个** 16B chunk -> 只能错开 2 种位置，不够 -> 用 padding
//   B tile: 一行 BN=128 个 half = 256B = **16 个** 16B chunk -> 足够错开 -> 用 swizzle
// 这也是为什么真实 kernel 里 BK 通常取 64 及以上: tile 够宽，swizzle 才有发挥空间。
//
// ---- B 的 swizzle 怎么设计 ----
// Bs 逻辑上是 BK 行 x BN 列(16 x 128)，每行 16 个 16B chunk。
// 逻辑位置 (k, col) 的物理位置 = k*BN + (((col>>3) ^ (k & 7)) << 3) + (col & 7)
//                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                                  把"块号"和"行号的低 3 位"异或
//
// 为什么这能消冲突: 16B chunk 的行 k 落在 bank = ((k*BN*2 + chunk*16)/4) mod 32
//   BN=128 -> k*BN*2 = 256k 字节，256k/4 = 64k，64k mod 32 = 0 —— 也就是说
//   **行号本身不改变 bank，全看 chunk 号**。异或之后，8 个 lane(行 0..7)在同一个
//   逻辑 chunk 上的物理 chunk 号变成 c^0, c^1, ..., c^7 —— 互不相同，
//   于是 8 个 16B 窗口正好铺满 32 个 bank。

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

constexpr int BM = 128, BN = 128;
constexpr int BK = 64;        // ← 实验: 改成 64 再编一次

constexpr int AST = BK + 8;   // A 仍然靠 padding（一行只有 2 个 chunk，swizzle 错不开）
constexpr int BST = BN;       // B 不 padding，靠 swizzle

constexpr int WM = 64,  WN = 32;
constexpr int WARPS_M = BM / WM;
constexpr int WARPS_N = BN / WN;
constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
constexpr int MF = WM / 16;
constexpr int NF = WN / 8;
constexpr int KK = BK / 16;

// ===== TODO 1: 搬 tile 用的 swizzle 映射（1 处标红）=====
// 逻辑位置 (k, col) -> 物理 half 下标。按上面的公式写。
__device__ __forceinline__ int swz_b(int k, int col){
    return k * BN + (((col >> 3) ^ (k & 7)) << 3) + (col & 7);
}

// ===== TODO 2: ldmatrix 用的 swizzle 映射（1 处标红）=====
// ldmatrix 的每个 lane 给的是"一行 8 个 half 的起点"，也就是某个 16B chunk 的开头，
// 所以这里 col 一定是 8 的倍数。返回这个 chunk 起点的物理 half 下标。
__device__ __forceinline__ int swz_b_chunk(int k, int col){
    return swz_b(k, col);
}

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
gemm_tc(const __half* __restrict__ A, const __half* __restrict__ B,
        float* __restrict__ C, int M, int N, int K){
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int group = lane >> 2;
    const int tid   = lane & 3;

    const int warp_m = warp % WARPS_M;
    const int warp_n = warp / WARPS_M;

    __shared__ __align__(16) __half As[BM * AST];
    __shared__ __align__(16) __half Bs[BK * BST];

    float acc[MF][NF][4];
    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf)
        for (int i=0;i<4;++i) acc[mf][nf][i] = 0.f;

    const __half* Ag = A + (size_t)blockIdx.y * BM * K;
    const __half* Bg = B + (size_t)blockIdx.x * BN;

    for (int k0 = 0; k0 < K; k0 += BK){
        for (int i = threadIdx.x; i < BM*BK; i += NTHREADS){
            const int r = i / BK, c = i - r*BK;
            As[r*AST + c] = Ag[(size_t)r*K + k0 + c];
        }
        // A 走 padding，B 走 swizzle —— 注意搬入和读出的映射必须严格一致，否则数值全错。
        for (int i = threadIdx.x; i < BK*BN; i += NTHREADS){
            const int r = i / BN, c = i - r*BN;
            Bs[swz_b(r, c)] = Bg[(size_t)(k0 + r)*N + c];
        }
        __syncthreads();

        for (int kk = 0; kk < KK; ++kk){
            uint32_t a[MF][4];
            for (int mf=0; mf<MF; ++mf){
                const int a_row = warp_m*WM + mf*16 + ((lane>>3)&1)*8 + (lane&7);
                const int a_col = kk*16 + (lane>>4)*8;
                ldm_x4(a[mf], smem_addr(&As[a_row*AST + a_col]));
            }
            uint32_t b[NF][2];
            for (int nf=0; nf<NF; ++nf){
                const int b_row = kk*16 + (lane & 15);
                const int b_col = warp_n*WN + nf*8;
                ldm_x2_trans(b[nf], smem_addr(&Bs[swz_b_chunk(b_row, b_col)]));
            }
            for (int mf=0; mf<MF; ++mf)
              for (int nf=0; nf<NF; ++nf)
                mma_m16n8k16(acc[mf][nf], a[mf], b[nf]);
        }
        __syncthreads();
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

// ================== 机械部分（与 V3 相同） ==================

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
    printf("[bench] BK=%d B=swizzle  %dx%dx%d  median %.3f ms  ->  %.2f TFLOPS\n",
           BK, M,N,K, med, flops/(med*1e-3)/1e12);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return flops/(med*1e-3)/1e12;
}

int main(){
    printf("AST=%d  BST=%d  smem=%zu B\n", AST, BST, (size_t)BM*AST*2 + (size_t)BK*BST*2);
    int rc = check(256,256,256);
    rc |= check(512,512,512);
    bench(2048,2048,2048, 10, 50);
    return rc;
}