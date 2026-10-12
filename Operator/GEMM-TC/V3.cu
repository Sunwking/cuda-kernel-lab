// Operator/GEMM-TC/V3.cu
// Step 4: 消 shared memory 的 bank conflict —— 用 padding，把行距错开
//
// 前置: V2.cu(46.19 TFLOPS)。本步只改 shared memory 的"摆放方式"，算法一行不变。
//
// 编译: nvcc -O3 -arch=native -Xptxas -v V3.cu -o V3.exe
//
// ---- 先把 bank conflict 的账算清楚 ----
// shared memory = 32 个 bank x 4 字节 = 一行 128B。一次 ldmatrix 里 8 个 lane 各读 16B，
// 而 16B 正好横跨 4 个 bank。所以理想情况: 8 个 lane 的 16B 窗口互不重叠 = 正好铺满 32 个 bank。
// "行 r 落在哪个 bank" = (r * 行距字节数 / 4) mod 32。
//
// A tile: 行距 BK*2 = 32B -> 每行错开 8 个 bank -> 行 0..7 落在 0,8,16,24,0,8,16,24
//         -> 0 和 4 撞、1 和 5 撞… = **2 路冲突**
// B tile: 行距 BN*2 = 256B -> 每行错开 64 个 bank, 64 mod 32 = 0
//         -> 8 行全从 bank 0 开始 = **8 路冲突**（B 才是大头)
//
// 修法: 给行距加一点 padding，让"每个 lane 的 16B 窗口"正好铺满 32 个 bank。
//   目标: 8 个行起点在 bank 上的间隔 = 4(或 4 的奇数倍)，这样窗口依次是 [0-3],[4-7],…,[28-31]。
//
// ---- 另外顺手把 K 块也做成可调的 ----
// 顶部的 BK 可以直接改成 64 再编一次，这就是 Step 4 的实验(见文件末尾的说明)。
// BK 变大后，一个 K 迭代里要连做 BK/16 次 mma，所以多了一层 kk 循环。

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
constexpr int BK = 64;        // ← 实验: 改成 64 再编一次，对比数字

// ===== TODO 1: padding 后的行距(2 处标红) =====
// A tile 每行有 BK 个 half = BK*2 字节，不做 padding 时行距就是 BK。
// 要让"行起点在 bank 上的间隔"变成 4，行距该是多少 half?
//   (A) BK=16 时: 行距 X 满足 (X*2/4) mod 32 在 8 行上依次错开 4 -> X = ?
//   (B) B tile 每行 BN 个 half，同理 -> 行距 Y = ?
// 提示: 两个都是 "原行距 + 8" 这个形式(正好是一个 16B chunk 的宽度)。
constexpr int AST = BK + 8;      // A tile 的行距(half 数)
constexpr int BST = BN + 8;      // B tile 的行距(half 数)

constexpr int WM = 64,  WN = 32;
constexpr int WARPS_M = BM / WM;              // 2
constexpr int WARPS_N = BN / WN;              // 4
constexpr int NTHREADS = WARPS_M * WARPS_N * 32;   // 256
constexpr int MF = WM / 16;                   // 4
constexpr int NF = WN / 8;                    // 4
constexpr int KK = BK / 16;                   // 一个 K 块里有几次 mma 的 k

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

    // 注意尺寸用的是 AST / BST，不是 BK / BN
    __shared__ __align__(16) __half As[BM * AST];
    __shared__ __align__(16) __half Bs[BK * BST];

    float acc[MF][NF][4];
    for (int mf=0; mf<MF; ++mf)
      for (int nf=0; nf<NF; ++nf)
        for (int i=0;i<4;++i) acc[mf][nf][i] = 0.f;

    const __half* Ag = A + (size_t)blockIdx.y * BM * K;
    const __half* Bg = B + (size_t)blockIdx.x * BN;

    for (int k0 = 0; k0 < K; k0 += BK){
        // 搬 tile 时，逻辑位置 (r,c) 要写到物理位置 r*行距 + c —— 这里就是 padding 生效的地方。
        // 下面这两行已经帮你写好了(用的是 AST/BST)。
        for (int i = threadIdx.x; i < BM*BK; i += NTHREADS){
            const int r = i / BK, c = i % BK;
            As[r*AST + c] = Ag[(size_t)r*K + k0 + c];
        }
        for (int i = threadIdx.x; i < BK*BN; i += NTHREADS){
            const int r = i / BN, c = i % BN;
            Bs[r*BST + c] = Bg[(size_t)(k0 + r)*N + c];
        }
        __syncthreads();

        for (int kk = 0; kk < KK; ++kk){
            // ===== TODO 2: ldmatrix 地址里的行距(2 处标红)=====
            // 逻辑上 (行, 列) 没变，但物理下标要用 r*行距 + 列。把两个 ??? 换成 TODO 1 的两个常量。
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
                ldm_x2_trans(b[nf], smem_addr(&Bs[b_row*BST + b_col]));
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

// ================== 机械部分（与 V2 相同） ==================

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
    printf("[bench] BK=%d  %dx%dx%d  median %.3f ms  ->  %.2f TFLOPS\n",
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

// ---- Step 4 的实验（做完填进 README 的对照表） ----
// 1. 当前 BK=16 + padding:  预期 ~50 TFLOPS
// 2. 把 BK 改成 64 再编:     预期 ~58 TFLOPS
// 3. 回到 BK=64，把 AST/BST 改回 BK/BN(即去掉 padding) 再编: 预期 ~39 TFLOPS
//    —— 第 3 步是关键: 说明 BK 不能单独调，tile 一变宽，bank conflict 会先把你吃掉。
// 4. 把三组数字和目标值都记进 README，然后进 Step 5（cp.async 流水线）。