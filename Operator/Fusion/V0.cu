/*
Residual + RMSNorm 融合  V0（标量版）

问题：未融合时是两次 global round trip
    hidden = x + residual          (add kernel:      读 2 写 1 = 12 B/元素)
    out    = rmsnorm(hidden) * g   (rmsnorm kernel:  读 1 写 1 =  8 B/元素)
                                                                 合计 20 B/元素
融合成一次之后
    读 x, 读 residual, 就地写回 residual, 写 out        =                16 B/元素

**所以这个练习的核心只有一句话**：第一遍算平方和的时候顺手把 (x+residual) 留在
寄存器里，第二遍直接用它 —— 千万不要第二遍再从 global 把 hidden 读一遍，
那样字节数和未融合一模一样，融合等于白做。

未融合的两个 kernel 已经写好，作为对照基线（别改它们）。
你要填的是 fused_add_rmsnorm_kernel 的两个循环，以及 main() 里的字节账。

编译 / 运行：
    nvcc -O3 -arch=native -Xcompiler "/utf-8" V0.cu -o V0.exe
    .\V0.exe

验收：三个 Check 都是 PASS，且打印出的 speedup 明显 > 1（理论上限见 README）。
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

#define CUDA_CHECK(call) do {                                   \
    cudaError_t e = call;                                       \
    if (e != cudaSuccess) {                                     \
        cerr << "CUDA err " << cudaGetErrorString(e) << " @" << __LINE__ << endl; \
        exit(1); } } while (0)

constexpr int BLOCK = 256;          // 一块一行
constexpr int ITEMS = 4096 / BLOCK; // 每线程负责的元素个数（编译期常量，见 README 思考题 3）

__device__ __forceinline__ float warpReduceSum(float v){
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// ===================== 未融合 1/2：hidden = x + residual =====================
__global__ void add_kernel(const float* x, const float* r, float* out, int D){
    const size_t base = (size_t)blockIdx.x * D;
    for (int i = threadIdx.x; i < D; i += BLOCK)
        out[base + i] = x[base + i] + r[base + i];
}

// ============ 未融合 2/2：单遍 RMSNorm（寄存器缓存，别改）============
template<int N_ITEMS>
__global__ void rmsnorm_cached_kernel(const float* in, const float* gamma,
                                      float* out, int D, float eps){
    __shared__ float sm[BLOCK / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t base = (size_t)blockIdx.x * D;

    float reg[N_ITEMS];
    float sum = 0.f;
    #pragma unroll
    for (int it = 0; it < N_ITEMS; ++it){
        const int i = tid + it * BLOCK;
        const float v = (i < D) ? in[base + i] : 0.f;
        reg[it] = v;
        sum += v * v;
    }
    sum = warpReduceSum(sum);
    if (lane == 0) sm[warp] = sum;
    __syncthreads();
    if (warp == 0){
        sum = (lane < BLOCK/32) ? sm[lane] : 0.f;
        sum = warpReduceSum(sum);
        if (lane == 0) sm[0] = sum;
    }
    __syncthreads();
    const float inv = rsqrtf(sm[0] / D + eps);
    #pragma unroll
    for (int it = 0; it < N_ITEMS; ++it){
        const int i = tid + it * BLOCK;
        if (i < D) out[base + i] = gamma[i] * reg[it] * inv;
    }
}

// =====================================================================
// 融合版：一个 kernel 完成 add + RMSNorm
//   读  x[i], residual[i]
//   写  residual[i]（就地更新成 hidden，下游的下一层还要把它当残差流用）
//       out[i]
// =====================================================================
template<int N_ITEMS>
__global__ void fused_add_rmsnorm_kernel(const float* x, float* residual,
                                         const float* gamma, float* out,
                                         int D, float eps){
    __shared__ float sm[BLOCK / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t base = (size_t)blockIdx.x * D;

    float reg[N_ITEMS];   // 存 hidden，给第二遍用
    float sum = 0.f;

    // ===== TODO 1: 第一遍（边写回边归约）=====
    // 这一遍要一口气做四件事，一件都不能少：
    //   (a) 读 x[base + i]
    //   (b) 读 residual[base + i]
    //   (c) 把 (a)+(b) 的**就地写回** residual[base + i]     <- 省下来的那 4 B 就在这
    //   (d) 把 (a)+(b) 累积进 sum（平方和），同时**留在 reg[it] 里**给第二遍
    //   (e) 越界保护：i >= D 时既不能读越界，也不能让垃圾值混进 sum
    //       —— reg[it] 也必须在越界时被写成一个确定值（sum 里的贡献是 0）
    // 提示：循环骨架和上面 rmsnorm_cached_kernel 的第一遍长得一样，只是多了 (b)(c)。
    #pragma unroll
    for (int it = 0; it < N_ITEMS; ++it){
        const int i = tid + it * BLOCK;
        const float xi = (i < D) ? x[base + i] : 0.f;
        const float ri = (i < D) ? residual[base + i] : 0.f;
        const float hi = xi + ri;
        if(i < D) residual[base + i] = hi;
        sum += hi * hi;
        reg[it] = hi;
    }

    // ---- 块内归约（已给好，和 RMSNorm 那套完全一样）----
    sum = warpReduceSum(sum);
    if (lane == 0) sm[warp] = sum;
    __syncthreads();
    if (warp == 0){
        sum = (lane < BLOCK/32) ? sm[lane] : 0.f;
        sum = warpReduceSum(sum);
        if (lane == 0) sm[0] = sum;
    }
    __syncthreads();
    const float inv = rsqrtf(sm[0] / D + eps);

    // ===== TODO 2: 第二遍（写出 out）=====
    // 唯一要做的决定是"被归一化的那个值从哪儿来"。
    // 这里选错，前面省下的字节会全部还回去（见 README 字节账那一节的第三行）。
    // 另外想一下：gamma[i] 的读为什么不计入字节账？（提示：gamma 只有 D*4 = 16 KB，
    // 所有 block 复用同一份，常驻 L2。）
    #pragma unroll
    for (int it = 0; it < N_ITEMS; ++it){
        const int i = tid + it * BLOCK;
        // if(i < D) out[base + i] = gamma[i] * residual[base + i] * inv;
        if(i < D) out[base + i] = gamma[i] * reg[it] * inv;
    }
}

// ============================== host 侧 ==============================
static void cpu_ref(const vector<float>& x, const vector<float>& r,
                    const vector<float>& gamma, vector<float>& hidden,
                    vector<float>& out, int M, int D, float eps){
    for (int row = 0; row < M; ++row){
        const size_t b = (size_t)row * D;
        double s = 0.0;
        for (int i = 0; i < D; ++i){
            const float h = x[b+i] + r[b+i];
            hidden[b+i] = h;
            s += (double)h * h;
        }
        const double inv = 1.0 / sqrt(s / D + eps);
        for (int i = 0; i < D; ++i)
            out[b+i] = gamma[i] * hidden[b+i] * (float)inv;
    }
}

static void check(const vector<float>& got, const vector<float>& ref, const char* tag){
    double maxErr = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        maxErr = max(maxErr, fabs((double)got[i] - (double)ref[i]));
    printf("  [%-26s] max abs err = %.3g -> %s\n", tag, maxErr,
           maxErr < 1e-4 ? "PASS" : "FAIL");
}

int main(){
    const int M = 32768, D = 4096;
    const float eps = 1e-6f;
    const size_t N = (size_t)M * D;
    const size_t bytes = N * sizeof(float);

    printf("M=%d D=%d  fp32  Block=%d  ITEMS=%d\n", M, D, BLOCK, ITEMS);

    vector<float> h_x(N), h_r(N), h_gamma(D), h_hidden(N), h_out(N);
    unsigned seed = 12345u;
    auto next = [&](){ seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xFFFF) / 65535.0f; };
    for (size_t i = 0; i < N; ++i) h_x[i] = next() * 2.f - 1.f;
    for (size_t i = 0; i < N; ++i) h_r[i] = next() * 2.f - 1.f;
    for (int i = 0; i < D; ++i)   h_gamma[i] = next() * 2.f;

    printf("Computing CPU reference...\n");
    cpu_ref(h_x, h_r, h_gamma, h_hidden, h_out, M, D, eps);

    float *d_x, *d_r, *d_g, *d_out, *d_res;
    CUDA_CHECK(cudaMalloc(&d_x,  bytes));
    CUDA_CHECK(cudaMalloc(&d_r,  bytes));
    CUDA_CHECK(cudaMalloc(&d_res,bytes));
    CUDA_CHECK(cudaMalloc(&d_g,  D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_gamma.data(), D*sizeof(float), cudaMemcpyHostToDevice));

    const int warm = 10, iter = 100;
    vector<float> ts(iter);
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    // ---- A: 未融合（add + rmsnorm 两个 kernel）----
    auto run_unfused = [&](){
        add_kernel<<<M, BLOCK>>>(d_x, d_r, d_out, D);                            // hidden -> d_out
        rmsnorm_cached_kernel<ITEMS><<<M, BLOCK>>>(d_out, d_g, d_res, D, eps);  // out    -> d_res
    };
    for (int i = 0; i < warm; ++i) run_unfused();
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < iter; ++i){
        CUDA_CHECK(cudaEventRecord(e0)); run_unfused(); CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ts[i], e0, e1));
    }
    sort(ts.begin(), ts.end());
    const float ms_unfused = ts[iter/2];
    vector<float> got_a(N);
    CUDA_CHECK(cudaMemcpy(got_a.data(), d_res, bytes, cudaMemcpyDeviceToHost));

    // ---- B: 融合（一个 kernel）。注意 residual 是**就地更新**的，每次要重置 ----
    CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), bytes, cudaMemcpyHostToDevice));
    auto run_fused = [&](){
        fused_add_rmsnorm_kernel<ITEMS><<<M, BLOCK>>>(d_x, d_r, d_g, d_res, D, eps);
    };
    for (int i = 0; i < warm; ++i) run_fused();
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < iter; ++i){
        CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e0)); run_fused(); CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ts[i], e0, e1));
    }
    sort(ts.begin(), ts.end());
    const float ms_fused = ts[iter/2];
    vector<float> got_b(N), got_res(N);
    CUDA_CHECK(cudaMemcpy(got_b.data(),   d_res, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_res.data(), d_r,   bytes, cudaMemcpyDeviceToHost));

    printf("\nCheck:\n");
    check(got_a,   h_out,    "unfused out");
    check(got_b,   h_out,    "fused out");
    check(got_res, h_hidden, "fused residual(in-place)");

    // ===== TODO 3: 字节账 =====
    // 别猜，按"每一趟 kernel 各读了几个数组、写了几个数组"逐项加，一个 fp32 = 4 B。
    //   add_kernel     : 读 x, r            写 out            -> 12 B/元素
    //   rmsnorm_cached : 读 in（gamma 不算） 写 out            ->  8 B/元素
    //   fused          : 读 x, residual     写 residual, out  -> 16 B/元素
    // 把下面两行填成 "N * 你算出来的 B/元素"：
    const double by_unfused = N * (4 + 4 + 4) + N * (4 + 4);
    const double by_fused   = N * (4 + 4 + 4 + 4);
    // const double by_fused   = N * (4 + 4 + 4 + 4 + 4);
    printf("\nunfused(2 kernels) : %8.4f ms   %.1f GB/s  (按 %.0f B/元素算)\n",
           ms_unfused, by_unfused / (ms_unfused * 1e6), by_unfused / (double)N);
    printf("fused  (1 kernel)  : %8.4f ms   %.1f GB/s  (按 %.0f B/元素算)\n",
           ms_fused,   by_fused   / (ms_fused   * 1e6), by_fused   / (double)N);
    printf("speedup = %.3fx   (纯字节比 = %.3fx)\n",
           ms_unfused / ms_fused, by_unfused / by_fused);

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_r)); CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_out));
    return 0;
}

/*
M=32768 D=4096  fp32  Block=256  ITEMS=16
Computing CPU reference...

Check:
  [unfused out               ] max abs err = 4.77e-07 -> PASS
  [fused out                 ] max abs err = 4.77e-07 -> PASS
  [fused residual(in-place)  ] max abs err = 0 -> PASS

unfused(2 kernels) :   3.5064 ms   765.6 GB/s  (按 20 B/元素算)
fused  (1 kernel)  :   2.8905 ms   743.0 GB/s  (按 16 B/元素算)
speedup = 1.213x   (纯字节比 = 1.250x)
*/

/*
第二遍写成 out[base+i] = gamma[i] * (x[base+i] + residual[base+i]) * inv
原因：这次重读是同一个 warp 在几十条指令前刚写过的同一行，命中 L1，根本没走到 DRAM。

M=32768 D=4096  fp32  Block=256  ITEMS=16
Computing CPU reference...

Check:
  [unfused out               ] max abs err = 4.77e-07 -> PASS
  [fused out                 ] max abs err = 4.77e-07 -> PASS
  [fused residual(in-place)  ] max abs err = 0 -> PASS

unfused(2 kernels) :   3.5022 ms   766.5 GB/s  (按 20 B/元素算)
fused  (1 kernel)  :   2.9239 ms   918.1 GB/s  (按 20 B/元素算)
speedup = 1.198x   (纯字节比 = 1.000x)
*/