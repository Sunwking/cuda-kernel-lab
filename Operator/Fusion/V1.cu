/*
Residual + RMSNorm 融合  V1：float4 向量化

V0 的标量版已经拿到 1.213x。但先记住一件事：
**float4 一个 DRAM 字节都不省** —— 还是 16 B/元素。它能省的是另外两样：

  1) global 访问的**指令条数**：每线程 16 次 LDG + 16 次 STG
                              ->  4 次 LDG.128 + 4 次 STG.128（地址计算、循环开销同步下降）
  2) `gamma` 的 **L2 请求数**：gamma 每个元素都要读一次（4 B/元素 = 512 MB 的 L2->SM 流量，
     不落 DRAM），用 float4 读能把 L2 sector 请求数降到 1/4。

所以这一版预期只有几个百分点的收益，甚至可能持平 —— 这正是要量出来的东西。

本文件把 **未融合 / V0（标量） / V1（float4）** 三个版本放在**同一个进程**里计时。
跨 session 的绝对数字不能比，同进程 A/B 才可以。

编译 / 运行：
    nvcc -O3 -arch=native -Xcompiler "/utf-8" V1.cu -o V1.exe
    .\V1.exe
扫 BLOCK（V1 支持改 BLKV）：
    nvcc -O3 -arch=native -Xcompiler "/utf-8" -DBLKV=512 V1.cu -o V1_b512.exe

要填的 TODO 只有 fused_add_rmsnorm_v1_kernel 里的两个循环。
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

using namespace std;

#define CUDA_CHECK(call) do {                                   \
    cudaError_t e = call;                                       \
    if (e != cudaSuccess) {                                     \
        cerr << "CUDA err " << cudaGetErrorString(e) << " @" << __LINE__ << endl; \
        exit(1); } } while (0)

#ifndef BLKV
#define BLKV 256
#endif

constexpr int BLOCK  = BLKV;
constexpr int DVAL   = 4096;                  // 本文件固定 shape，ITEMS/VITEMS 才能是编译期常量
constexpr int ITEMS  = DVAL / BLOCK;          // V0：每线程几个标量
constexpr int VITEMS = (DVAL / 4) / BLOCK;    // V1：每线程几个 float4
static_assert(DVAL % BLOCK == 0,             "DVAL must be a multiple of BLOCK");
static_assert((DVAL / 4) % BLOCK == 0,       "DVAL/4 must be a multiple of BLOCK (float4)");

__device__ __forceinline__ float warpReduceSum(float v){
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// ---------- 块内归约，三个 kernel 公用（已给好）----------
template<int NITEMS>
__device__ __forceinline__ float blockReduceSum(float sum, float* sm,
                                                int lane, int warp){
    sum = warpReduceSum(sum);
    if (lane == 0) sm[warp] = sum;
    __syncthreads();
    if (warp == 0){
        sum = (lane < BLOCK/32) ? sm[lane] : 0.f;
        sum = warpReduceSum(sum);
        if (lane == 0) sm[0] = sum;
    }
    __syncthreads();
    return sm[0];
}

// ===================== 未融合 1/2：hidden = x + residual =====================
__global__ void add_kernel(const float* x, const float* r, float* out, int D){
    const size_t base = (size_t)blockIdx.x * D;
    for (int i = threadIdx.x; i < D; i += BLOCK)
        out[base + i] = x[base + i] + r[base + i];
}

// ============ 未融合 2/2：单遍 RMSNorm（寄存器缓存）============
__global__ void rmsnorm_cached_kernel(const float* in, const float* gamma,
                                      float* out, int D, float eps){
    __shared__ float sm[BLOCK / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t base = (size_t)blockIdx.x * D;

    float reg[ITEMS];
    float sum = 0.f;
    #pragma unroll
    for (int it = 0; it < ITEMS; ++it){
        const int i = tid + it * BLOCK;
        const float v = (i < D) ? in[base + i] : 0.f;
        reg[it] = v;
        sum += v * v;
    }
    const float inv = rsqrtf(blockReduceSum<ITEMS>(sum, sm, lane, warp) / D + eps);
    #pragma unroll
    for (int it = 0; it < ITEMS; ++it){
        const int i = tid + it * BLOCK;
        if (i < D) out[base + i] = gamma[i] * reg[it] * inv;
    }
}

// ===== V0：标量版融合（已给好，作为同进程的 A/B 基线，别改）=====
__global__ void fused_add_rmsnorm_kernel(const float* x, float* residual,
                                         const float* gamma, float* out,
                                         int D, float eps){
    __shared__ float sm[BLOCK / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t base = (size_t)blockIdx.x * D;

    float reg[ITEMS];
    float sum = 0.f;
    #pragma unroll
    for (int it = 0; it < ITEMS; ++it){
        const int i = tid + it * BLOCK;
        const float xi = (i < D) ? x[base + i] : 0.f;
        const float ri = (i < D) ? residual[base + i] : 0.f;
        const float hi = xi + ri;
        if (i < D) residual[base + i] = hi;
        reg[it] = hi;
        sum += hi * hi;
    }
    const float inv = rsqrtf(blockReduceSum<ITEMS>(sum, sm, lane, warp) / D + eps);
    #pragma unroll
    for (int it = 0; it < ITEMS; ++it){
        const int i = tid + it * BLOCK;
        if (i < D) out[base + i] = gamma[i] * reg[it] * inv;
    }
}

// ============================================================
// V1：float4 版融合
//   每线程负责 VITEMS 个 float4，行内 float4 下标 iv = tid + it*BLOCK。
//   warp 内 32 个 lane 拿到 iv..iv+31  ->  32 x 16 B = 512 B 连续，合并访存。
//
//   **前提**：x / residual / out / gamma 的首地址要 16 B 对齐（cudaMalloc 给的是 256 B 对齐，
//   够用），并且 D 能被 4*BLOCK 整除 —— 文件顶上已经 static_assert 了，
//   所以这一版**不需要任何 i < D 判界**（判界本身也是指令）。
// ============================================================
__global__ void fused_add_rmsnorm_v1_kernel(const float* __restrict__ x,
                                            float* __restrict__ residual,
                                            const float* __restrict__ gamma,
                                            float* __restrict__ out,
                                            int D, float eps){
    __shared__ float sm[BLOCK / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t base4 = (size_t)blockIdx.x * (D / 4);   // 本行起点，单位 = float4

    // 四个 float4 视角的指针（已给好）
    const float4* x4 = reinterpret_cast<const float4*>(x)      + base4;
    float4*       r4 = reinterpret_cast<float4*>(residual)     + base4;
    float4*       o4 = reinterpret_cast<float4*>(out)          + base4;
    const float4* g4 = reinterpret_cast<const float4*>(gamma);  // gamma 是按列共享的，没有 base4

    float4 reg[VITEMS];
    float sum = 0.f;

    // ===== TODO 1: 第一遍（float4）=====
    // 在 `const int iv = tid + it * BLOCK;` 之后，用**一条** float4 读 x、**一条** float4 读 residual，
    // 相加得到一个 float4 的 hidden，然后：
    //   (a) 把它就地写回 r4[iv]
    //   (b) 把它的四个分量的平方和累加进 sum
    //   (c) 整个 float4 存进 reg[it] 留给第二遍
    // 最容易错的一步：`iv` 是 **float4 的下标**，不是元素下标（元素下标 = 4*iv，本文件里用不到它）。
    #pragma unroll
    for (int it = 0; it < VITEMS; ++it){
        const int iv = tid + it * BLOCK;
        float4 xi = x4[iv];
        float4 ri = r4[iv];
        float4 hi = make_float4(xi.x + ri.x, xi.y + ri.y, xi.z + ri.z, xi.w + ri.w);
        reg[it] = hi;
        r4[iv] = hi;
        sum += hi.x * hi.x + hi.y * hi.y + hi.z * hi.z + hi.w * hi.w;
    }

    const float inv = rsqrtf(blockReduceSum<VITEMS>(sum, sm, lane, warp) / D + eps);

    // ===== TODO 2: 第二遍（float4）=====
    // 用一条 float4 读 gamma（注意是 g4[iv]，**同一份 gamma 被所有 block 共用**），
    // 和 reg[it] 的四个分量分别相乘，再一条 float4 写回 o4[iv]。
    // 写 gamma 的读时想一下：为什么必须也是 float4，只读 gamma.x 那种做法在这里行不通？
    #pragma unroll
    for (int it = 0; it < VITEMS; ++it){
        const int iv = tid + it * BLOCK;
        float4 gi = g4[iv];
        float4 hi = reg[it];
        o4[iv] = make_float4(gi.x * hi.x * inv, gi.y * hi.y * inv, gi.z * hi.z * inv, gi.w * hi.w * inv);
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
    printf("  [%-28s] max abs err = %.3g -> %s\n", tag, maxErr,
           maxErr < 1e-4 ? "PASS" : "FAIL");
}

int main(){
    const int M = 32768, D = 4096;
    const float eps = 1e-6f;
    const size_t N = (size_t)M * D;
    const size_t bytes = N * sizeof(float);

    printf("M=%d D=%d fp32 BLOCK=%d ITEMS=%d VITEMS=%d V0v1=%d\n",
           M, D, BLOCK, ITEMS, VITEMS, (int)(DVAL/4));

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

    // reset 在**计时区之外**：fused 是就地更新的，每次要重置输入；但重置本身不算时间。
    auto measure = [&](const function<void()>& reset, const function<void()>& launch){
        for (int i = 0; i < warm; ++i){ reset(); launch(); }
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < iter; ++i){
            reset();
            CUDA_CHECK(cudaEventRecord(e0)); launch(); CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ts[i], e0, e1));
        }
        sort(ts.begin(), ts.end());
        return ts[iter/2];
    };
    const function<void()> noop  = [](){};
    const function<void()> resetR = [&](){ CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), bytes, cudaMemcpyHostToDevice)); };

    vector<float> got_u(N), got_v0(N), got_v0r(N), got_v1(N), got_v1r(N);

    // ---- A: 未融合（两个 kernel），20 B/元素 ----
    const float ms_unfused = measure(noop, [&](){
        add_kernel<<<M, BLOCK>>>(d_x, d_r, d_out, D);
        rmsnorm_cached_kernel<<<M, BLOCK>>>(d_out, d_g, d_res, D, eps);
    });
    CUDA_CHECK(cudaMemcpy(got_u.data(), d_res, bytes, cudaMemcpyDeviceToHost));

    // ---- B: V0 标量融合，16 B/元素 ----
    const float ms_v0 = measure(resetR, [&](){
        fused_add_rmsnorm_kernel<<<M, BLOCK>>>(d_x, d_r, d_g, d_res, D, eps);
    });
    CUDA_CHECK(cudaMemcpy(got_v0.data(),  d_res, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_v0r.data(), d_r,   bytes, cudaMemcpyDeviceToHost));

    // ---- C: V1 float4 融合，16 B/元素 ----
    const float ms_v1 = measure(resetR, [&](){
        fused_add_rmsnorm_v1_kernel<<<M, BLOCK>>>(d_x, d_r, d_g, d_res, D, eps);
    });
    CUDA_CHECK(cudaMemcpy(got_v1.data(),  d_res, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_v1r.data(), d_r,   bytes, cudaMemcpyDeviceToHost));

    printf("\nCheck:\n");
    check(got_u,   h_out,    "unfused out");
    check(got_v0,  h_out,    "V0 out");
    check(got_v0r, h_hidden, "V0 residual(in-place)");
    check(got_v1,  h_out,    "V1 out");
    check(got_v1r, h_hidden, "V1 residual(in-place)");

    const double N_BYTES_UNFUSED = N * (4 + 4 + 4) + N * (4 + 4);   // 12 + 8
    const double N_BYTES_FUSED   = N * (4 + 4 + 4 + 4);             // 16
    printf("\n%-26s %9s %10s   %s\n", "impl", "ms", "GB/s", "bytes/elem");
    printf("%-26s %9.4f %10.1f   %d B/元素\n", "unfused(add+rmsnorm)",
           ms_unfused, N_BYTES_UNFUSED/(ms_unfused*1e6), 20);
    printf("%-26s %9.4f %10.1f   %d B/元素\n", "fused V0 (scalar)",
           ms_v0, N_BYTES_FUSED/(ms_v0*1e6), 16);
    printf("%-26s %9.4f %10.1f   %d B/元素\n", "fused V1 (float4)",
           ms_v1, N_BYTES_FUSED/(ms_v1*1e6), 16);
    printf("\nV1 vs V0      = %.3fx\n", ms_v0 / ms_v1);
    printf("V1 vs unfused = %.3fx\n", ms_unfused / ms_v1);

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_r)); CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_out));
    return 0;
}

/*
M=32768 D=4096 fp32 BLOCK=256 ITEMS=16 VITEMS=4 V0v1=1024
Computing CPU reference...

Check:
  [unfused out                 ] max abs err = 4.77e-07 -> PASS
  [V0 out                      ] max abs err = 4.77e-07 -> PASS
  [V0 residual(in-place)       ] max abs err = 0 -> PASS
  [V1 out                      ] max abs err = 4.77e-07 -> PASS
  [V1 residual(in-place)       ] max abs err = 0 -> PASS

impl                              ms       GB/s   bytes/elem
unfused(add+rmsnorm)          3.4656      774.6   20 B/元素
fused V0 (scalar)             2.9136      737.1   16 B/元素
fused V1 (float4)             2.8641      749.8   16 B/元素

V1 vs V0      = 1.017x
V1 vs unfused = 1.210x
*/
