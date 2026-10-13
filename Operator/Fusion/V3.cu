/*
SwiGLU 融合  V3：out = silu(gate) * up

前两版的中间张量都是"自己刚写出去、下一趟又要完整读回来"的量（hidden / y），
所以融合上限 = 省掉的那份字节 / 新的总字节：

    V0  Residual + RMSNorm : 20 -> 16 B/元素   上限 1.25x
    V2  RMSNorm + RoPE     : 16 ->  8 B/元素   上限 2.00x
    V3  SwiGLU             : 20 -> 12 B/元素   上限 1.67x

**为什么 V3 又变低了？** 因为 gate / up 都是"本来就存在、必须从 DRAM 读进来"的输入，
融合只能省掉"自己写出去又自己读回来"的那部分（silu 的中间结果 t，8 B/元素），
省不掉 up 的 4 B。于是有融合的通用公式：

    收益上限 = 1 + (被消掉的中间量字节) / (融合后的总字节)

    V0: 1 + 4/16 = 1.25x      V2: 1 + 8/8 = 2.00x      V3: 1 + 8/12 = 1.67x

--------------------------------------------------------------------------
V3 的新东西：**第一次把计算塞进一个本该是纯搬运的 kernel。**

silu 需要一次 exp 加一次除法（在 SM 上是 MUFU.EX2 + MUFU.RCP，两条 SFU 指令）。
问题是：这会不会把这版从 memory-bound 顶成 compute-bound？

本机的算力预算（背下来，这是判断"能不能白塞计算"的尺子）：

    FP32 峰值 ≈ 44.75 TFLOPS        可达带宽 ≈ 763.5 GB/s
    -> 每搬 1 字节，可以配 ~58 条 FLOP
    silu 每个元素大约 8~10 FLOP + 2 SFU，而每个元素要搬 12 字节 -> 远在预算内

所以预期是"几乎免费"。但要量出来才算数（文件里给了对照组）。

--------------------------------------------------------------------------
文件里有 4 个 kernel：

    1) silu_kernel         未融合 1/2：读 gate 写 t        8 B/元素    已给好
    2) mul_kernel          未融合 2/2：读 t、up 写 out    12 B/元素    已给好
    3) copy2_kernel        对照组：读两个数组直接相加写回  12 B/元素    已给好
    4) fused_swiglu_kernel 融合版                         12 B/元素    TODO 1

    对照组 copy2 和融合版**字节数完全一样**（都是读 2 写 1），区别只有"有没有算激活"。
    两者时间之差，就是 silu 的价格。

TODO 1：fused_swiglu_kernel 的循环体
TODO 2：act() —— silu 的实现（提示： silu(x) = x * sigmoid(x) = x / (1 + exp(-x))）

编译 / 运行：
    nvcc -O3 -arch=native -Xcompiler "/utf-8" V3.cu -o V3.exe
    .\V3.exe

做完之后把 act() 里的 __expf 换成 expf 再编一次、再跑一次 —— 那是第 3 题要量的东西。
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

using namespace std;

#define CUDA_CHECK(call) do {                                   \
    cudaError_t e = call;                                       \
    if (e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); \
        exit(1); } } while (0)

constexpr int BLOCKV = 256;
constexpr int DVALV  = 4096;                        // 本文件固定 shape
constexpr int VITEMS = (DVALV / 4) / BLOCKV;        // 每线程 4 个 float4
static_assert((DVALV / 4) % BLOCKV == 0, "DVALV/4 must be a multiple of BLOCKV");

// ==================== TODO 2: silu ====================
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// 先用 __expf（SFU 版，快但精度低），跑通之后再换 expf 量一次。
__device__ __forceinline__ float act(float x){
    return x / (1 + expf(-x));
}

// ============ 未融合 1/2：t = silu(gate)    读 gate 写 t = 8 B/元素 ============
__global__ void silu_kernel(const float4* __restrict__ gate, float4* __restrict__ t, int D4){
    const size_t base = (size_t)blockIdx.x * D4;
    #pragma unroll
    for (int it = 0; it < VITEMS; ++it){
        const int iv = threadIdx.x + it * BLOCKV;
        const float4 g = gate[base + iv];
        t[base + iv] = make_float4(act(g.x), act(g.y), act(g.z), act(g.w));
    }
}

// ============ 未融合 2/2：out = t * up    读 2 写 1 = 12 B/元素 ============
__global__ void mul_kernel(const float4* __restrict__ t, const float4* __restrict__ up,
                           float4* __restrict__ out, int D4){
    const size_t base = (size_t)blockIdx.x * D4;
    #pragma unroll
    for (int it = 0; it < VITEMS; ++it){
        const int iv = threadIdx.x + it * BLOCKV;
        const float4 a = t[base + iv], b = up[base + iv];
        out[base + iv] = make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
    }
}

// ============ 对照组：读两个数组、直接相加写回，一点激活都不算（12 B/元素）============
// 它和融合版字节数一模一样，时间差就是"算 silu 的价格"。
__global__ void copy2_kernel(const float4* __restrict__ a, const float4* __restrict__ b,
                             float4* __restrict__ out, int D4){
    const size_t base = (size_t)blockIdx.x * D4;
    #pragma unroll
    for (int it = 0; it < VITEMS; ++it){
        const int iv = threadIdx.x + it * BLOCKV;
        const float4 x = a[base + iv], y = b[base + iv];
        out[base + iv] = make_float4(x.x+y.x, x.y+y.y, x.z+y.z, x.w+y.w);
    }
}

// ==================== 融合版：out = silu(gate) * up（12 B/元素）====================
__global__ void fused_swiglu_kernel(const float4* __restrict__ gate,
                                    const float4* __restrict__ up,
                                    float4* __restrict__ out, int D4){
    const size_t base = (size_t)blockIdx.x * D4;
    // ===== TODO 1 =====
    // 循环结构和上面三个 kernel 完全一样（`const int iv = threadIdx.x + it * BLOCKV;`
    // 之外的下标都是 `base + iv`）。要做的只有：
    //   读一个 gate 的 float4、读一个 up 的 float4
    //   对 gate 的 4 个分量各调用一次 act()
    //   逐分量乘起来，一条 float4 写回 out
    // 这里**不需要**任何中间数组、也不需要 __syncthreads —— 每个元素自己独立，
    // "融合"在这里的体现就是：中间结果 t 从头到尾没有落过地。
    #pragma unroll
    for (int it = 0; it < VITEMS; ++it){
        const int iv = threadIdx.x + it * BLOCKV;
        float4 a = gate[base + iv], b = up[base + iv];
        float4 x = make_float4(act(a.x), act(a.y), act(a.z), act(a.w));
        out[base + iv] = make_float4(x.x * b.x, x.y * b.y, x.z * b.z, x.w * b.w); 
    }
}

// ============================== host 侧 ==============================
static void cpu_ref(const vector<float>& gate, const vector<float>& up,
                    vector<float>& out, size_t N){
    for (size_t i = 0; i < N; ++i){
        const double g = gate[i];
        out[i] = (float)((g / (1.0 + exp(-g))) * (double)up[i]);
    }
}

static void check(const vector<float>& got, const vector<float>& ref, const char* tag){
    double maxErr = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        maxErr = max(maxErr, fabs((double)got[i] - (double)ref[i]));
    printf("  [%-24s] max abs err = %.3g -> %s\n", tag, maxErr,
           maxErr < 1e-4 ? "PASS" : "FAIL");
}

int main(){
    const int    M     = 32768;
    const int    D     = DVALV;
    const size_t N     = (size_t)M * D;
    const size_t bytes = N * sizeof(float);
    const int    D4    = D / 4;

    printf("M=%d D=%d fp32 BLOCK=%d VITEMS=%d\n", M, D, BLOCKV, VITEMS);

    vector<float> h_gate(N), h_up(N), h_out(N);
    unsigned seed = 12345u;
    auto next = [&](){ seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xFFFF) / 65535.0f; };
    for (size_t i = 0; i < N; ++i) h_gate[i] = next() * 6.f - 3.f;   // silu 要有点负区间才看得出
    for (size_t i = 0; i < N; ++i) h_up[i]   = next() * 2.f - 1.f;

    printf("Computing CPU reference...\n");
    cpu_ref(h_gate, h_up, h_out, N);

    float *d_gate, *d_up, *d_t, *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_gate, bytes));
    CUDA_CHECK(cudaMalloc(&d_up,   bytes));
    CUDA_CHECK(cudaMalloc(&d_t,    bytes));
    CUDA_CHECK(cudaMalloc(&d_a,    bytes));
    CUDA_CHECK(cudaMalloc(&d_b,    bytes));
    CUDA_CHECK(cudaMalloc(&d_out,  bytes));
    CUDA_CHECK(cudaMemcpy(d_gate, h_gate.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up,   h_up.data(),   bytes, cudaMemcpyHostToDevice));

    const int warm = 10, iter = 100;
    vector<float> ts(iter);
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    auto measure = [&](const function<void()>& launch){
        for (int i = 0; i < warm; ++i) launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < iter; ++i){
            CUDA_CHECK(cudaEventRecord(e0)); launch(); CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ts[i], e0, e1));
        }
        sort(ts.begin(), ts.end());
        return ts[iter/2];
    };

    vector<float> got_u(N), got_f(N);
    const float4* G4 = reinterpret_cast<const float4*>(d_gate);
    const float4* U4 = reinterpret_cast<const float4*>(d_up);
    float4* T4 = reinterpret_cast<float4*>(d_t);
    float4* A4 = reinterpret_cast<float4*>(d_a);
    float4* B4 = reinterpret_cast<float4*>(d_b);
    float4* O4 = reinterpret_cast<float4*>(d_out);

    const float ms_unfused = measure([&](){
        silu_kernel<<<M, BLOCKV>>>(G4, T4, D4);
        mul_kernel<<<M, BLOCKV>>>(T4, U4, O4, D4);
    });
    CUDA_CHECK(cudaMemcpy(got_u.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    const float ms_copy2 = measure([&](){
        copy2_kernel<<<M, BLOCKV>>>(G4, U4, A4, D4);
    });
    (void)B4;

    const float ms_fused = measure([&](){
        fused_swiglu_kernel<<<M, BLOCKV>>>(G4, U4, O4, D4);
    });
    CUDA_CHECK(cudaMemcpy(got_f.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    printf("\nCheck:\n");
    check(got_u, h_out, "unfused out");
    check(got_f, h_out, "fused out");

    const double BY_UNFUSED = (double)N * (4 + 4) + (double)N * (4 + 4 + 4);  // 8 + 12
    const double BY_FUSED   = (double)N * (4 + 4 + 4);                        // 12
    printf("\n%-28s %9s %10s   %s\n", "impl", "ms", "GB/s", "bytes/elem");
    printf("%-28s %9.4f %10.1f   %d B/elem\n", "unfused(silu+mul)",
           ms_unfused, BY_UNFUSED / (ms_unfused * 1e6), 20);
    printf("%-28s %9.4f %10.1f   %d B/elem\n", "copy2 (no activation)",
           ms_copy2, BY_FUSED / (ms_copy2 * 1e6), 12);
    printf("%-28s %9.4f %10.1f   %d B/elem\n", "fused V3 (silu*up)",
           ms_fused, BY_FUSED / (ms_fused * 1e6), 12);
    printf("\nspeedup (unfused/fused) = %.3fx   (upper bound %.2fx)\n",
           ms_unfused / ms_fused, BY_UNFUSED / BY_FUSED);
    printf("activation price (fused/copy2) = %.3fx\n", ms_fused / ms_copy2);

    CUDA_CHECK(cudaFree(d_gate)); CUDA_CHECK(cudaFree(d_up)); CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
    return 0;
}

/*
__expf

M=32768 D=4096 fp32 BLOCK=256 VITEMS=4
Computing CPU reference...

Check:
  [unfused out             ] max abs err = 4.77e-07 -> PASS
  [fused out               ] max abs err = 4.77e-07 -> PASS

impl                                ms       GB/s   bytes/elem
unfused(silu+mul)               3.4957      767.9   20 B/elem
copy2 (no activation)           2.0712      777.6   12 B/elem
fused V3 (silu*up)              2.0763      775.7   12 B/elem

speedup (unfused/fused) = 1.684x   (upper bound 1.67x)
activation price (fused/copy2) = 1.002x

expf
M=32768 D=4096 fp32 BLOCK=256 VITEMS=4
Computing CPU reference...

Check:
  [unfused out             ] max abs err = 2.38e-07 -> PASS
  [fused out               ] max abs err = 2.38e-07 -> PASS

impl                                ms       GB/s   bytes/elem
unfused(silu+mul)               3.4717      773.2   20 B/elem
copy2 (no activation)           2.0483      786.3   12 B/elem
fused V3 (silu*up)              2.0585      782.4   12 B/elem

speedup (unfused/fused) = 1.687x   (upper bound 1.67x)
activation price (fused/copy2) = 1.005x

*/