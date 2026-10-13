/*
RMSNorm + RoPE 融合  V2：真实 LLaMA 结构

V0/V1 的收益上限只有 1.25x，因为融合只省掉了 4 B/元素的中间张量。
这一版换一条链：

    y   = rmsnorm_per_head(x) * gamma_q      // 每个 head 单独归一化（head_dim = 128）
    out = rope(y, cos, sin)                  // 旋转位置编码：把每一对元素混起来

未融合本来就是**两个 kernel**：rmsnorm_head(读 x 写 y) + rope(读 y 写 out)。
融合之后只剩 读 x + 写 out。

字节账（一个 fp32 元素 = 4 B）：

    未融合 rmsnorm_head : 读 x   4 B    写 y   4 B   ->  8 B
    未融合 rope         : 读 y   4 B    写 out 4 B   ->  8 B
    未融合合计                                            16 B
    融合                : 读 x   4 B    写 out 4 B   ->  8 B

    **理论上限 = 16 / 8 = 2.00x**   <- 比 V0 的 1.25x 高一档
    （cos/sin 表每行 2*64*4 = 512 B，被同一行的 4 个 block 共用、常驻 L2，
      折算约 0.1 B/元素，和 gamma 一样不落 DRAM，先忽略。）

为什么上限变高了：融合省的字节数 = 2 x 中间张量的字节数，
中间张量在所有访问里占的比例越大，上限越高。V0 的中间张量只占 4/20，这里占 8/16。

--------------------------------------------------------------------------
比 V0/V1 多出来的两个新东西（这就是这一版要学的东西）：

1) **归一化的维度变了。** 不再是整行 4096，而是每个 head 的 128 个元素。
   于是"一个 block 归约一整行"变成"**一个 warp 归约一个 head**"：
   __syncthreads 和 shared memory **全都不需要了**，纯 warp shuffle 就够
   （归约只有 5 步，比 block 归约那套两级 shuffle+smem 简单得多）。

2) **RoPE 是"混对"运算**：把一对 (a, b) 变成 (a*c - b*s, a*s + b*c)。
   融合之后 y 还躺在寄存器里，所以这一步**不需要任何跨线程交换**——
   下面 TODO 3 会让你想清楚为什么，以及换成 HF/NeoX 的 rotate_half 之后为什么就变了。

--------------------------------------------------------------------------
线程映射（已给好，先读懂它再动手）：

    一个 block = NWARP = 8 个 warp = 8 个 head = 8 * 128 = 1024 个元素
    一个 warp  = 一个 head = 128 个元素 = 32 个 float4
    一个 lane  = 一个 float4 = 4 个元素 = RoPE 的两对
    grid.x     = M * NHEAD / NWARP = 131072

    lane l 持有元素 4l .. 4l+3  ->  RoPE 的两对是 (4l,4l+1) 和 (4l+2,4l+3)
    它们对应的角度下标是 d = 2l 和 d = 2l+1（cos/sin 表每行只有 HDIM/2 = 64 个角度）
    所以本 lane 要的 cos/sin 正好是表里第 l 个 float2 -> c2[lane] / s2[lane]（已给好）

编译 / 运行：
    nvcc -O3 -arch=native -Xcompiler "/utf-8" V2.cu -o V2.exe
    .\V2.exe

要填的 TODO 只有 3 个，都在 fused_rmsnorm_rope_kernel 里。
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

constexpr int   NHEAD  = 32;                 // head 数
constexpr int   HDIM   = 128;                // head_dim
constexpr int   DIM    = NHEAD * HDIM;       // 4096，一整行
constexpr int   NWARP  = 8;                   // 每 block 8 个 warp = 8 个 head
constexpr int   BLOCKN = NWARP * 32;         // 256 线程
constexpr int   ND     = HDIM / 2;           // cos/sin 表每行 64 个角度
constexpr float ROPE_BASE = 10000.0f;

__device__ __forceinline__ float warpReduceSum(float v){
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// ============ 未融合 1/2：per-head RMSNorm（读 x 写 y）============
__global__ void rmsnorm_head_kernel(const float* __restrict__ x,
                                    const float* __restrict__ gamma,
                                    float* __restrict__ y, float eps){
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gh   = blockIdx.x * NWARP + warp;
    const int row  = gh / NHEAD, head = gh % NHEAD;
    const size_t off = (size_t)row * (DIM / 4) + (size_t)head * (HDIM / 4);

    const float4* x4 = reinterpret_cast<const float4*>(x) + off;
    float4*       y4 = reinterpret_cast<float4*>(y)       + off;
    const float4* g4 = reinterpret_cast<const float4*>(gamma);

    const float4 v = x4[lane];
    float s = v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    s = warpReduceSum(s);
    // 广播回 32 个 lane，让每个 lane 都能用
    s = __shfl_sync(0xffffffffu, s, 0);
    const float inv = rsqrtf(s / (float)HDIM + eps);
    y4[lane] = make_float4(v.x*inv*g4[lane].x, v.y*inv*g4[lane].y,
                           v.z*inv*g4[lane].z, v.w*inv*g4[lane].w);
}

// ============ 未融合 2/2：RoPE（读 y, cos/sin，写 out）============
__global__ void rope_kernel(const float* __restrict__ y,
                            const float* __restrict__ cos_tab,
                            const float* __restrict__ sin_tab,
                            float* __restrict__ out){
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gh   = blockIdx.x * NWARP + warp;
    const int row  = gh / NHEAD, head = gh % NHEAD;
    const size_t off = (size_t)row * (DIM / 4) + (size_t)head * (HDIM / 4);

    const float4* y4 = reinterpret_cast<const float4*>(y)   + off;
    float4*       o4 = reinterpret_cast<float4*>(out)       + off;
    const float2* c2 = reinterpret_cast<const float2*>(cos_tab + (size_t)row * ND);
    const float2* s2 = reinterpret_cast<const float2*>(sin_tab + (size_t)row * ND);

    const float4 v = y4[lane];
    const float2 cc = c2[lane], ss = s2[lane];
    o4[lane] = make_float4(v.x*cc.x - v.y*ss.x, v.x*ss.x + v.y*cc.x,
                           v.z*cc.y - v.w*ss.y, v.z*ss.y + v.w*cc.y);
}

// ============================================================
// 融合版：归一化 + gamma + RoPE，一趟走完
// ============================================================
__global__ void fused_rmsnorm_rope_kernel(const float* __restrict__ x,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ cos_tab,
                                          const float* __restrict__ sin_tab,
                                          float* __restrict__ out,
                                          float eps){
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    // ===== TODO 1: 把 (blockIdx.x, warp) 拆成 (row, head) =====
    // gh 是"全局第几个 head"，head 是**行内循环**的，一个 token 有 NHEAD 个 head。
    // 提示：一整行是 DIM 个元素，head 的 stride 就是 HDIM。
    const int gh = blockIdx.x * NWARP + warp;
    const int row = gh / NHEAD, head = gh % NHEAD;

    // ---- 下面这些指针已经给好：本 head 的 128 个元素 = 32 个 float4，lane l 拿第 l 个 ----
    const size_t off = (size_t)row * (DIM / 4) + (size_t)head * (HDIM / 4);
    const float4* x4 = reinterpret_cast<const float4*>(x)   + off;
    float4*       o4 = reinterpret_cast<float4*>(out)       + off;
    const float4* g4 = reinterpret_cast<const float4*>(gamma);                 // 所有 head 共用
    const float2* c2 = reinterpret_cast<const float2*>(cos_tab + (size_t)row * ND);
    const float2* s2 = reinterpret_cast<const float2*>(sin_tab + (size_t)row * ND);
    const int iv = lane;                                                       // 本 lane 的 float4 下标

    const float4 v = x4[iv];                                                   // 本 lane 的 4 个元素

    // ===== TODO 2: warp 内归约，算出本 head 的 inv =====
    // 三步：
    //   (1) 先把本 lane 这 4 个元素的平方和算成 sum
    //   (2) 用 warpReduceSum 把它归约成整颗 head(128 个元素)的平方和
    //   (3) 得到 inv = rsqrtf(sum / HDIM + eps)，让**所有 lane** 都能用
    // 坑：文件里的 warpReduceSum 是 __shfl_down 版 —— 归约完**只有 lane 0 拿到总和**，
    //     别的 lane 手里是残值。所以第 (3) 步之前必须把结果广播回 32 个 lane。
    //     （广播用 __shfl_sync(0xffffffffu, s, 0)。想一下：不广播会错成什么样？）
    float sum = 0.f;
    sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    sum = warpReduceSum(sum);
    sum = __shfl_sync(0xffffffffu, sum, 0);
    const float inv = rsqrtf(sum / (float)HDIM + eps);

    // ===== TODO 3: 归一化 -> gamma -> RoPE -> 写回 =====
    // 顺序不能颠倒：先归一化、再乘 gamma、**最后**才做 RoPE。
    // （旋转会把同一个 head 里的两个元素混起来，混完再乘 gamma 就错了。）
    //   1) h = v * inv * g4[iv]                // 逐分量，三个 float4 相乘，得到本 lane 的 4 个归一化值
    //   2) RoPE 的配对关系：本 lane 的 (h.x, h.y) 用 (c2[iv].x, s2[iv].x)，
    //                       (h.z, h.w) 用 (c2[iv].y, s2[iv].y)
    //      旋转公式：一对 (a, b) -> (a*c - b*s, a*s + b*c)
    //   3) o4[iv] = 旋转后的 float4
    // 想一下：为什么这里一次 __shfl / 一次 shared 交换都不需要？
    //         （提示：RoPE 要配对的 (a,b) 是不是本来就在同一个 lane 手里？）
    float4 h = make_float4(v.x * inv * g4[iv].x, v.y * inv * g4[iv].y,
                    v.z * inv * g4[iv].z, v.w * inv * g4[iv].w);
    o4[iv] = make_float4(h.x * c2[iv].x - h.y * s2[iv].x, h.x * s2[iv].x + h.y * c2[iv].x,
                         h.z * c2[iv].y - h.w * s2[iv].y, h.z * s2[iv].y + h.w * c2[iv].y);
}

// ============================== host 侧 ==============================
static void cpu_ref(const vector<float>& x, const vector<float>& g,
                    const vector<float>& cs, const vector<float>& sn,
                    vector<float>& out, int M, float eps){
    for (int row = 0; row < M; ++row){
        const size_t b  = (size_t)row * DIM;
        const size_t tb = (size_t)row * ND;
        for (int h = 0; h < NHEAD; ++h){
            const size_t hb = b + (size_t)h * HDIM;
            double s = 0.0;
            for (int i = 0; i < HDIM; ++i){ const double v = x[hb + i]; s += v * v; }
            const double inv = 1.0 / sqrt(s / HDIM + eps);
            for (int d = 0; d < ND; ++d){
                const double a  = (double)x[hb + 2*d]     * inv * g[2*d];
                const double c  = (double)x[hb + 2*d + 1] * inv * g[2*d + 1];
                const double co = cs[tb + d], si = sn[tb + d];
                out[hb + 2*d]     = (float)(a * co - c * si);
                out[hb + 2*d + 1] = (float)(a * si + c * co);
            }
        }
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
    const int    M       = 32768;
    const float  eps     = 1e-6f;
    const size_t N       = (size_t)M * DIM;
    const size_t bytes   = N * sizeof(float);
    const int    gridx   = M * NHEAD / NWARP;

    printf("M=%d NHEAD=%d HDIM=%d DIM=%d NWARP=%d BLOCK=%d grid.x=%d\n",
           M, NHEAD, HDIM, DIM, NWARP, BLOCKN, gridx);

    vector<float> h_x(N), h_g(HDIM), h_cos((size_t)M * ND), h_sin((size_t)M * ND), h_out(N);
    unsigned seed = 12345u;
    auto next = [&](){ seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xFFFF) / 65535.0f; };
    for (size_t i = 0; i < N; ++i) h_x[i] = next() * 2.f - 1.f;
    for (int i = 0; i < HDIM; ++i) h_g[i] = next() * 2.f;

    for (int p = 0; p < M; ++p)
        for (int d = 0; d < ND; ++d){
            const float th = (float)p * powf(ROPE_BASE, -2.0f * d / (float)HDIM);
            h_cos[(size_t)p * ND + d] = cosf(th);
            h_sin[(size_t)p * ND + d] = sinf(th);
        }

    printf("Computing CPU reference...\n");
    cpu_ref(h_x, h_g, h_cos, h_sin, h_out, M, eps);

    float *d_x, *d_y, *d_g, *d_cos, *d_sin, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   bytes));
    CUDA_CHECK(cudaMalloc(&d_y,   bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_g,   HDIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, (size_t)M * ND * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, (size_t)M * ND * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x,   h_x.data(),   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g,   h_g.data(),   HDIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), (size_t)M * ND * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), (size_t)M * ND * sizeof(float), cudaMemcpyHostToDevice));

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

    const float ms_unfused = measure([&](){
        rmsnorm_head_kernel<<<gridx, BLOCKN>>>(d_x, d_g, d_y, eps);
        rope_kernel<<<gridx, BLOCKN>>>(d_y, d_cos, d_sin, d_out);
    });
    CUDA_CHECK(cudaMemcpy(got_u.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    const float ms_fused = measure([&](){
        fused_rmsnorm_rope_kernel<<<gridx, BLOCKN>>>(d_x, d_g, d_cos, d_sin, d_out, eps);
    });
    CUDA_CHECK(cudaMemcpy(got_f.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    printf("\nCheck:\n");
    check(got_u, h_out, "unfused out");
    check(got_f, h_out, "fused out");

    const double BY_UNFUSED = (double)N * (4 + 4 + 4 + 4);
    const double BY_FUSED   = (double)N * (4 + 4);
    printf("\n%-26s %9s %10s   %s\n", "impl", "ms", "GB/s", "bytes/elem");
    printf("%-26s %9.4f %10.1f   %d B/elem\n", "unfused(rmsnorm+rope)",
           ms_unfused, BY_UNFUSED / (ms_unfused * 1e6), 16);
    printf("%-26s %9.4f %10.1f   %d B/elem\n", "fused V2",
           ms_fused, BY_FUSED / (ms_fused * 1e6), 8);
    printf("\nspeedup (unfused/fused) = %.3fx   (upper bound %.2fx)\n",
           ms_unfused / ms_fused, BY_UNFUSED / BY_FUSED);

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_cos)); CUDA_CHECK(cudaFree(d_sin));
    return 0;
}

/*
M=32768 NHEAD=32 HDIM=128 DIM=4096 NWARP=8 BLOCK=256 grid.x=131072
Computing CPU reference...

Check:
  [unfused out             ] max abs err = 9.54e-07 -> PASS
  [fused out               ] max abs err = 9.54e-07 -> PASS

impl                              ms       GB/s   bytes/elem
unfused(rmsnorm+rope)         2.8601      750.8   16 B/elem
fused V2                      1.4408      745.3   8 B/elem

speedup (unfused/fused) = 1.985x   (upper bound 2.00x)
*/