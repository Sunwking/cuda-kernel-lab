/*
-arch=sm_120
M = 2048
N = 2048
K = 2048

Computing CPU reference...
CPU reference finished.
block = 256
grid = (16, 16)
STAGES = 4

============================================
                Result
============================================
Average Kernel Time = 0.753281 ms
Performance = 22.8067 TFLOPS
Arithmetic Intensity = 32 FLOPs/Byte
Check = PASS

C[0][0] = 2048
Expected = 2048

Benchmark finished.

V6 : Ampere cp.async + 多级流水线 (N-stage software pipeline)

在 V5 的基础上, 把 "global -> 寄存器 -> shared" 的软件双缓冲
换成硬件异步拷贝指令 cp.async, 并把流水线从 2 级加粗到 STAGES 级。

V5 的做法(目标卡 Turing/sm_75, 没有 cp.async):
    prologue: global -> reg -> shared[0]
    loop t:   1) global -> reg        只发 LDG, 不等
              2) shared -> reg -> FMA 一大段外积, 用来盖住 1) 的延迟
              3) reg -> shared[cur^1] 算完再落盘
              __syncthreads()
  代价: a_reg/b_reg 占用寄存器, 必须 __launch_bounds__(256,2) 压住
        寄存器数, 否则 occupancy 掉到 25%。

V6 的做法(sm_80+):
    prologue: 发出前 STAGES-1 个 tile 的 cp.async, 每个 tile 一组
    loop t:   1) cp.async.wait_group<STAGES-2>  等 tile t 到齐
              2) __syncthreads()
              3) 用 shared[t%STAGES] 做外积 FMA
              4) 把 tile (t+STAGES-1) 用 cp.async 填进空出来的 buffer
              回到 1)
  好处: 数据 global -> shared 不经过寄存器, 寄存器压力骤降;
        在飞(in-flight)的 tile 从 1 个变成 STAGES-1 个, 访存延迟容错
        余量大得多; BK 可以放心调大。

两个坑, 也是和 V5 唯一的实质差别:
  1) cp.async 每次只搬 **连续** 的一段(4/8/16 B)到 shared。
     A 要转置成 As[BK][BM], 而 global 的 16B 连续块是 "一行里的 4 个 k",
     转置后这 4 个数落在 As 的 4 个不同行上 -> 不连续 -> 只能用 4B 拷贝
     (.ca 支持 4/8/16, .cg 只支持 16)。
     B 不需要转置, 16B 块在 global/shared 里都是连续的 -> 直接 16B (.cg)。
     (V5 的 A 路径其实是 1 条 LDG.128 + 4 条 STS.32, V6 用 4 条 cp.async.4
      就替代了, 指令数更少而且不占寄存器。)
  2) wait_group 只保证 "自己这个线程发的拷贝" 完成, 所以等完必须
     __syncthreads() 才能读整块 shared。
  3) 尾部(还剩不到 STAGES-1 个 tile)不再发新 tile 时, 也必须 commit 一个
     "空组"(空组立即完成), 否则 "已提交组数" 会变少, wait<STAGES-2>
     推不出 "tile t 已完成", 可能读到还在搬的数据。

若想 A 也用满 16B 拷贝, 得让 A 的 fragment 改成 "沿 K 的 4 个连续元素"
(即 k 向量化), 或改用 CUTLASS 那种 swizzle 布局 —— 那是下一步 V7 的事。
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cstdint>

using namespace std;

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;\
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

int M, N, K;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8 * 2;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int BLOCK_SIZE = 256;

// 流水线级数。3~4 级在 A100 上最甜: 每多一级就多一个 tile 的时间去
// 掩盖访存延迟, 但 shared 用量随之线性增长。
//   A: BK*(BM+4)*4 = 4224 B/级   B: BK*BN*4 = 4096 B/级  => 8320 B/级
//   STAGES=4 => 33.3 KB, 2 个 block/SM 需要 66.6 KB (A100 每 SM 164 KB, 够)
//   STAGES=3 => 25.0 KB, 2 个 block/SM 需要 50.0 KB
// 注意 Turing 每 SM 只有 64 KB shared: STAGES=4 时 66.6 KB 放不下两个
// block, 只会驻留 1 个 (occupancy 25%) -> 在 sm_75 上应该换 STAGES=3。
constexpr int STAGES = 4;
static_assert(STAGES >= 2, "STAGES >= 2");

#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

// ============================================================
// cp.async 封装
//
// sm_80+  : 真正的异步拷贝 (LDGSTS), 不经过寄存器
// sm_75 及以下: 退化成同步的 global -> shared 拷贝, 语义仍然正确
//              (流水线还在, 只是没有任何重叠, 会比 V5 慢)
// ============================================================
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 4 字节: A 转置加载用, .cg 不支持 4B 所以用 .ca
__device__ __forceinline__ void cp_async_4(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"(smem_u32(smem_dst)), "l"(gmem_src));
#else
    *reinterpret_cast<float*>(smem_dst) =
        *reinterpret_cast<const float*>(gmem_src);
#endif
}

// 16 字节: B 用, 只过 L2 不过 L1
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_u32(smem_dst)), "l"(gmem_src));
#else
    *reinterpret_cast<float4*>(smem_dst) =
        *reinterpret_cast<const float4*>(gmem_src);
#endif
}

// 把 "上次 commit 以来发出的所有 cp.async" 打包成一组
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n");
#endif
}

// 等到 "未完成的组数 <= N" 才继续 (N 必须是编译期常量)
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// ============================================================
// V6 : cp.async + STAGES 级流水线
//
// Block Tile = 128 x 128, Block Size = 256 (8 warps)
// Warp Tile  = 32 x 64,  线程布局 4 x 8 (z 字排列), 每线程 8 x 8
// 计算部分与 V5 完全一致, 只是 shared 多了 stage 这一维。
// ============================================================
template <int BM, int BN, int BK, int TM, int TN, int STAGES>
__global__ void __launch_bounds__(BLOCK_SIZE, 2) sgemm_async_pipeline
    (const float* __restrict__ A, const float* __restrict__ B,
     float* __restrict__ C, int M, int N, int K) {

    // STAGES 块 buffer 轮转。A 转置存放, +4 是为了错开 bank。
    __shared__ float As[STAGES][BK][BM + 4];
    __shared__ float Bs[STAGES][BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;

    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;

    const int lane_row = (lane_id & 1) + (lane_id >> 4) * 2;
    const int lane_col = (lane_id & 15) >> 1;

    float a_frag[2][4], b_frag[2][4];
    float c_frag[4][4][4] = {0.0f};

    const int num_tiles = (K + BK - 1) / BK;

    // 每个线程要发的拷贝条数
    constexpr int A_ELEMS = (BM * BK + BLOCK_SIZE - 1) / BLOCK_SIZE;      // 4B 粒度, =4
    constexpr int B_VECS  = (BK * BN / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 16B 粒度, =1

    // ------------------------------------------------------------
    // 发出第 t 个 k-tile 的异步拷贝, 落到第 stage 块 buffer
    // 注意: 只 "发出", 不等待。同一 tile 的两条流由一个 commit 收尾,
    //       这样 wait_group 的计数单位就是 "一个 tile"。
    // ------------------------------------------------------------
    auto load_tile = [&](int t, int stage) {
        const int bk = t * BK;

        // ---- A: (BM x BK) 逐元素转置搬进 As[stage][col][row] ----
        // 线程分配 e = tid + i*BLOCK_SIZE, row = e/BK, col = e%BK:
        //   global 侧: 一个 warp 里 8 个线程共享一行, 读 8 个连续 float
        //              = 32B, 正好一个 sector, 合并访存满分
        //   shared 侧: 写地址 = col*(BM+4) + row, bank = (4*col+row)%32,
        //              warp 内 32 个地址刚好铺满 32 个 bank, 无冲突
        #pragma unroll
        for (int i = 0; i < A_ELEMS; ++i) {
            const int e = tid + i * BLOCK_SIZE;
            if (t < num_tiles && e < BM * BK) {
                const int row = e / BK, col = e % BK;
                const int g_row = block_row + row, g_col = bk + col;
                if (g_row < M && g_col < K)
                    cp_async_4(&As[stage][col][row], &A[g_row * K + g_col]);
                else
                    As[stage][col][row] = 0.0f;
            }
        }

        // ---- B: (BK x BN) 不转置, 直接 16B 拷贝 ----
        #pragma unroll
        for (int i = 0; i < B_VECS; ++i) {
            const int idx = 4 * (tid + i * BLOCK_SIZE);
            if (t < num_tiles && idx < BK * BN) {
                const int row = idx / BN, col = idx % BN;
                const int g_row = bk + row, g_col = block_col + col;
                if (g_row < K && g_col + 3 < N) {
                    cp_async_16(&Bs[stage][row][col], &B[g_row * N + g_col]);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        Bs[stage][row][col + j] =
                            (g_row < K && g_col + j < N) ? B[g_row * N + g_col + j] : 0.0f;
                }
            }
        }

        // 即使 t 越界也要提交一个"空组": 空组立即完成, 这样
        // "一个组 = 一个 tile" 的计数在尾部依然成立, wait<STAGES-2>
        // 才能始终推出 "tile t 已完成"。
        cp_async_commit();
    };

    // ------------------------------------------------------------
    // prologue: 把前 STAGES-1 个 tile 全部发出去
    // 之后每轮循环只需要 "等一个、算一个、发一个"
    // ------------------------------------------------------------
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        load_tile(s, s);

    // ------------------------------------------------------------
    // 主循环
    // ------------------------------------------------------------
    for (int t = 0; t < num_tiles; ++t) {
        const int cur = t % STAGES;

        // 此刻一共发出过 t + STAGES - 1 组, 等 "未完成 <= STAGES-2"
        // 就意味着 tile 0..t 全部到位。
        cp_async_wait<STAGES - 2>();
        // wait_group 只管本线程; 必须同步后其他线程写的部分才可见。
        // 同时这次同步也保证了上一轮读完的 buffer 可以安全覆写。
        __syncthreads();

        // ---- 计算: 与 V5 完全相同, 只是访问第 cur 块 buffer ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int thread_row = warp_row * 4 * TM + lane_row * 4 + z * 16;
                const int thread_col = warp_col * 8 * TN + lane_col * 4 + z * 32;

                FLOAT4(a_frag[z][0]) = FLOAT4(As[cur][k][thread_row]);
                FLOAT4(b_frag[z][0]) = FLOAT4(Bs[cur][k][thread_col]);
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    #pragma unroll
                    for (int z = 0; z < 4; ++z)
                        c_frag[z][i][j] += a_frag[z / 2][i] * b_frag[z % 2][j];
        }

        // 把 tile (t+STAGES-1) 填进本轮算完的那个 buffer
        // ((t+STAGES-1) % STAGES == (t-1) % STAGES, 正是上一轮读过的)
        load_tile(t + STAGES - 1, (t + STAGES - 1) % STAGES);
    }

    // ---- 写回 Global Memory: 与 V5 相同 ----
    #pragma unroll
    for (int z = 0; z < 4; ++z) {
        const int t_r = warp_row * 4 * TM + lane_row * 4 + (z / 2) * 16;
        const int t_c = warp_col * 8 * TN + lane_col * 4 + (z % 2) * 32;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = block_row + t_r + i;
            if (r >= M) continue;

            const int c0 = block_col + t_c;
            if (N % 4 == 0 && c0 + 3 < N) {
                FLOAT4(C[r * N + c0]) = make_float4(c_frag[z][i][0], c_frag[z][i][1],
                                                    c_frag[z][i][2], c_frag[z][i][3]);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (c0 + j < N)
                        C[r * N + c0 + j] = c_frag[z][i][j];
            }
        }
    }
}

// ============================================================
// CPU 版本，用于验证 CUDA 计算结果
// ============================================================
void sgemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = sum;
        }
    }
}

// ============================================================
// 检查结果
// ============================================================
bool check_result(const float* C, float* C_relif, int M, int N) {
    const float tolerance = 1e-3f;

    for (int i = 0; i < M * N; i++) {
        float diff = fabs(C[i] - C_relif[i]);
        if (diff > tolerance) {
            cerr << "Mismatch at index " << i << ", GPU = " << C[i]
                 << ", CPU = " << C_relif[i] << ", diff = " << diff << endl;
            return false;
        }
    }
    return true;
}

int main() {

    M = N = K = 8 * 1024;

    cout << "M = " << M << endl;
    cout << "N = " << N << endl;
    cout << "K = " << K << endl;
    cout << endl;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_C_ref = (float*)malloc(size_C);

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        cerr << "Host memory allocation failed!" << endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; i++) h_C_ref[i] = 1.0f * K;

    cout << "Computing CPU reference..." << endl;
    // sgemm_cpu(h_A, h_B, h_C_ref, M, N, K);
    cout << "CPU reference finished." << endl;

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cout << "block = " << block.x << endl;
    printf("grid = (%d, %d)\n", grid.x, grid.y);
    cout << "STAGES = " << STAGES << endl;

    // Kernel 第一次运行 (warm-up)
    sgemm_async_pipeline<BM, BN, BK, TM, TN, STAGES>
        <<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 500;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
        sgemm_async_pipeline<BM, BN, BK, TM, TN, STAGES>
            <<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    float avg_ms = total_ms / iterations;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double seconds = avg_ms / 1000.0;
    double tflops = flops / seconds / 1e12;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    bool pass = check_result(h_C, h_C_ref, M, N);

    cout << endl;
    cout << "============================================" << endl;
    cout << "                Result" << endl;
    cout << "============================================" << endl;

    cout << "Average Kernel Time = " << avg_ms << " ms" << endl;
    cout << "Performance = " << tflops << " TFLOPS" << endl;

    double tile_flops = 2.0 * (double)BM * (double)BN * (double)BK;
    double memory_bytes = ((double)BM + (double)BN) * (double)BK * sizeof(float);
    double arithmetic_intensity = tile_flops / memory_bytes;

    cout << "Arithmetic Intensity = " << arithmetic_intensity << " FLOPs/Byte" << endl;
    cout << "Check = " << (pass ? "PASS" : "FAIL") << endl;

    cout << endl;
    cout << "C[0][0] = " << h_C[0] << endl;
    cout << "Expected = " << h_C_ref[0] << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    cout << endl;
    cout << "Benchmark finished." << endl;

    return 0;
}
