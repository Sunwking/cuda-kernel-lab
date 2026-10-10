/*
M = 4096
N = 4096

Computing CPU reference...
CPU reference finished.
block = 1024
grid = (4096)
V4 register-cached path = YES

============================================
                Result
============================================
Average Kernel Time = 0.178355 ms
Performance = 470.333 GFLOPS
AR = 0.625 FLOPs/Byte
Check = PASS
Row 0: first = 2.30626e-07, last = 0.000687487
Expected first = 2.30626e-07, Expected last = 0.000687486

Benchmark finished.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdint>

using namespace std;

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int M, N;
constexpr int BLOCK_SIZE = 1024;

// 每个线程最多缓存的 float4 个数（决定 N 的上限：4 * BLOCK_SIZE * ITEMS_MAX）
constexpr int ITEMS_MAX = 8;

// ---------------- 规约工具（与 V2/V3 相同）----------------

__device__ float Warp_reduce_Max(float val){
    for(int offset = 16 ; offset ; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ float Warp_reduce_Sum(float val){
    for(int offset = 16 ; offset ; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float Block_reduce_Max(float val){
    __shared__ float Warp_Max[32];

    int tid = threadIdx.x;
    int Warp_id = tid >> 5;
    int lane_id = tid & 31;

    val = Warp_reduce_Max(val);
    if(lane_id == 0) Warp_Max[Warp_id] = val;
    __syncthreads();

    int num_Warps = blockDim.x >> 5;
    val = (lane_id < num_Warps) ? Warp_Max[lane_id] : -INFINITY;
    if(Warp_id == 0) val = Warp_reduce_Max(val);

    __shared__ float block_result;
    if(tid == 0) block_result = val;
    __syncthreads();
    return block_result;
}

__device__ float Block_reduce_Sum(float val){
    __shared__ float Warp_Sum[32];

    int tid = threadIdx.x;
    int Warp_id = tid >> 5;
    int lane_id = tid & 31;

    val = Warp_reduce_Sum(val);
    if(lane_id == 0) Warp_Sum[Warp_id] = val;
    __syncthreads();

    int num_Warps = blockDim.x >> 5;
    val = (lane_id < num_Warps) ? Warp_Sum[lane_id] : 0.0f;
    if(Warp_id == 0) val = Warp_reduce_Sum(val);

    __shared__ float block_result;
    if(tid == 0) block_result = val;
    __syncthreads();

    return block_result;
}

// ---------------- V4：寄存器缓存 + 单遍访存 ----------------
//
// V1/V2/V3 都对 x 读了 3 遍、对 y 写了 1 遍（合计 4 趟访存），
// 而 2080 Ti 的 L2 只有 6MB，装不下 67MB 的 x，三遍读全部落到 DRAM。
// 既然瓶颈是访存而不是计算/规约，唯一有效的优化就是减少访存趟数。
//
// V4 让每个线程把自己负责的 16 个元素（4 个 float4）全部缓存在寄存器里：
//   读 x 一次 -> 寄存器内求 max -> block 规约 -> 寄存器内求 exp/sum
//   -> block 规约 -> 寄存器内缩放 -> 写 y 一次
// 访存从 4 趟降到 2 趟，同时 expf 也只算一次（V1~V3 都算了两遍）。
__global__ void softmax_v4(const float* __restrict__ in, float* __restrict__ out, int M, int N)
{
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const float*  x  = in  + (size_t)row * N;
    float*        y  = out + (size_t)row * N;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4*       y4 = reinterpret_cast<float4*>(y);

    int items = (N >> 2) / blockDim.x;   // 每个线程持有的 float4 个数

    // Pass 1: 一次性载入寄存器，顺便求 local max
    float4 v[ITEMS_MAX];
    float local_max = -INFINITY;
#pragma unroll
    for(int k = 0 ; k < ITEMS_MAX ; ++k){
        v[k] = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        if(k < items){
            v[k] = x4[tid + k * blockDim.x];
            local_max = fmaxf(local_max, fmaxf(fmaxf(v[k].x, v[k].y), fmaxf(v[k].z, v[k].w)));
        }
    }
    float max_val = Block_reduce_Max(local_max);

    // Pass 2: 直接在寄存器里做 exp 并累加（结果原地保存，Pass 3 复用）
    float local_sum = 0.0f;
#pragma unroll
    for(int k = 0 ; k < ITEMS_MAX ; ++k){
        if(k < items){
            v[k].x = expf(v[k].x - max_val);
            v[k].y = expf(v[k].y - max_val);
            v[k].z = expf(v[k].z - max_val);
            v[k].w = expf(v[k].w - max_val);
            local_sum += (v[k].x + v[k].y) + (v[k].z + v[k].w);
        }
    }
    float sum = Block_reduce_Sum(local_sum);

    // Pass 3: 寄存器内缩放后写回，无需再读 x
    float inv_sum = 1.0f / sum;
#pragma unroll
    for(int k = 0 ; k < ITEMS_MAX ; ++k){
        if(k < items){
            v[k].x *= inv_sum;
            v[k].y *= inv_sum;
            v[k].z *= inv_sum;
            v[k].w *= inv_sum;
            y4[tid + k * blockDim.x] = v[k];
        }
    }
}

// ---------------- 通用回退版本（与 V2 完全一致）----------------
// 当 N 不满足 V4 的向量化/寄存器条件时使用，保证任何形状都正确。
__global__ void softmax_generic(float* in, float* out, int M, int N)
{
    int tid = threadIdx.x;
    int row = blockIdx.x;

    float* x = in + row * N;
    float* y = out + row * N;

    float local_max = -INFINITY;
    for(int i = tid ; i < N ; i += blockDim.x)
        local_max = fmaxf(local_max, x[i]);
    float max_val = Block_reduce_Max(local_max);

    float local_sum = 0.0f;
    for(int i = tid ; i < N ; i += blockDim.x)
        local_sum += expf(x[i] - max_val);
    float sum = Block_reduce_Sum(local_sum);

    float inv_sum = 1.0f / sum;
    for(int i = tid ; i < N ; i += blockDim.x)
        y[i] = expf(x[i] - max_val) * inv_sum;
}

// V4 的适用条件：N 能被 4 整除、每行能整分给 block、且每个线程的 float4 数不超上限
inline bool v4_applicable(int N, int block){
    if(N % 4 != 0) return false;
    int n4 = N / 4;
    if(n4 % block != 0) return false;
    int items = n4 / block;
    return items >= 1 && items <= ITEMS_MAX;
}

void softmax_launch(const float* d_in, float* d_out, int M, int N,
                    dim3 grid, dim3 block, bool& fast_path)
{
    fast_path = v4_applicable(N, block.x);
    if(fast_path){
        softmax_v4<<<grid, block>>>(d_in, d_out, M, N);
    } else {
        softmax_generic<<<grid, block>>>((float*)d_in, d_out, M, N);
    }
}

void softmax_cpu(const float* in, float* out, int M, int N) {
    for (int row = 0; row < M; ++row) {
        const float* x = in + row * N;
        float max_val = -INFINITY;
        for (int i = 0; i < N; ++i)
            max_val = fmaxf(max_val, x[i]);

        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
            sum += expf(x[i] - max_val);

        float inv_sum = 1.0f / sum;
        for (int i = 0; i < N; ++i)
            out[row * N + i] = expf(x[i] - max_val) * inv_sum;
    }
}

bool check_result(const float* C, const float* C_ref, int M, int N) {
    const float tolerance = 1e-5f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C[i] - C_ref[i]);
        if (diff > tolerance) {
            cerr << "Mismatch at index (" << i / N << "," << i % N << "), GPU = " << C[i]
                 << ", CPU = " << C_ref[i] << ", diff = " << diff << endl;
            return false;
        }
    }
    return true;
}

int main() {
    M = N = 4096;

    cout << "M = " << M << endl;
    cout << "N = " << N << endl;
    cout << endl;

    size_t size_in = (size_t)M * N * sizeof(float);
    size_t size_out = (size_t)M * N * sizeof(float);

    float* h_in = (float*)malloc(size_in);
    float* h_out = (float*)malloc(size_out);
    float* h_out_ref = (float*)malloc(size_out);

    if (!h_in || !h_out || !h_out_ref) {
        cerr << "Host memory allocation failed!" << endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < M * N; ++i) {
        h_in[i] = static_cast<float>((i % 17) - 8) * 0.5f + static_cast<float>((i / 17) % 7) * 0.25f;
    }

    cout << "Computing CPU reference..." << endl;
    softmax_cpu(h_in, h_out_ref, M, N);
    cout << "CPU reference finished." << endl;

    float* d_in;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size_in));
    CUDA_CHECK(cudaMalloc(&d_out, size_out));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid(M);

    cout << "block = " << block.x << endl;
    printf("grid = (%d)\n", grid.x);

    bool fast_path = false;
    softmax_launch(d_in, d_out, M, N, grid, block, fast_path);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cout << "V4 register-cached path = " << (fast_path ? "YES" : "NO (fallback)") << endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 200;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        softmax_launch(d_in, d_out, M, N, grid, block, fast_path);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

    // V4 的实际访存量：读 1 遍 + 写 1 遍
    double flops = 5.0 * (double)M * (double)N;
    double seconds = avg_ms / 1000.0;
    double performance = flops / seconds / 1e9;  // GFLOPS
    double memory_bytes = 2.0 * (double)M * (double)N * sizeof(float);
    double ar = flops / memory_bytes; // FLOPs/Byte

    CUDA_CHECK(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));

    bool pass = check_result(h_out, h_out_ref, M, N);

    cout << endl;
    cout << "============================================" << endl;
    cout << "                Result" << endl;
    cout << "============================================" << endl;
    cout << "Average Kernel Time = " << avg_ms << " ms" << endl;
    cout << "Performance = " << performance << " GFLOPS" << endl;
    cout << "AR = " << ar << " FLOPs/Byte" << endl;
    cout << "Check = " << (pass ? "PASS" : "FAIL") << endl;
    cout << "Row 0: first = " << h_out[0] << ", last = " << h_out[N - 1] << endl;
    cout << "Expected first = " << h_out_ref[0] << ", Expected last = " << h_out_ref[N - 1] << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    free(h_in);
    free(h_out);
    free(h_out_ref);

    cout << endl;
    cout << "Benchmark finished." << endl;
    return 0;
}
