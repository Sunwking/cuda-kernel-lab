/*
M = 4096
N = 4096

Computing CPU reference...
CPU reference finished.
block = 1024
grid = (4096)

============================================
                Result
============================================
Average Kernel Time = 0.18277 ms
Performance = 458.972 GFLOPS
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
// constexpr int BLOCK_SIZE = 256;
constexpr int BLOCK_SIZE = 1024;

// V4：在 V3 的基础上进行
// 两边 Kernel (Pass 1: 求 max, Pass 2 融合 exp-sum + normalize)
// 需要额外 Shared Memory 暂存 exp 值

// Warp 内进行 Max 规约
__device__ float Warp_reduce_Max(float val){
    for(int offset = 16 ; offset ; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp 内进行 Sum 规约
__device__ float Warp_reduce_Sum(float val){
    for(int offset = 16 ; offset ; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 两级规约: Block 内进行 Max规约
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

    // 广播给所有进程
    __shared__ float block_result;
    if(tid == 0) block_result = val;
    __syncthreads();
    return block_result;
}

// 两级规约：Block 内进行 Sum 规约
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

__global__ void softmax_v4(float* in, float* out, int M, int N) {
    
    int tid = threadIdx.x;
    int row = blockIdx.x;

    float* x = in + row * N;
    float* y = out + row * N;

    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4;

    // 用寄存器数组记录每个线程的值
    float4 data[4];
    int idx = 0;
    // Pass 1: 向量化求行最大值
    float local_max = -INFINITY;
    #pragma unroll
    for(int i = tid ;  i < N4 ; i += blockDim.x){
        data[idx] = x4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(data[idx].x, data[idx].y),
                                           fmaxf(data[idx].z, data[idx].w)));
        idx ++;
    }
    
    float max_val = Block_reduce_Max(local_max);

    // Pass 2: 计算 exp 并暂存，同时累加 Sum
    float local_sum = 0.0f;
    for(int i = tid ; i < N4 ; i += blockDim.x){
        int idx = i / blockDim.x;
        data[idx].x = expf(data[idx].x - max_val);
        data[idx].y = expf(data[idx].y - max_val);
        data[idx].z = expf(data[idx].z - max_val);
        data[idx].w = expf(data[idx].w - max_val);

        local_sum += data[idx].x + data[idx].y + data[idx].z + data[idx].w;
    }    
    float sum = Block_reduce_Sum(local_sum);

    // 直接从寄存器数组 中读取值并归一化写出
    float inv_sum = 1.0f / sum;
    for(int i = tid ; i < N4 ; i += blockDim.x){
        int idx = i / blockDim.x;
        float4 result;
        result.x = data[idx].x * inv_sum;
        result.y = data[idx].y * inv_sum;
        result.z = data[idx].z * inv_sum;
        result.w = data[idx].w * inv_sum;
        y4[i] = result;
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
    size_t size = (size_t)N * sizeof(float);

    float* h_in = (float*)malloc(size_in);
    float* h_out = (float*)malloc(size_out);
    float* h_out_ref = (float*)malloc(size_out);

    if (!h_in || !h_out || !h_out_ref) {
        cerr << "Host memory allocation failed!" << endl;
        return EXIT_FAILURE;
    }

    // 初始化：每一行的输入值都不相同，确保 softmax 的结果有意义
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

    softmax_v4<<<grid, block, size>>>(d_in, d_out, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 200;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        softmax_v4<<<grid, block, size>>>(d_in, d_out, M, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

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
