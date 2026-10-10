/*
M = 4096
N = 4096

Computing CPU reference...
CPU reference finished.
block = 256
grid = (4096)

============================================
                Result
============================================
Average Kernel Time = 0.18176 ms
Performance = 461.521 GFLOPS
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
constexpr int BLOCK_SIZE = 256;

// V3：在 V2 的基础上进行 Float4 向量化

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

// V3: 在 V2(Warp 和 block 内分别进行规约) 的基础上进行 Float4 向量化 
__global__ void softmax_v3(float* in, float* out, int M, int N) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    float* x = in + row * N;
    float* y = out + row * N;

    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4;

    // Pass 1: 向量化求行最大值
    float local_max = -INFINITY;
    for(int i = tid ;  i < N4 ; i += blockDim.x){
        float4 data = x4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(data.x, data.y), fmaxf(data.z, data.w)));
    }
    for(int i = N4 * 4 + tid ; i < N ; i += blockDim.x){
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = Block_reduce_Max(local_max);

    // Pass 2: 向量化求指数和
    float local_sum = 0.0f;
    for(int i = tid ; i < N4 ; i += blockDim.x){
        float4 data = x4[i];
        local_sum += expf(data.x - max_val) + expf(data.y - max_val) + 
                     expf(data.w - max_val) + expf(data.z - max_val);
    }
    for(int i = N4 * 4 + tid ; i < N ; i += blockDim.x)
        local_sum += expf(x[i] - max_val);

    float sum = Block_reduce_Sum(local_sum);

    // Pass 3: 向量化归一化
    float inv_sum = 1.0f / sum;
    for(int i = tid ; i < N4 ; i += blockDim.x){
        float4 data = x4[i];
        float4 result;
        result.x = expf(data.x - max_val) * inv_sum;
        result.y = expf(data.y - max_val) * inv_sum;
        result.z = expf(data.z - max_val) * inv_sum;
        result.w = expf(data.w - max_val) * inv_sum;
        y4[i] = result;
    }
    for(int i = N4 * 4 + tid ; i < N ; i += blockDim.x)
        y[i] = expf(x[i] - max_val) * inv_sum;
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
    //size_t size = (size_t)BLOCK_SIZE * sizeof(float);

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

    softmax_v3<<<grid, block>>>(d_in, d_out, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 200;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        softmax_v3<<<grid, block>>>(d_in, d_out, M, N);
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
