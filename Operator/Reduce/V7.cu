/*
Data size = 512 MiB
Block size = 256
Grid size = 2048
Result   = 134217728
Expected = 134217728
Check: PASS
Kernel time = 0.648096025 ms
Effective bandwidth = 828.3817387 GB/s
*/

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

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

__device__ float warp_reduce_sum(float val){
    // 每次将右半边的值加到左半边
    for(int offset = 16 ; offset > 0 ; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val; // lane 0 为结果
}

// V7: Warp Shuffle + 两级规约
// 采用 Grid-Stride Loop
__global__ void reduce_v7(const float* in, float* out, int N){
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;
    
    int stride = blockDim.x * gridDim.x;

    float val = 0.0f;
    for(int idx = gid ; idx < N ; idx += stride)
        val += in[idx];

    // 第一级: Warp 内规约
    val = warp_reduce_sum(val);

    // 将每个 Warp 的结果（仅 lane 0 有效）存入 Shared Memory
    __shared__ float warp_results[32]; // 最多 32 个 Warp(1024 / 32)

    int lane_id = tid % 32;
    int Warp_id = tid / 32;
    if(lane_id == 0){
        warp_results[Warp_id] = val;
    }

    __syncthreads();

    // 第二级：Warp 间规约（用 Warp 0 处理）
    int num_warps = blockDim.x / 32;
    if(Warp_id == 0){
        val = (lane_id < num_warps) ? warp_results[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }

    if(tid == 0) out[blockIdx.x] = val;
}

int main(){
    int N = 1 << 27;

    size_t size = N * sizeof(float);

    cout << "N = " << N << endl;
    cout << "Data size = "<< size / (1024.0 * 1024.0) << " MiB" << endl;

    float* h_in = new float[N];
    for(int i = 0 ; i < N ; i ++ )
        h_in[i] = 1.0f;

    int blockSize = 256;
    int gridSize = (N + 2 * blockSize - 1) / (2 * blockSize);
    //如果gridsize为 1024 呢
    if(gridSize > 1024) gridSize = 3 * 1024;

    size_t outSize = static_cast<size_t>(gridSize) * sizeof(float);

    float* h_out = new float[gridSize];
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, outSize));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    cout << "Block size = " << blockSize << endl;

    cout << "Grid size = " << gridSize << endl;

    // Warm up
    reduce_v7<<<gridSize, blockSize>>> (d_in, d_out, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for(int i = 0 ; i < 100 ; i ++ )
        reduce_v7<<<gridSize, blockSize>>> (d_in, d_out, N);
    // reduce_v7<<<gridSize, blockSize>>> (d_in, d_out, N);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, outSize, cudaMemcpyDeviceToHost));

    double result = 0.0;
    for(int i = 0 ; i < gridSize ; i ++ )
        result += h_out[i];

    std::cout.precision(10);
    double expected = static_cast<double>(N);

    cout << "Result   = " << result << endl;
    cout << "Expected = " << expected << endl;

    double error = fabs(result - expected);

    if (error < 1e-3) {
        cout << "Check: PASS" << endl;
    } else {
        cout << "Check: FAIL" << endl;
    }
    
    milliseconds /= 100.0f; // 平均每次的时间

    std::cout << "Kernel time = "<< milliseconds<< " ms" << std::endl;

    double bandwidth = static_cast<double>(size) / (milliseconds * 1e6);

    cout << "Effective bandwidth = "<< bandwidth << " GB/s" << endl;


    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    delete[] h_in;
    delete[] h_out;

    return 0;
}
