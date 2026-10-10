/*
N = 134217728
Data size = 512 MiB
Block size = 256
Grid size = 1024
Result   = 134217728
Expected = 134217728
Check: PASS
Kernel time = 0.6315839887 ms
Effective bandwidth = 850.0388256 GB/s
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

// V8: float4 向量化加载 + Grid Stride Loop + Warp Shuffle
__global__ void reduce_v8(float* in, float* out, int N){
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;    
    int stride = blockDim.x * gridDim.x;

    // float4 加载：每线程每次处理 4 个 float
    float4* in4 = reinterpret_cast<float4*>(in);
    int N4 = N / 4;

    // Grid Stride Loop：每个线程以 gridDim.x * blockDim.x 为步长迭代
    float val = 0.0f;
    for(int idx = gid ; idx < N4 ; idx += stride){
        float4 data = in4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    // 处理 n 不是 4 的倍数时的尾部元素
    int tail_start = N4 * 4;
    for(int idx = tail_start + gid ; idx < N ; idx += stride)
        val += in[idx];

    // 第一级: Warp 内规约
    val = warp_reduce_sum(val);

    // 将每个 Warp 的结果（仅 lane 0 有效）存入 Shared Memory
    __shared__ float warp_results[32]; // 最多 32 个 Warp(1024 / 32)

    int lane_id = tid % 32;
    int Warp_id = tid / 32;
    if(lane_id == 0) warp_results[Warp_id] = val;
    
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

    // 固定 Grid 大小为 SM 数 × 4（最大化 GPU 利用率）
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int gridSize = 1024;
    int blockSize = 256;

    size_t outSize = static_cast<size_t>(gridSize) * sizeof(float);

    float* h_out = new float[gridSize];
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, outSize));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    cout << "Block size = " << blockSize << endl;

    cout << "Grid size = " << gridSize << endl;

    // Warm up
    reduce_v8<<<gridSize, blockSize>>> (d_in, d_out, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    reduce_v8<<<gridSize, blockSize>>> (d_in, d_out, N);

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
    
    std::cout << "Kernel time = "<< milliseconds << " ms" << std::endl;

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
