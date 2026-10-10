/*
N = 134217728
Data size = 512 MiB
Block size = 256
Grid size = 524288
Result   = 134217728
Expected = 134217728
Check: PASS
Kernel time = 1.287296057 ms
Effective bandwidth = 417.0531784 GB/s
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

// V1: 采用 stride index 方式，消除 Warp Divergence
// s = 1 ：index = 2·tid，活跃线程是 tid 0~127，恰好 4 个完整 Warp → 无分化
// s = 2：index = 4·tid，活跃线程是 tid 0~63，恰好 2 个完整 Warp → 无分化
// s = 4：活跃线程 tid 0~31，恰好 1 个完整 Warp → 无分化
// s = 8：活跃线程 tid 0~15，同属 Warp 0 → 仅 Warp 0 内分化
// s = 16、32、64、128：活跃线程只剩 Warp 0 内少量线程 → 每轮 1 次分化

// 性能问题：虽然消除了 Warp Divergence
// 但访问 Shared Memmory 存在 Bank Conflict
__global__ void reduce_v1(const float* in, float* out, int N){
    extern __shared__ float smem[];

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    smem[tid] = (gid < N) ? in[gid] : 0.0f;

    __syncthreads();

    for(int stride = 1 ; stride < blockDim.x ; stride *= 2){
        int idx = tid * 2 * stride;
        if(idx < blockDim.x)
            smem[idx] += smem[idx + stride];
        __syncthreads();
    }

    if(tid == 0){
        //产生 GridDim 个 
        out[blockIdx.x] = smem[0];
    }
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
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t outSize = static_cast<size_t>(gridSize) * sizeof(float);

    float* h_out = new float[gridSize];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, outSize));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    cout << "Block size = " << blockSize << endl;

    cout << "Grid size = " << gridSize << endl;

    // Warm up
    reduce_v1<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    reduce_v1<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, N);

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
