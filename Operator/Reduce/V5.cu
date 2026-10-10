/*
N = 134217728
Data size = 512 MiB
Block size = 256
Grid size = 262144
Result   = 134217728
Expected = 134217728
Check: PASS
Kernel time = 0.6335999966 ms
Effective bandwidth = 847.334146 GB/s
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

// V5: 完全循环展开（模板参数）
template <int BLOCK_SIZE> 
__global__ void reduce_v5(const float* in, float* out, int N){
    extern __shared__ float smem[];

    //int gid = (blockDim.x) * blockIdx.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;
    int gid = (2 * BLOCK_SIZE) * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // 每个线程加载 2 个相距 BLOCK_SIZE 的元素并求和
    float val = 0.0f;
    if(gid < N) val += in[gid];
    if(gid + BLOCK_SIZE < N) val += in[gid + BLOCK_SIZE];
    //for(int idx = gid ; idx < N ; idx += stride)
    //    val += in[idx];

    smem[tid] = val;

    __syncthreads();

    // 编译期展开：BLOCK_SIZE 已知，不满足的分支会被编译器直接删除
    // 即 例如 BLOCK_SIZE = 256, 下面代码被直接删除
    if (BLOCK_SIZE >= 512) { 
        if (tid < 256) smem[tid] += smem[tid + 256]; 
        __syncthreads(); 
    }
    if (BLOCK_SIZE >= 256) { 
        if (tid < 128) smem[tid] += smem[tid + 128];
        __syncthreads(); 
    }
    if (BLOCK_SIZE >= 128) {
        if (tid <  64) smem[tid] += smem[tid +  64];
        __syncthreads(); 
    }

    // 最后一个 Warp 内的规约，无需 __syncthreads()
    if(tid < 32){
        volatile float* vsmem = smem;
        if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
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
    int gridSize = (N + 2 * blockSize - 1) / (2 * blockSize);
    // 如果gridsize为 1024 呢
    //if(gridSize > 1024) gridSize = 1024;

    size_t outSize = static_cast<size_t>(gridSize) * sizeof(float);

    float* h_out = new float[gridSize];
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, outSize));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    cout << "Block size = " << blockSize << endl;

    cout << "Grid size = " << gridSize << endl;

    // Warm up
    switch (blockSize) {
        case 512: reduce_v5<512><<<gridSize, 512, 512*sizeof(float)>>>(d_in, d_out, N); break;
        case 256: reduce_v5<256><<<gridSize, 256, 256*sizeof(float)>>>(d_in, d_out, N); break;
        case 128: reduce_v5<128><<<gridSize, 128, 128*sizeof(float)>>>(d_in, d_out, N); break;
    }


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    switch (blockSize) {
        case 512: reduce_v5<512><<<gridSize, 512, 512*sizeof(float)>>>(d_in, d_out, N); break;
        case 256: reduce_v5<256><<<gridSize, 256, 256*sizeof(float)>>>(d_in, d_out, N); break;
        case 128: reduce_v5<128><<<gridSize, 128, 128*sizeof(float)>>>(d_in, d_out, N); break;
    }

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
