/*
N = 134217728
Data size = 512 MiB
Block size = 256
Grid size = 262144
Result   = 134217728
Expected = 134217728
Check: PASS
Kernel time = 0.6481418014 ms
Effective bandwidth = 828.3232325 GB/s
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

// 辅助函数：展开最后 32 个线程的规约
// warpReduce 中的 smem 必须声明为 volatile
// 防止编译器将中间结果缓存到寄存器，导致其他线程读取到旧值
__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}

// V4: 展开最后一个 Warp(在 V3 的基础上)
// 最后 s <= 32 有 6 轮：32、16、8、4、2、1
// 使用 Warp 内展开代替 Block 级 __syncthreads()
__global__ void reduce_v4(const float* in, float* out, int N){
    extern __shared__ float smem[];

    //int gid = (blockDim.x) * blockIdx.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;
    int gid = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // 每个线程加载 2 个相距 blockDim.x 的元素并求和
    float val = 0.0f;
    if(gid < N) val += in[gid];
    if(gid + blockDim.x < N) val += in[gid + blockDim.x];
    //for(int idx = gid ; idx < N ; idx += stride)
    //    val += in[idx];

    smem[tid] = val;

    __syncthreads();

    // 步长从 blockDim.x / 2 开始，每轮减半
    // 保证了每个 Warp 内 32 个线程访问 32 个不同的 Bank
    for(int s = blockDim.x / 2 ; s > 32 ; s >>= 1){
        if(tid < s){
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    
    // 最后一个 Warp 内的规约，无需 __syncthreads()
    if(tid < 32){
        warpReduce(smem, tid);
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
    reduce_v4<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0 ; i < 100 ; i ++ )
        reduce_v4<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, N);
    // reduce_v4<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, N);

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
    
    milliseconds /= 100; // 平均每次 kernel 的时间
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
