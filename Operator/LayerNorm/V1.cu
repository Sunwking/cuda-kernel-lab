/*
M = 32768
D = 4096
N = 134217728
Data size = 512 MiB (in) + 512 MiB (out)
Block size = 256
Grid size = 32768
Computing CPU reference...
CPU reference finished.
Result   = -939834.0447
Expected = -939834.3395
Max error = 9.536743164e-07
Check: PASS
Kernel time = 1.424925327 ms
Effective bandwidth = 753.5425214 GB/s
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

__device__ float warpReduceSum(float val){
    for(int offset = 16 ; offset ; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// V1: 在 V0 的基础上，使用 warpReduceSum 进行 warp 内归约，减少共享内存访问
__global__ void layernorm_v1_kernel(
    float* in, const float* gamma, const float* beta,
    float* out, int D, float eps){
    
    __shared__ float sm[32];
    __shared__ float sm_var[32];

    int row = blockIdx.x; // 每个block处理一行
    int tid = threadIdx.x; // 每个线程处理一列
    int warp_id = tid >> 5; // 每个warp处理一行
    int lane_id = tid & 31; // 每个线程在warp内的索引

    float* x = in + row * D;
    float* y = out + row * D;

    // 1. 计算均值
    float sum = 0.0f;
    for(int i = tid ; i < D ; i += blockDim.x)
        sum += x[i];
    
    sum = warpReduceSum(sum);
    if(lane_id == 0)
        sm[warp_id] = sum;

    __syncthreads();

    int num_warps = blockDim.x >> 5;
    
    if(warp_id == 0){
        sum = (tid < num_warps) ? sm[tid] : 0.0f;
        sum = warpReduceSum(sum);
    }

    if (tid == 0)
        sm[0] = sum;
    __syncthreads();

    float mean = sm[0] / D;

    // 2. 计算方差
    float var_sum = 0.0f;
    for(int i = tid ; i < D ; i += blockDim.x){
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }

    var_sum = warpReduceSum(var_sum);
    if(lane_id == 0)
        sm_var[warp_id] = var_sum;
    __syncthreads();

    int num_warps_var = blockDim.x >> 5;
    
    if(warp_id == 0){
        var_sum = (tid < num_warps_var) ? sm_var[tid] : 0.0f;
        var_sum = warpReduceSum(var_sum);
    }

    if (tid == 0)
        sm_var[0] = var_sum;
    __syncthreads();

    float var = sm_var[0] / D;
    
    float inv_std = rsqrtf(var + eps);
    // 3. 归一化
    for(int i = tid ; i < D ; i += blockDim.x){
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// CPU 参考实现：double 累加，作为正确性基准
void layernorm_cpu(const float* in, const float* gamma, const float* beta,
                   float* out, int M, int D, float eps){
    for(int row = 0 ; row < M ; row ++){
        const float* x = in + (size_t)row * D;
        float* y = out + (size_t)row * D;

        double sum = 0.0;
        for(int i = 0 ; i < D ; i ++) sum += x[i];
        double mean = sum / D;

        double var_sum = 0.0;
        for(int i = 0 ; i < D ; i ++){
            double diff = (double)x[i] - mean;
            var_sum += diff * diff;
        }
        double var = var_sum / D;
        double inv_std = 1.0 / sqrt(var + eps);

        for(int i = 0 ; i < D ; i ++)
            y[i] = (float)(((double)x[i] - mean) * inv_std * gamma[i] + beta[i]);
    }
}

int main(){
    const int M = 32768;      // 行数 = block 数
    const int D = 4096;       // 每行长度
    const long long N = (long long)M * D;
    const float eps = 1e-5f;

    const int blockSize = 256;
    const int gridSize  = M;

    size_t sizeIn  = (size_t)N * sizeof(float);
    size_t sizeOut = (size_t)N * sizeof(float);

    cout << "M = " << M << endl;
    cout << "D = " << D << endl;
    cout << "N = " << N << endl;
    cout << "Data size = " << sizeIn / (1024.0 * 1024.0) << " MiB (in) + "
                            << sizeOut / (1024.0 * 1024.0) << " MiB (out)" << endl;
    cout << "Block size = " << blockSize << endl;
    cout << "Grid size = " << gridSize << endl;

    float* h_in    = new float[N];
    float* h_out   = new float[N];
    float* h_ref   = new float[N];
    float* h_gamma = new float[D];
    float* h_beta  = new float[D];

    // 初始化：x / gamma / beta 用同一个 LCG，保证每次运行可复现
    unsigned int seed = 12345u;
    for(long long i = 0 ; i < N ; i ++){
        seed = seed * 1664525u + 1013904223u;
        h_in[i] = ((seed >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;   // [-1, 1)
    }
    for(int i = 0 ; i < D ; i ++){
        seed = seed * 1664525u + 1013904223u;
        h_gamma[i] = ((seed >> 8) & 0xFFFF) / 65535.0f * 2.0f;       // [0, 2)
        seed = seed * 1664525u + 1013904223u;
        h_beta[i]  = ((seed >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f; // [-1, 1)
    }

    cout << "Computing CPU reference..." << endl;
    layernorm_cpu(h_in, h_gamma, h_beta, h_ref, M, D, eps);
    cout << "CPU reference finished." << endl;

    float *d_in, *d_out, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_in,    sizeIn));
    CUDA_CHECK(cudaMalloc(&d_out,   sizeOut));
    CUDA_CHECK(cudaMalloc(&d_gamma, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta,  D * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in,    h_in,    sizeIn,             cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, D * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  h_beta,  D * sizeof(float),  cudaMemcpyHostToDevice));

    // Warm up
    layernorm_v1_kernel<<<gridSize, blockSize>>>(d_in, d_gamma, d_beta, d_out, D, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0 ; i < iterations ; i ++)
        layernorm_v1_kernel<<<gridSize, blockSize>>>(d_in, d_gamma, d_beta, d_out, D, eps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float milliseconds = total_ms / iterations;

    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeOut, cudaMemcpyDeviceToHost));

    // 逐元素比对：Result/Expected 取全量输出的校验和，Check 看最大绝对误差
    std::cout.precision(10);
    double result = 0.0, expected = 0.0, maxErr = 0.0;
    for(long long i = 0 ; i < N ; i ++){
        result   += h_out[i];
        expected += h_ref[i];
        double diff = fabs((double)h_out[i] - (double)h_ref[i]);
        if(diff > maxErr) maxErr = diff;
    }

    cout << "Result   = " << result << endl;
    cout << "Expected = " << expected << endl;
    cout << "Max error = " << maxErr << endl;

    if (maxErr < 1e-3)
        cout << "Check: PASS" << endl;
    else
        cout << "Check: FAIL" << endl;

    cout << "Kernel time = " << milliseconds << " ms" << endl;

    // 有效带宽 = (读 x + 写 y) / 时间
    double bytes = 2.0 * (double)sizeIn;
    double bandwidth = bytes / (milliseconds * 1e6);
    cout << "Effective bandwidth = " << bandwidth << " GB/s" << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));

    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;
    delete[] h_gamma;
    delete[] h_beta;

    return 0;
}