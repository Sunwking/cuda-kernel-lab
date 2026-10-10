/*
M = 32768
D = 4096
N = 134217728
Data size = 512 MiB (in) + 512 MiB (out)
Block size = 256
Grid size = 32768
Computing CPU reference...
CPU reference finished.
Result   = -939833.9987
Expected = -939834.3395
Max error = 9.536743164e-07
Check: PASS
Kernel time = 1.433107138 ms
Effective bandwidth = 749.2404411 GB/s
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
// V2: 在 V1 的基础上，单遍 merge 计算 sum/sumsq，并用 float4 向量化 + shared 缓存中间结果
// V3: 在 V2 的基础上，把中间结果从 shared memory 挪到寄存器
//     - 每个线程用 ITEMS_MAX 个 float4 缓存自己负责的 16 个元素，x 全程只读一次
//     - 单遍 merge：sum / sum_sq 在同一次寄存器遍历里算完
//     - float4 向量化 load / store，同时省掉 16 KB 的 shared 占用和 4 次 shared 读写
constexpr int ITEMS_MAX = 4;   // 每线程缓存的 float4 个数 = D4 / blockDim = 1024 / 256

__global__ void layernorm_v3_kernel(
    float* in, const float* gamma, const float* beta,
    float* out, int D, float eps){
    __shared__ float sm[32];
    __shared__ float sm_sq[32];

    int row = blockIdx.x; // 每个block处理一行
    int tid = threadIdx.x; // 每个线程处理一列
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    float* x = in + (size_t)row * D;
    float* y = out + (size_t)row * D;

    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4  = reinterpret_cast<const float4*>(beta);

    int D4 = D >> 2;
    int items = D4 / blockDim.x;   // 每个线程负责几个 float4

    // 1. 一次载入寄存器，同时算 sum 与 sum_sq（单遍 merge）
    float4 v[ITEMS_MAX];
    float sum = 0.0f, sum_sq = 0.0f;
    #pragma unroll
    for(int k = 0 ; k < ITEMS_MAX ; k ++){
        v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if(k < items){
            v[k] = x4[tid + k * blockDim.x];
            sum    += (v[k].x + v[k].y) + (v[k].z + v[k].w);
            sum_sq += (v[k].x * v[k].x + v[k].y * v[k].y)
                    + (v[k].z * v[k].z + v[k].w * v[k].w);
        }
    }

    // 2. 两级规约：warp 内 shuffle + 跨 warp
    sum    = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    if(lane_id == 0){
        sm[warp_id]    = sum;
        sm_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    int num_warps = blockDim.x >> 5;

    if(warp_id == 0){
        sum    = (tid < num_warps) ? sm[tid]    : 0.0f;
        sum_sq = (tid < num_warps) ? sm_sq[tid] : 0.0f;
        sum    = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);
    }

    if(tid == 0){
        sm[0]    = sum;
        sm_sq[0] = sum_sq;
    }
    __syncthreads();

    float mean    = sm[0] / D;
    float var     = sm_sq[0] / D - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // 3. 用寄存器里的 x 直接归一化写出（x 全程只读了一次）
    #pragma unroll
    for(int k = 0 ; k < ITEMS_MAX ; k ++){
        if(k < items){
            float4 g = gamma4[tid + k * blockDim.x];
            float4 b = beta4 [tid + k * blockDim.x];
            float4 r;
            r.x = (v[k].x - mean) * inv_std * g.x + b.x;
            r.y = (v[k].y - mean) * inv_std * g.y + b.y;
            r.z = (v[k].z - mean) * inv_std * g.z + b.z;
            r.w = (v[k].w - mean) * inv_std * g.w + b.w;
            y4[tid + k * blockDim.x] = r;
        }
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

    // V3 的适用条件：D 能被 4 整除，且每行的 float4 能整分给 block，
    // 每线程缓存的 float4 数不超过 ITEMS_MAX（寄存器上限）
    if(D % 4 != 0 || (D / 4) % blockSize != 0 || (D / 4) / blockSize > ITEMS_MAX){
        cerr << "V3 kernel requires D % 4 == 0, (D/4) % blockSize == 0 and "
             << "(D/4)/blockSize <= " << ITEMS_MAX << endl;
        return EXIT_FAILURE;
    }

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
    layernorm_v3_kernel<<<gridSize, blockSize>>>(d_in, d_gamma, d_beta, d_out, D, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0 ; i < iterations ; i ++)
        layernorm_v3_kernel<<<gridSize, blockSize>>>(d_in, d_gamma, d_beta, d_out, D, eps);
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