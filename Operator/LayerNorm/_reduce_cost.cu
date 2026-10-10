#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
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

// 1 读 + 1 写（标量）—— 纯访存下限
__global__ void k_copy1(float* in, float* out, int D){
    int row = blockIdx.x;
    float* x = in  + (size_t)row * D;
    float* y = out + (size_t)row * D;
    for(int i = threadIdx.x ; i < D ; i += blockDim.x) y[i] = x[i] * 2.0f;
}

// 1 读 + 1 写（float4）
__global__ void k_copy4(float* in, float* out, int D){
    int row = blockIdx.x;
    int n4  = D >> 2;
    float4* x4 = (float4*)(in  + (size_t)row * D);
    float4* y4 = (float4*)(out + (size_t)row * D);
    for(int i = threadIdx.x ; i < n4 ; i += blockDim.x){
        float4 v = x4[i];
        v.x*=2.f; v.y*=2.f; v.z*=2.f; v.w*=2.f;
        y4[i] = v;
    }
}

// 3 读 + 1 写，但完全不跨线程规约（每线程用自己的局部均值）
__global__ void k_noreduce(float* in, float* out, int D, float eps){
    int row = blockIdx.x, tid = threadIdx.x;
    float* x = in  + (size_t)row * D;
    float* y = out + (size_t)row * D;
    float sum = 0.0f;
    for(int i = tid ; i < D ; i += blockDim.x) sum += x[i];
    float mean = sum / D;
    float vs = 0.0f;
    for(int i = tid ; i < D ; i += blockDim.x){ float d = x[i] - mean; vs += d * d; }
    float inv = rsqrtf(vs / D + eps);
    for(int i = tid ; i < D ; i += blockDim.x) y[i] = (x[i] - mean) * inv;
}

// 3 读 + 1 写 + shared 树形规约（V0）
__global__ void k_tree(float* in, float* out, int D, float eps){
    __shared__ float sm[256];
    __shared__ float sm_var[256];
    int row = blockIdx.x, tid = threadIdx.x;
    float* x = in  + (size_t)row * D;
    float* y = out + (size_t)row * D;
    float sum = 0.0f;
    for(int i = tid ; i < D ; i += blockDim.x) sum += x[i];
    sm[tid] = sum; __syncthreads();
    for(int s = blockDim.x/2 ; s > 0 ; s >>= 1){ if(tid < s) sm[tid] += sm[tid+s]; __syncthreads(); }
    float mean = sm[0] / D;
    float vs = 0.0f;
    for(int i = tid ; i < D ; i += blockDim.x){ float d = x[i] - mean; vs += d * d; }
    sm_var[tid] = vs; __syncthreads();
    for(int s = blockDim.x/2 ; s > 0 ; s >>= 1){ if(tid < s) sm_var[tid] += sm_var[tid+s]; __syncthreads(); }
    float var = sm_var[0] / D;
    float inv = rsqrtf(var + eps);
    for(int i = tid ; i < D ; i += blockDim.x) y[i] = (x[i] - mean) * inv;
}

// 3 读 + 1 写 + warp shuffle 两级规约（V1）
__global__ void k_shfl(float* in, float* out, int D, float eps){
    __shared__ float sm[32];
    __shared__ float sm_var[32];
    int row = blockIdx.x, tid = threadIdx.x;
    int warp_id = tid >> 5, lane_id = tid & 31;
    float* x = in  + (size_t)row * D;
    float* y = out + (size_t)row * D;
    float sum = 0.0f;
    for(int i = tid ; i < D ; i += blockDim.x) sum += x[i];
    sum = warpReduceSum(sum);
    if(lane_id == 0) sm[warp_id] = sum;
    __syncthreads();
    int num_warps = blockDim.x >> 5;
    if(warp_id == 0){
        sum = (tid < num_warps) ? sm[tid] : 0.0f;
        sum = warpReduceSum(sum);
    }
    if(tid == 0) sm[0] = sum;
    __syncthreads();
    float mean = sm[0] / D;
    float vs = 0.0f;
    for(int i = tid ; i < D ; i += blockDim.x){ float d = x[i] - mean; vs += d * d; }
    vs = warpReduceSum(vs);
    if(lane_id == 0) sm_var[warp_id] = vs;
    __syncthreads();
    if(warp_id == 0){
        vs = (tid < num_warps) ? sm_var[tid] : 0.0f;
        vs = warpReduceSum(vs);
    }
    if(tid == 0) sm_var[0] = vs;
    __syncthreads();
    float var = sm_var[0] / D;
    float inv = rsqrtf(var + eps);
    for(int i = tid ; i < D ; i += blockDim.x) y[i] = (x[i] - mean) * inv;
}

template <class F>
void bench(const char* name, F f, int iters, double bytes){
    f(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, t;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&t));
    CUDA_CHECK(cudaEventRecord(s));
    for(int i = 0 ; i < iters ; i ++) f();
    CUDA_CHECK(cudaEventRecord(t));
    CUDA_CHECK(cudaEventSynchronize(t));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, t));
    ms /= iters;
    printf("%-18s %8.4f ms   %8.2f GB/s\n", name, ms, bytes / (ms * 1e6));
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(t));
}

int main(){
    const int M = 32768, D = 4096;
    const long long N = (long long)M * D;
    const float eps = 1e-5f;
    const int blockSize = 256, gridSize = M;
    const int iters = 100;

    size_t sizeIn = (size_t)N * sizeof(float);
    float *h_in  = new float[N];
    unsigned int seed = 12345u;
    for(long long i = 0 ; i < N ; i ++){
        seed = seed * 1664525u + 1013904223u;
        h_in[i] = ((seed >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
    }
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  sizeIn));
    CUDA_CHECK(cudaMalloc(&d_out, sizeIn));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeIn, cudaMemcpyHostToDevice));

    const double B = 2.0 * (double)sizeIn;   // 每趟 1 GiB

    printf("M=%d  D=%d  block=%d  grid=%d  iters=%d\n\n", M, D, blockSize, gridSize, iters);
    bench("copy1  1R+1W", [&]{ k_copy1<<<gridSize,blockSize>>>(d_in, d_out, D); },  iters, B);
    bench("copy4  1R+1W", [&]{ k_copy4<<<gridSize,blockSize>>>(d_in, d_out, D); },  iters, B);
    bench("noreduce",     [&]{ k_noreduce<<<gridSize,blockSize>>>(d_in, d_out, D, eps); }, iters, B);
    bench("tree   (V0)",  [&]{ k_tree<<<gridSize,blockSize>>>(d_in, d_out, D, eps); },     iters, B);
    bench("shfl   (V1)",  [&]{ k_shfl<<<gridSize,blockSize>>>(d_in, d_out, D, eps); },     iters, B);

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    return 0;
}