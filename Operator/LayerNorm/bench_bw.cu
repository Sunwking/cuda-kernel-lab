#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

__device__ __forceinline__ float warpReduceSum(float v){
#pragma unroll
    for(int o=16;o;o>>=1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// 纯拷贝：测这台机器"读+写"能达到的实测带宽上限
__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n4){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n4) out[i] = in[i];
}

template<int BLOCK>
__global__ void rmsnorm_kernel(const float4* __restrict__ in, const float4* __restrict__ g,
                               float4* __restrict__ out, int D, float eps){
    __shared__ float sm[32];
    const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int D4 = D>>2;                  // 每行 float4 个数 = 1024
    constexpr int PER = (4096/4)/BLOCK;   // 每线程 float4 个数
    const float4* x4 = in  + (size_t)blockIdx.x*D4;
    float4*       y4 = out + (size_t)blockIdx.x*D4;

    float4 reg[PER];
    float sum=0.f;
#pragma unroll
    for(int k=0;k<PER;k++){
        int i = k*BLOCK + tid;
        float4 xi = x4[i];
        reg[k]=xi;
        sum += xi.x*xi.x+xi.y*xi.y+xi.z*xi.z+xi.w*xi.w;
    }
    sum = warpReduceSum(sum);
    if(lane==0) sm[warp]=sum;
    __syncthreads();
    constexpr int NW = BLOCK/32;
    if(warp==0){
        float s = (lane<NW)?sm[lane]:0.f;
        s = warpReduceSum(s);
        if(lane==0) sm[0]=s;
    }
    __syncthreads();
    const float inv = rsqrtf(sm[0]/(float)D + eps);
#pragma unroll
    for(int k=0;k<PER;k++){
        int i=k*BLOCK+tid;
        float4 xi=reg[k], gi=g[i], yi;
        yi.x=gi.x*xi.x*inv; yi.y=gi.y*xi.y*inv; yi.z=gi.z*xi.z*inv; yi.w=gi.w*xi.w*inv;
        y4[i]=yi;
    }
}

template<class F>
float timeit(F f, int iters=50){
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for(int i=0;i<5;i++) f();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for(int i=0;i<iters;i++) f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms/iters;
}

int main(){
    const int M=32768, D=4096;
    const size_t bytes=(size_t)M*D*sizeof(float);
    const double GiB = 2.0*(double)bytes;      // 读+写

    float *in,*out,*g,*tmp;
    CK(cudaMalloc(&in,bytes)); CK(cudaMalloc(&out,bytes));
    CK(cudaMalloc(&tmp,bytes)); CK(cudaMalloc(&g,D*sizeof(float)));
    CK(cudaMemset(in,0,bytes)); CK(cudaMemset(out,0,bytes));

    printf("traffic per iter = %.1f MiB (read+write)\n\n", GiB/(1024.0*1024.0));

    float ms = timeit([&]{ CK(cudaMemcpy(tmp,in,bytes,cudaMemcpyDeviceToDevice)); });
    printf("cudaMemcpy D2D      : %8.4f ms  %8.1f GB/s\n", ms, GiB/(ms*1e6));

    int n4 = M*D/4;
    ms = timeit([&]{ copy_kernel<<<(n4+255)/256,256>>>(reinterpret_cast<const float4*>(in),
                                                      reinterpret_cast<float4*>(tmp), n4); });
    printf("copy kernel (float4): %8.4f ms  %8.1f GB/s\n", ms, GiB/(ms*1e6));

    ms = timeit([&]{ rmsnorm_kernel<256><<<M,256>>>(reinterpret_cast<const float4*>(in),
                        reinterpret_cast<const float4*>(g), reinterpret_cast<float4*>(out), D, 1e-5f); });
    printf("rmsnorm  BLOCK=256  : %8.4f ms  %8.1f GB/s\n", ms, GiB/(ms*1e6));

    ms = timeit([&]{ rmsnorm_kernel<512><<<M,512>>>(reinterpret_cast<const float4*>(in),
                        reinterpret_cast<const float4*>(g), reinterpret_cast<float4*>(out), D, 1e-5f); });
    printf("rmsnorm  BLOCK=512  : %8.4f ms  %8.1f GB/s\n", ms, GiB/(ms*1e6));

    ms = timeit([&]{ rmsnorm_kernel<1024><<<M,1024>>>(reinterpret_cast<const float4*>(in),
                        reinterpret_cast<const float4*>(g), reinterpret_cast<float4*>(out), D, 1e-5f); });
    printf("rmsnorm  BLOCK=1024 : %8.4f ms  %8.1f GB/s\n", ms, GiB/(ms*1e6));

    CK(cudaFree(in)); CK(cudaFree(out)); CK(cudaFree(tmp)); CK(cudaFree(g));
    return 0;
}
