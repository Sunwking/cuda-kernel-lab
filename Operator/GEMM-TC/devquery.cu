// Operator/GEMM-TC/devquery.cu
// 把本机（RTX 5070 Ti / sm_120）的硬约束打出来，用来算 occupancy。
// 编译: nvcc -O2 -arch=native devquery.cu -o devquery.exe
#include <cuda_runtime.h>
#include <cstdio>
int main(){
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p,0);
    printf("name                : %s  sm_%d%d\n", p.name, p.major, p.minor);
    printf("SM count            : %d\n", p.multiProcessorCount);
    printf("regsPerSM           : %d\n", p.regsPerMultiprocessor);
    printf("sharedPerSM         : %d B (%.0f KB)\n", (int)p.sharedMemPerMultiprocessor, p.sharedMemPerMultiprocessor/1024.0);
    printf("sharedPerBlock      : %d B (%.0f KB)\n", (int)p.sharedMemPerBlock, p.sharedMemPerBlock/1024.0);
    printf("sharedPerBlockOptin : %d B (%.0f KB)\n", (int)p.sharedMemPerBlockOptin, p.sharedMemPerBlockOptin/1024.0);
    printf("maxThreadsPerSM     : %d\n", p.maxThreadsPerMultiProcessor);
    printf("maxBlocksPerSM      : %d\n", p.maxBlocksPerMultiProcessor);
    //printf("clockRate           : %.0f MHz\n", p.clockRate/1000.0);
    return 0;
}
