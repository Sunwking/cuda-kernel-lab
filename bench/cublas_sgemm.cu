// bench/cublas_sgemm.cu
// Purpose: produce a reproducible cuBLAS SGEMM baseline you can quote on a resume.
// Build:   nvcc -O3 -arch=native bench/cublas_sgemm.cu -o bench/cublas_sgemm.exe -lcublas
// Run:     .\bench\cublas_sgemm.exe
// Output:  per-shape mean/median/min TFLOPS under fp32 and tf32 math modes.

#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t _e = (x);                                              \
        if (_e != cudaSuccess) {                                           \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(_e));                                \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

#define CB(x)                                                              \
    do {                                                                   \
        cublasStatus_t _s = (x);                                           \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                 \
            printf("cuBLAS error %d at %s:%d\n", (int)_s, __FILE__,        \
                   __LINE__);                                              \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

struct Stats {
    double mean, median, min;
};

// C(M,N) = A(M,K) * B(K,N), all row-major.
// cuBLAS is column-major, so we compute C^T = B^T * A^T via gemm(N, M, K).
static Stats bench_shape(int M, int N, int K, int warmup, int iters,
                         cublasMath_t math_mode, const char *tag) {
    const size_t bytes_a = (size_t)M * K * sizeof(float);
    const size_t bytes_b = (size_t)K * N * sizeof(float);
    const size_t bytes_c = (size_t)M * N * sizeof(float);

    std::vector<float> h_a((size_t)M * K, 1.0f), h_b((size_t)K * N, 1.0f);
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CK(cudaMalloc(&a, bytes_a));
    CK(cudaMalloc(&b, bytes_b));
    CK(cudaMalloc(&c, bytes_c));
    CK(cudaMemcpy(a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b, h_b.data(), bytes_b, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CB(cublasCreate(&handle));
    CB(cublasSetMathMode(handle, math_mode));

    const float alpha = 1.0f, beta = 0.0f;
    auto launch = [&]() {
        CB(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                       b, N, a, K, &beta, c, N));
    };

    for (int i = 0; i < warmup; ++i) launch();
    CK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));

    std::vector<double> ms(iters);
    for (int i = 0; i < iters; ++i) {
        float ms_i = 0.0f;
        CK(cudaEventRecord(start));
        launch();
        CK(cudaEventRecord(stop));
        CK(cudaEventSynchronize(stop));
        CK(cudaEventElapsedTime(&ms_i, start, stop));
        ms[i] = (double)ms_i;
    }

    const double flops = 2.0 * (double)M * N * K;
    auto tflops = [&](double t_ms) { return flops / (t_ms * 1e-3) / 1e12; };

    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    Stats s;
    s.mean = tflops(std::accumulate(ms.begin(), ms.end(), 0.0) / iters);
    s.median = tflops(sorted[iters / 2]);
    s.min = tflops(sorted.front());

    printf("  %-6s mean %6.2f  median %6.2f  min %6.2f TFLOPS   (best %7.3f ms)\n",
           tag, s.mean, s.median, s.min, sorted.front());

    CB(cublasDestroy(handle));
    CK(cudaFree(a));
    CK(cudaFree(b));
    CK(cudaFree(c));
    return s;
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));

    int clock_khz = 0;
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const int sm_count = prop.multiProcessorCount;
    const double clock_ghz = clock_khz * 1e-6;
    // sm_120: 128 FP32 FMA lanes per SM per clock.
    const double peak_fp32 = sm_count * 128.0 * 2.0 * clock_ghz * 1e9 / 1e12;

    printf("GPU            : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("SM / clock     : %d SMs @ %.3f GHz\n", sm_count, clock_ghz);
    printf("FP32 SIMT peak : %.2f TFLOPS\n", peak_fp32);
    printf("CUDA runtime   : %d.%d\n\n", CUDART_VERSION / 1000,
           (CUDART_VERSION % 1000) / 10);

    const int shapes[][3] = {{2048, 2048, 2048}, {4096, 4096, 4096}, {8192, 8192, 8192}};
    const int warmup = 20, iters = 100;

    for (auto &sh : shapes) {
        const int M = sh[0], N = sh[1], K = sh[2];
        printf("M = N = K = %d\n", M);
        Stats fp32 = bench_shape(M, N, K, warmup, iters, CUBLAS_DEFAULT_MATH, "fp32");
        Stats tf32 = bench_shape(M, N, K, warmup, iters, CUBLAS_TF32_TENSOR_OP_MATH, "tf32");
        printf("  fp32 vs SIMT peak : %5.1f%%\n", 100.0 * fp32.median / peak_fp32);
        printf("  tf32/fp32 ratio   : %5.2fx  (<1.3 means cuBLAS fp32 is NOT using tensor cores)\n\n",
               tf32.median / fp32.median);
    }
    return 0;
}