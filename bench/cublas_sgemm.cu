// bench/cublas_sgemm.cu
// Purpose: reproducible GEMM baselines for this machine -- FP32 SIMT, TF32 tensor
//          core, and FP16-in/FP32-accumulate tensor core. These are the denominators
//          every hand-written kernel in this repo is measured against.
// Build:   nvcc -O3 -arch=native bench/cublas_sgemm.cu -o bench/cublas_sgemm.exe -lcublas
// Run:     .\bench\cublas_sgemm.exe

#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

enum Mode { FP32, TF32, FP16 };

static const char *mode_name(Mode m) {
    return m == FP32 ? "fp32" : (m == TF32 ? "tf32" : "fp16");
}

struct Stats { double mean, median, min; };

// C(M,N) = A(M,K) * B(K,N), all row-major.
// cuBLAS is column-major, so we compute C^T = B^T * A^T via gemm(N, M, K).
static Stats bench_shape(int M, int N, int K, int warmup, int iters, Mode mode) {
    const size_t elems_a = (size_t)M * K, elems_b = (size_t)K * N, elems_c = (size_t)M * N;
    const bool half_in = (mode == FP16);

    void *a = nullptr, *b = nullptr, *c = nullptr;
    CK(cudaMalloc(&a, half_in ? elems_a * sizeof(__half) : elems_a * sizeof(float)));
    CK(cudaMalloc(&b, half_in ? elems_b * sizeof(__half) : elems_b * sizeof(float)));
    CK(cudaMalloc(&c, elems_c * sizeof(float)));   // accumulator always fp32

    // All-ones inputs; A/B are exactly representable in every mode we test.
    if (half_in) {
        std::vector<__half> ha(elems_a), hb(elems_b);
        for (auto &v : ha) v = __float2half(1.0f);
        for (auto &v : hb) v = __float2half(1.0f);
        CK(cudaMemcpy(a, ha.data(), elems_a * sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b, hb.data(), elems_b * sizeof(__half), cudaMemcpyHostToDevice));
    } else {
        std::vector<float> ha(elems_a, 1.0f), hb(elems_b, 1.0f);
        CK(cudaMemcpy(a, ha.data(), elems_a * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b, hb.data(), elems_b * sizeof(float), cudaMemcpyHostToDevice));
    }

    cublasHandle_t handle;
    CB(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;
    auto launch = [&]() {
        if (half_in) {
            CB(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            b, CUDA_R_16F, N, a, CUDA_R_16F, K, &beta,
                            c, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
        } else {
            CB(cublasSetMathMode(handle, mode == TF32 ? CUBLAS_TF32_TENSOR_OP_MATH
                                                      : CUBLAS_DEFAULT_MATH));
            CB(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           (const float *)b, N, (const float *)a, K, &beta,
                           (float *)c, N));
        }
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

    printf("  %-6s mean %7.2f  median %7.2f  min %7.2f TFLOPS   (best %8.3f ms)\n",
           mode_name(mode), s.mean, s.median, s.min, sorted.front());

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
        Stats f32 = bench_shape(M, N, K, warmup, iters, FP32);
        Stats tf  = bench_shape(M, N, K, warmup, iters, TF32);
        Stats h   = bench_shape(M, N, K, warmup, iters, FP16);
        printf("  fp32 vs SIMT peak : %6.1f%%\n", 100.0 * f32.median / peak_fp32);
        printf("  tf32/fp32         : %6.2fx  (<1.3x means cuBLAS fp32 is NOT on tensor cores)\n",
               tf.median / f32.median);
        printf("  fp16/fp32         : %6.2fx  <-- tensor core headroom on this GPU\n", h.median / f32.median);
        printf("  fp16 vs SIMT peak : %6.1f%%\n\n", 100.0 * h.median / peak_fp32);
    }
    return 0;
}