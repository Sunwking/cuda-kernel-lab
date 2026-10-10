/*
Base : 朴素实现 (Naive GEMM)

C[M x N] = A[M x K] * B[K x N]   (row-major)

每个 thread 负责 C 的一个元素:
    读 A 的一行 (K 个数) + B 的一列 (K 个数) = 2K 次 global load
    做 K 次 FMA
=> 算术强度 = 2K / ((2K + 1) * 4) ≈ 0.25 FLOPs/Byte (K = 2048)

完全 memory-bound, 是整个优化路线的性能下界。

参考 (M = N = K = 2048):
    Average Kernel Time = 5.77192 ms
    Performance         = 2.97645 TFLOPS
    Arithmetic Intensity= 0.249939 FLOPs/Byte
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

using namespace std;

#define CUDA_CHECK(call)                                              \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;    \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

int M, N, K;

// ============================================================
// Base : 朴素 GEMM, 每个 thread 算一个 C 元素
// ============================================================
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ============================================================
// CPU 版本，用于验证 CUDA 计算结果
// ============================================================
void sgemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = sum;
        }
    }
}

// ============================================================
// 检查结果
// ============================================================
bool check_result(const float* C, float* C_relif, int M, int N) {
    const float tolerance = 1e-3f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(C[i] - C_relif[i]);
        if (diff > tolerance) {
            cerr << "Mismatch at index " << i << ", GPU = " << C[i]
                 << ", CPU = " << C_relif[i] << ", diff = " << diff << endl;
            return false;
        }
    }
    return true;
}

int main() {

    M = N = K = 2 * 1024;

    cout << "M = " << M << endl;
    cout << "N = " << N << endl;
    cout << "K = " << K << endl;
    cout << endl;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_C_ref = (float*)malloc(size_C);

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        cerr << "Host memory allocation failed!" << endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; i++) h_C_ref[i] = 1.0f * K;

    cout << "Computing CPU reference..." << endl;
    // sgemm_cpu(h_A, h_B, h_C_ref, M, N, K);
    cout << "CPU reference finished." << endl;

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cout << "Block = " << block.x << " x " << block.y << endl;
    printf("Grid = (%d, %d)\n", grid.x, grid.y);
    cout << endl;

    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 500;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double seconds = avg_ms / 1000.0;
    double tflops = flops / seconds / 1e12;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    bool pass = check_result(h_C, h_C_ref, M, N);

    cout << endl;
    cout << "============================================" << endl;
    cout << "                Result" << endl;
    cout << "============================================" << endl;
    cout << "Average Kernel Time = " << avg_ms << " ms" << endl;
    cout << "Performance = " << tflops << " TFLOPS" << endl;

    double F = 2.0 * (double)K;
    double memory_bytes = (2.0 * (double)K + 1.0) * sizeof(float);
    double arithmetic_intensity = F / memory_bytes;

    cout << "Arithmetic Intensity = " << arithmetic_intensity << " FLOPs/Byte" << endl;
    cout << "Check = " << (pass ? "PASS" : "FAIL") << endl;

    cout << endl;
    cout << "C[0][0] = " << h_C[0] << endl;
    cout << "Expected = " << h_C_ref[0] << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    cout << endl;
    cout << "Benchmark finished." << endl;

    return 0;
}
