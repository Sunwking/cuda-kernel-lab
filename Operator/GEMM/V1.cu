/*
M = 2048
N = 2048
K = 2048

Computing CPU reference...
CPU reference finished.
block = 256
grid = (16, 16)

============================================
                Result
============================================
Average Kernel Time = 1.45918 ms
Performance = 11.7736 TFLOPS
Arithmetic Intensity = 32 FLOPs/Byte
Check = PASS

C[0][0] = 2048
Expected = 2048

Benchmark finished.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

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

int M, N, K;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int BLOCK_SIZE = 256;

// V1: Thread Block 级 Tiling 优化
// 将 C(M * N) 划分为若干 BM * BN 的分块
// 每 Block 处理一个 BM * BN 分块
// A 和 B 均沿 K 维度以步长 BK 迭代
// A(M, K) --> A(BM, BK)
// B(K, N) --> B(BK, BN)

template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_block_tiling(float* A, float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.x;

    // 共有 256 个线程
    // 加载 tileA 时的线程重排 为 32 行 8 列
    constexpr int A_BLOCK_X = BK;  // = 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;  // = 32
    int a_thread_x = tid % A_BLOCK_X;
    int a_thread_y = tid / A_BLOCK_X;

    // 加载 tileB 时的线程重排 8 行 32 列
    constexpr int B_BLOCK_X = 32;
    //constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;  // = 8
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;

    // 计算 tileC 时的线程排布（16×16）
    constexpr int C_BLOCK_X = 16;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
    int c_thread_x = tid % C_BLOCK_X;
    int c_thread_y = tid / C_BLOCK_X;

    // 每个线程负责 Tm × Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y;  // = 8
    constexpr int Tn = BN / C_BLOCK_X;  // = 8
    float Ct[Tm][Tn] = {0.0f};

    // K-Loop
    for (int k = 0; k < K; k += BK) {
        // 协作加载 tileA（使用跨步循环覆盖 BM 行）
        #pragma unroll
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {
            int r = block_row + i, c = k + a_thread_x;
            As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }

        // 协作加载 tileB（使用跨步循环覆盖 BN 列）
        #pragma unroll
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
            int r = k + b_thread_y, c = block_col + j;
            Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }

        __syncthreads();

        // 外积方式计算 As × Bs
        #pragma unroll
        for (int p = 0; p < BK; p++) {
            for (int i = 0; i < Tm; i++) {
                int row = c_thread_y + i * C_BLOCK_Y;
                for (int j = 0; j < Tn; j++) {
                    int col = c_thread_x + j * C_BLOCK_X;
                    Ct[i][j] += As[row][p] * Bs[p][col];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
    for (int i = 0; i < Tm; i++) {
        int r = block_row + c_thread_y + i * C_BLOCK_Y;
        for (int j = 0; j < Tn; j++) {
            int c = block_col + c_thread_x + j * C_BLOCK_X;
            if (r < M && c < N) C[r * N + c] = Ct[i][j];
        }
    }
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
bool check_result(
    const float* C, float* C_relif,  int M, int N){
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

int main(){
    
    M = N = K = 2 * 1024;

    cout << "M = " << M << endl;
    cout << "N = " << N << endl;
    cout << "K = " << K << endl;

    cout << endl;

    // --------------------------------------------------------
    // 2. 计算矩阵大小
    // --------------------------------------------------------
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

     // --------------------------------------------------------
    // 3. 分配 Host 内存
    // --------------------------------------------------------

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_C_ref = (float*)malloc(size_C);

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        cerr << "Host memory allocation failed!" << endl;
        return EXIT_FAILURE;
    }

    // --------------------------------------------------------
    // 4. 初始化数据
    // --------------------------------------------------------

    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = 1.0f;
    }

    for(int i = 0 ; i < M * N ; i ++ )
        h_C_ref[i] = 1.0f * K;

    // --------------------------------------------------------
    // 5. CPU 计算 Reference
    // --------------------------------------------------------

    cout << "Computing CPU reference..." << endl;

    //sgemm_cpu(h_A, h_B, h_C_ref, M, N, K); 

    cout << "CPU reference finished." << endl;

    // --------------------------------------------------------
    // 6. 分配 GPU 内存
    // --------------------------------------------------------

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // --------------------------------------------------------
    // 7. Host → Device
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // 8. 设置 CUDA Kernel 配置
    //
    // 一个 Block = 256
    //
    // 每个 Thread 计算 C 的 Tm * Tn 个元素
    // --------------------------------------------------------

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cout << "block = " << block.x << endl;
    printf("grid = (%d, %d)\n", grid.x, grid.y);
    // --------------------------------------------------------
    // 9. Kernel 第一次运行
    //
    // 主要用于 warm-up
    // 避免第一次 CUDA Kernel 启动带来的额外开销
    // --------------------------------------------------------
    sgemm_block_tiling<BM, BN, BK, BLOCK_SIZE>
        <<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // 10. CUDA Event
    // 用于准确测量 GPU Kernel 时间
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------------------------------------------
    // 11. Benchmark
    // --------------------------------------------------------

    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0 ; i < iterations ; i ++ )
        sgemm_block_tiling<BM, BN, BK, BLOCK_SIZE>
            <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

     // --------------------------------------------------------
    // 12. 计算平均 Kernel 时间
    // --------------------------------------------------------

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    float avg_ms = total_ms / iterations;

     // --------------------------------------------------------
    // 13. 计算 FLOPs
    //
    // GEMM:
    //
    // C = A × B
    //
    // 每一个 C 元素：
    //
    // K 次乘法
    // K 次加法
    //
    // ≈ 2K FLOPs
    //
    // 一共 M × N 个元素
    //
    // 所以：
    //
    // FLOPs = 2 × M × N × K
    // --------------------------------------------------------

    double flops = 2.0 * (double)M * (double)N * (double)K;

    // --------------------------------------------------------
    // 14. 计算 TFLOPS
    //
    // avg_ms → 秒
    // --------------------------------------------------------

    double seconds = avg_ms / 1000.0;

    double tflops = flops / seconds / 1e12;

     // --------------------------------------------------------
    // 15. Device → Host
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // 16. 正确性检查
    // --------------------------------------------------------

    bool pass = check_result(h_C, h_C_ref, M, N);

     // --------------------------------------------------------
    // 17. 输出结果
    // --------------------------------------------------------

    cout << endl;
    cout << "============================================" << endl;
    cout << "                Result" << endl;
    cout << "============================================" << endl;

    cout << "Average Kernel Time = "
         << avg_ms
         << " ms" << endl;


    cout << "Performance = "
         << tflops
         << " TFLOPS" << endl;

    double F = 2.0 * (double)BM * (double)BN * (double)BK;
    double memory_bytes = ((double)BM + (double)BN) * (double)BK * sizeof(float);
    double arithmetic_intensity = F / memory_bytes;

    cout << "Arithmetic Intensity = " << arithmetic_intensity << " FLOPs/Byte" << endl;
    cout << "Check = "
         << (pass ? "PASS" : "FAIL")
         << endl;


    // --------------------------------------------------------
    // 18. 打印一个结果
    // --------------------------------------------------------

    cout << endl;

    cout << "C[0][0] = " << h_C[0] << endl;

    cout << "Expected = " << h_C_ref[0] << endl;

    // --------------------------------------------------------
    // 19. 清理 CUDA Event
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    // --------------------------------------------------------
    // 20. 释放 GPU 内存
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));


    // --------------------------------------------------------
    // 21. 释放 Host 内存
    // --------------------------------------------------------

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);


    cout << endl;
    cout << "Benchmark finished." << endl;

    return 0;
}
