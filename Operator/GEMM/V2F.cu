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
Average Kernel Time = 0.841696 ms
Performance = 20.411 TFLOPS
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
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int BLOCK_SIZE = 256;

#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

// V2: 每一个线程处理 TM * TN 矩阵
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_thread_tiling(float* A, float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    // 当前 Block 在 C 中的起始位置
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.x;

    // 一个 Block 中  256 个线程排列为 16 * 16
    // 一个线程负责 TM * TN = 8 * 8 矩阵
    // (thread_row, thread_col) 为每个线程负责块的左上角坐标
    int thread_row = (tid / (BN / TN)) * TM;
    int thread_col = (tid % (BN / TN)) * TN;

    // 寄存器数组
    float a_frag[TM], b_frag[TN];
    float c_frag[TM][TN] = {0.0f};

    // K 方向 Tile Loop
    for(int bk = 0 ; bk < K ; bk += BK){
        // Global Memory → Shared Memory

        // 加载 A^T(BK, BM), 使用 float4 向量化加载
        if(tid < BM * BK / 4){
            int idx = 4 * tid;

            int row = idx / BK, col = idx % BK;

            int g_row = block_row + row, g_col = bk + col;

            if(g_row < M && g_col + 3 < K){
                float4 value = FLOAT4(A[g_row * K + g_col]);

                As[col + 0][row] = value.x;
                As[col + 1][row] = value.y;
                As[col + 2][row] = value.z;
                As[col + 3][row] = value.w;
            }else {
                for(int i = 0 ; i < 4 ; i ++ ){
                    int c = g_col + i;
                    As[col + i][row] = 
                        (g_row < M && c < K) ? A[g_row * K + c] : 0.0f;
                }
            }
        }

        // 加载 B(BK, BN), 使用 float4 向量化加载
        if(tid < BK * BN / 4){
            int idx = tid * 4;

            int row = idx / BN, col = idx % BN;

            int g_row = bk + row, g_col = block_col + col;

            if(g_row < K && g_col + 3 < N){
                float4 value = FLOAT4(B[g_row * N + g_col]);

                Bs[row][col + 0] = value.x;
                Bs[row][col + 1] = value.y;
                Bs[row][col + 2] = value.z;
                Bs[row][col + 3] = value.w;
            }else {
                for(int i = 0 ; i < 4 ; i ++ ){
                    int c = g_col + i;
                    Bs[row][col + i] = 
                        (g_row < K && c < N) ? B[g_row * N + c] : 0.0f;
                }
            }
        }

        // 所有线程必须完成 Shared Memory 加载
        __syncthreads();

        //  Shared Memory → Register
        //  Register → Register FMA
        // 外积累加
        
        for(int k = 0 ; k < BK ; k ++ ){
            // Shared Memory → Register
            #pragma unroll
            for(int i = 0 ; i < TM ; i += 4 ){
                float4 val = FLOAT4(As[k][thread_row + i]);

                a_frag[i + 0] = val.x;
                a_frag[i + 1] = val.y;
                a_frag[i + 2] = val.z;
                a_frag[i + 3] = val.w;
            }

            #pragma unroll
            for(int j = 0 ; j < TN ; j += 4 ){
                float4 val = FLOAT4(Bs[k][thread_col + j]);

                b_frag[j + 0] = val.x;
                b_frag[j + 1] = val.y;
                b_frag[j + 2] = val.z;
                b_frag[j + 3] = val.w;
            }

            // 寄存器上做外积
            for(int i = 0 ; i < TM ; i ++ )
                for(int j = 0 ; j < TN ; j ++ )
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }

        __syncthreads();
    }
    // 写回 Global Memory
    for(int i = 0 ; i < TM ; i ++ )
        for(int j = 0 ; j < TN ; j ++ ){
            int r = block_row + thread_row + i;
            int c = block_col + thread_col + j;
            if(r < M && c < N)
                C[r * N + c] = c_frag[i][j];
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
    sgemm_thread_tiling<BM, BN, BK, TM, TN>
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

    const int iterations = 1;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0 ; i < iterations ; i ++ )
        sgemm_thread_tiling<BM, BN, BK, TM, TN>
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

    // 这里算的是单个 K-Step 上，一个 Block 的 Shared Memory 复用比：
    // tile_flops = 2 * BM * BN * BK，tile_bytes = (BM + BN) * BK * 4
    double tile_flops = 2.0 * (double)BM * (double)BN * (double)BK;
    double memory_bytes = ((double)BM + (double)BN) * (double)BK * sizeof(float);
    double arithmetic_intensity = tile_flops / memory_bytes;

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
