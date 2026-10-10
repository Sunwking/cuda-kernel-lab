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
Average Kernel Time = 0.818458 ms
Performance = 20.9905 TFLOPS
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

// ============================================================
// V4P : Scatter 4×4 + Shared Memory Padding
//
// 在 V4 的基础上消除 shared memory 的 bank conflict:
//
//   1. As / Bs 的行 stride 由 BM(128) 补到 BM + 4(132)
//      128 % 32 = 0, 导致转置写入 As[col+i][row] 时不同的 k
//      落在同一个 bank; 132 % 32 = 4, 正好错开。
//      补 4 而不是 1, 是为了保持 16 字节对齐, 让 float4 仍然可用。
//   2. B 的 4 次标量写入合并为一次 float4 写入。
//
// Block Tile = 128 × 128
// Block Size = 256 threads = 8 warps
//
// 一个 Warp:
//
//              64
//       ┌───────────────┐
//       │   16×32 │16×32│
//     32│─────────┼─────│
//       │   16×32 │16×32│
//       └─────────┴─────┘
//
// Warp Tile = 32 × 64
//
// 每个 Warp 中线程布局: 4 × 8
// 每个线程负责: 4 个 4×4 子块
//
// 子块位置:
//
//     Block 0: (0,  0)
//     Block 1: (0, 32)
//     Block 2: (16, 0)
//     Block 3: (16,32)
//
// ============================================================

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_warp_tiling_scatter
    (float* A, float* B, float* C, int M, int N, int K) {
    // 行方向补 4 个 float, 打破 128(=4*32) 的 bank 对齐
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];

    // 当前 Block 在 C 中的起始位置
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.x;

    // Warp-level Tiling
    // Warp 编号和 laneID
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    // 一个 Block 分为 256 / 32 = 8 = 4 行 2 列个 Warp
    // 所属 Warp 行列号
    int warp_row = warp_id >> 1;
    int warp_col = warp_id & 1;

    // 一个 Warp 有 32 个线程排列为 4 * 8
    int lane_row = lane_id >> 3;
    int lane_col = lane_id & 7;

    // 寄存器数组
    // 每个线程负责 4 个 4×4 子块，所以片段的第二维是 4（子块边长），
    // 而不是 TM / TN。a_frag[z][i] 是第 z 个行半区的第 i 行，
    // b_frag[z][j] 是第 z 个列半区的第 j 列。
    float a_frag[2][4], b_frag[2][4];
    float c_frag[4][4][4] = {0.0f};

    // K 方向 Tile Loop
    for(int bk = 0 ; bk < K ; bk += BK){
        // Global Memory → Shared Memory

        // 加载 A^T(BK, BM), 使用 float4 向量化加载
        if(tid < BM * BK / 4){
            int idx = 4 * tid;

            int row = idx / BK, col = idx % BK;

            int g_row = block_row + row, g_col = bk + col;

            // K 必须是 4 的倍数, 才能保证 A 的行首地址 16 字节对齐
            if(K % 4 == 0 && g_row < M && g_col + 3 < K){
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

            // N 必须是 4 的倍数, 才能保证 B 的行首地址 16 字节对齐
            if(N % 4 == 0 && g_row < K && g_col + 3 < N){
                // 一次 float4 写入 shared: warp 内地址连续, 不再有 bank conflict
                FLOAT4(Bs[row][col]) = FLOAT4(B[g_row * N + g_col]);
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
        #pragma unroll
        for(int k = 0 ; k < BK ; k ++ ){
            // Shared Memory → Register
            // 每个线程的 4 个 4×4 子块在 Warp Tile(32×64) 中的位置:
            //   z = 0 : (lane_row*4,      lane_col*4)
            //   z = 1 : (lane_row*4,      lane_col*4 + 32)
            //   z = 2 : (lane_row*4 + 16, lane_col*4)
            //   z = 3 : (lane_row*4 + 16, lane_col*4 + 32)
            #pragma unroll
            for(int z = 0 ; z < 2 ; z ++ ){
                // z 同时表示行半区(上/下)与列半区(左/右)
                int thread_row = warp_row * 4 * TM + lane_row * 4 + z * 16;
                int thread_col = warp_col * 8 * TN + lane_col * 4 + z * 32;

                FLOAT4(a_frag[z][0]) = FLOAT4(As[k][thread_row]);
                // float4 val = FLOAT4(As[k][thread_row]);
                // a_frag[z][0] = val.x;
                // a_frag[z][1] = val.y;
                // a_frag[z][2] = val.z;
                // a_frag[z][3] = val.w;

                FLOAT4(b_frag[z][0]) = FLOAT4(Bs[k][thread_col]);
                // float4 val1 = FLOAT4(Bs[k][thread_col]);
                // b_frag[z][0] = val1.x;
                // b_frag[z][1] = val1.y;
                // b_frag[z][2] = val1.z;
                // b_frag[z][3] = val1.w;
            }

            // 外积累加: 4 个 4×4 子块
            #pragma unroll
            for(int i = 0 ; i < 4 ; i ++ )
                #pragma unroll
                for(int j = 0 ; j < 4 ; j ++ )
                    #pragma unroll
                    for(int z = 0 ; z < 4 ; z ++ )
                        c_frag[z][i][j] += a_frag[z / 2][i] * b_frag[z % 2][j];
        }

        __syncthreads();
    }
    // 写回 Global Memory
    // 4 个子块分别写回, 每个子块是一行 4 个连续的 float
    #pragma unroll
    for(int z = 0 ; z < 4 ; z ++ ){
        int t_r = warp_row * 4 * TM + lane_row * 4 + (z / 2) * 16;
        int t_c = warp_col * 8 * TN + lane_col * 4 + (z % 2) * 32;

        #pragma unroll
        for(int i = 0 ; i < 4 ; i ++ ){
            int r = block_row + t_r + i;
            if(r >= M)
                continue;

            int c0 = block_col + t_c;
            if(N % 4 == 0 && c0 + 3 < N){
                FLOAT4(C[r * N + c0]) = make_float4(c_frag[z][i][0], c_frag[z][i][1],
                                                    c_frag[z][i][2], c_frag[z][i][3]);
            }else{
                #pragma unroll
                for(int j = 0 ; j < 4 ; j ++ )
                    if(c0 + j < N)
                        C[r * N + c0 + j] = c_frag[z][i][j];
            }
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
    sgemm_warp_tiling_scatter<BM, BN, BK, TM, TN>
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

    const int iterations = 500;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0 ; i < iterations ; i ++ )
        sgemm_warp_tiling_scatter<BM, BN, BK, TM, TN>
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
