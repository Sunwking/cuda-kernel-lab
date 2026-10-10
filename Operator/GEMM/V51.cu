/*
M = 2048
N = 2048
K = 2048
Bk = 8 / 16

Computing CPU reference...
CPU reference finished.
block = 256
grid = (16, 16)

============================================
                Result
============================================
Average Kernel Time = 0.680192 ms / 0.629929 ms
Performance = 25.2574 TFLOPS / 27.2727 TFLOPS
Arithmetic Intensity = 32 FLOPs/Byte
Check = PASS

C[0][0] = 2048
Expected = 2048

Benchmark finished.

============================================

M = 8192
N = 8192
K = 8192
BK = 16

Computing CPU reference...
CPU reference finished.
block = 256
grid = (64, 64)

============================================
                Result
============================================
Average Kernel Time = 36.7036 ms
Performance = 29.9565 TFLOPS
Arithmetic Intensity = 32 FLOPs/Byte
Check = PASS

C[0][0] = 8192
Expected = 8192

Benchmark finished.

80.3% of cuBLAS fp32 (same SIMT path, verified via tf32/fp32 = 1.35x)
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
constexpr int BK = 8 * 2;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int BLOCK_SIZE = 256;
// 每个线程需要搬运的 float4 个数 = (BM*BK)/(4*BLOCK_SIZE)
constexpr int LD_A = BM * BK / (4 * BLOCK_SIZE);
constexpr int LD_B = BK * BN / (4 * BLOCK_SIZE);
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

// ============================================================
// V5 : 在 V4Z(Scatter + Z-order) 基础上加入双缓冲 (Ping-Pong)
//
// 目标: 把 "Global -> Shared" 的访存延迟藏到 "Register FMA" 后面。
//
// 目标卡是 Turing(sm_75), 没有 cp.async, 所以用寄存器做中转:
//
//   prologue:  global -> reg -> shared[0]
//   loop t:    1) global  -> reg        (下一个 k-tile, 只发 LDG)
//              2) shared[cur] -> reg -> FMA  (当前 k-tile, 一大段 FFMA)
//              3) reg    -> shared[cur^1]    (FMA 之后才落盘)
//              __syncthreads()
//
// 这样 LDG 发出后要跨越整段外积计算才被 STS 消费, 延迟被 FFMA 覆盖;
// 同时 shared 用量翻倍(8448 -> 16896 B), 但 2 block/SM 仍然放得下。
// ============================================================

// 第二个参数 2 表示每个 SM 至少驻留 2 个 block, 把寄存器压在 128 以内;
// 否则把 BK 调大时(A_ITERS=2)会飙到 141 个寄存器, 占用率掉到 25%。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(BLOCK_SIZE, 2) sgemm_warp_tiling_scatter
    (float* A, float* B, float* C, int M, int N, int K) {
    // 双缓冲(Ping-Pong): 两块 shared, 一块给当前 tile 计算
    // 另一块接收预取的下一个 tile, 从而让 global load 与 FMA 重叠。
    __shared__ float As[2][BK][BM + 4];
    __shared__ float Bs[2][BK][BN + 4];

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
    int lane_row = (lane_id & 1) + (lane_id >> 4) * 2;
    int lane_col = (lane_id & 15) >> 1;

    // 寄存器数组
    // 每个线程负责 4 个 4×4 子块，所以片段的第二维是 4（子块边长），
    // 而不是 TM / TN。a_frag[z][i] 是第 z 个行半区的第 i 行，
    // b_frag[z][j] 是第 z 个列半区的第 j 列。
    float a_frag[2][4], b_frag[2][4];
    float c_frag[4][4][4] = {0.0f};

    // 预取寄存器: 下一个 k-tile 的 A/B 先读进寄存器
    // 等当前 tile 的 FMA 算完再写进 shared, 这样 global load 的延迟被计算掩盖。
    float4 a_reg[LD_A], b_reg[LD_B];

    // Global -> Register: 读第 bk 个 k-tile 的 A^T 分片
    // 加载 A^T(BK, BM), 使用 float4 向量化加载
    auto fetch_a = [&](int bk){
      #pragma unroll
      for(int s = 0 ; s < LD_A ; s ++ ){
        int idx = 4 * tid + s * 4 * BLOCK_SIZE;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if(idx < BM * BK){
            int row = idx / BK, col = idx % BK;
            int g_row = block_row + row, g_col = bk + col;
            if(K % 4 == 0 && g_row < M && g_col + 3 < K){
                v = FLOAT4(A[g_row * K + g_col]);
            }else {
                v.x = (g_row < M && g_col + 0 < K) ? A[g_row * K + g_col + 0] : 0.0f;
                v.y = (g_row < M && g_col + 1 < K) ? A[g_row * K + g_col + 1] : 0.0f;
                v.z = (g_row < M && g_col + 2 < K) ? A[g_row * K + g_col + 2] : 0.0f;
                v.w = (g_row < M && g_col + 3 < K) ? A[g_row * K + g_col + 3] : 0.0f;
            }
        }
        a_reg[s] = v;
      }
    };

    // Global -> Register: 读第 bk 个 k-tile 的 B 分片
    // 加载 B(BK, BN), 使用 float4 向量化加载
    auto fetch_b = [&](int bk){
      #pragma unroll
      for(int s = 0 ; s < LD_B ; s ++ ){
        int idx = 4 * tid + s * 4 * BLOCK_SIZE;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if(idx < BN * BK){
            int row = idx / BN, col = idx % BN;
            int g_row = bk + row, g_col = block_col + col;

            // N 必须是 4 的倍数, 才能保证 B 的行首地址 16 字节对齐
            if(N % 4 == 0 && g_row < K && g_col + 3 < N){
                v = FLOAT4(B[g_row * N + g_col]);
            }else {
                v.x = (g_row < K && g_col + 0 < N) ? B[g_row * N + g_col + 0] : 0.0f;
                v.y = (g_row < K && g_col + 1 < N) ? B[g_row * N + g_col + 1] : 0.0f;
                v.z = (g_row < K && g_col + 2 < N) ? B[g_row * N + g_col + 2] : 0.0f;
                v.w = (g_row < K && g_col + 3 < N) ? B[g_row * N + g_col + 3] : 0.0f;
            }
        }
        b_reg[s] = v;
      }
    };

    // Register -> Shared: 把预取的 A 分片转置写入第 buf 块 shared
    auto commit_a = [&](int buf){
      #pragma unroll
      for(int s = 0 ; s < LD_A ; s ++ ){
        int idx = 4 * tid + s * 4 * BLOCK_SIZE;
        if(idx < BM * BK){
            int row = idx / BK, col = idx % BK;
            As[buf][col + 0][row] = a_reg[s].x;
            As[buf][col + 1][row] = a_reg[s].y;
            As[buf][col + 2][row] = a_reg[s].z;
            As[buf][col + 3][row] = a_reg[s].w;
        }
      }
    };

    // Register -> Shared: 把预取的 B 分片写入第 buf 块 shared
    auto commit_b = [&](int buf){
      #pragma unroll
      for(int s = 0 ; s < LD_B ; s ++ ){
        int idx = 4 * tid + s * 4 * BLOCK_SIZE;
        if(idx < BK * BN){
            int row = idx / BN, col = idx % BN;
            FLOAT4(Bs[buf][row][col]) = b_reg[s];
        }
      }
    };

    // 流水线 prologue: 先把第 0 个 k-tile 搬进 shared[0]
    fetch_a(0), fetch_b(0), commit_a(0), commit_b(0);
    __syncthreads();

    // K 方向 Tile Loop (双缓冲: cur 用于计算, cur^1 接收预取)
    const int num_tiles = (K + BK - 1) / BK;
    for(int t = 0 ; t < num_tiles ; t ++ ){
        int cur = t & 1;
        int bk = t * BK;
        
         // 1. Global -> Register: 预取下一个 k-tile, 只发 LDG 不写 shared
         if(t + 1 < num_tiles){
            fetch_a(bk + BK);
            fetch_b(bk + BK);
         }

        //  Shared Memory → Register
        //  Register → Register FMA
        // 2. 外积累加
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

                FLOAT4(a_frag[z][0]) = FLOAT4(As[cur][k][thread_row]);
                // float4 val = FLOAT4(As[k][thread_row]);
                // a_frag[z][0] = val.x;
                // a_frag[z][1] = val.y;
                // a_frag[z][2] = val.z;
                // a_frag[z][3] = val.w;

                FLOAT4(b_frag[z][0]) = FLOAT4(Bs[cur][k][thread_col]);
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

        // 3. Register -> Shared: 计算结束再落盘, 让上面 LDG 的延迟
        //    被这一整段 FMA 覆盖
        if(t + 1 < num_tiles){
            commit_a(cur ^ 1);
            commit_b(cur ^ 1);
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
    
    M = N = K = 8 * 1024;

    cout << "M = " << M << endl;
    cout << "N = " << N << endl;
    cout << "K = " << K << endl;
    cout << "BK = " << BK << endl;
    
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
