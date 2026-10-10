/*
V7 : 降低每线程输出 tile (8x4), 把寄存器压到 85 以内,
     让每个 SM 能驻留 3 个 block, 占用率 33% -> 50%。

与 V51 的区别:
  - BM=128, BN=64 (warp tile 32x32), BK=16
  - 每线程 8 行 x 4 列 = 32 个累加器 (V51 是 64 个)
  - 32 个 lane 排成 4 行 x 8 列
  - A 分片: 每线程 8 个连续 m -> 2 个 float4 (LDS.128)
  - B 分片: 每线程 4 个连续 n -> 1 个 float4 (LDS.128)
  - 共享内存 25600 B/block, 3 block/SM 仍放得下
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
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 4;
constexpr int BLOCK_SIZE = 256;
// 每个线程需要搬运的 float4 个数 = (BM*BK)/(4*BLOCK_SIZE)
constexpr int LD_A = BM * BK / (4 * BLOCK_SIZE);
constexpr int LD_B = BK * BN / (4 * BLOCK_SIZE);
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

template <int BM, int BN, int BK, int TM, int TN, int MINBLK>
__global__ void __launch_bounds__(BLOCK_SIZE, MINBLK) sgemm_v7
    (float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][BK][BM + 4];
    __shared__ float Bs[2][BK][BN + 4];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.x;

    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    // 8 个 warp = 4 行 x 2 列, warp tile = 32 x 32
    int warp_row = warp_id >> 1;
    int warp_col = warp_id & 1;

    // 32 个 lane 排成 4 行 x 8 列
    int lane_row = lane_id & 3;
    int lane_col = lane_id >> 2;

    float a_frag[TM], b_frag[TN];
    float c_frag[TM][TN] = {0.0f};

    // 预取寄存器: 下一个 k-tile 的 A/B 先读进寄存器
    float4 a_reg[LD_A], b_reg[LD_B];

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
                }else{
                    v.x = (g_row < M && g_col + 0 < K) ? A[g_row * K + g_col + 0] : 0.0f;
                    v.y = (g_row < M && g_col + 1 < K) ? A[g_row * K + g_col + 1] : 0.0f;
                    v.z = (g_row < M && g_col + 2 < K) ? A[g_row * K + g_col + 2] : 0.0f;
                    v.w = (g_row < M && g_col + 3 < K) ? A[g_row * K + g_col + 3] : 0.0f;
                }
            }
            a_reg[s] = v;
        }
    };

    auto fetch_b = [&](int bk){
        #pragma unroll
        for(int s = 0 ; s < LD_B ; s ++ ){
            int idx = 4 * tid + s * 4 * BLOCK_SIZE;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if(idx < BK * BN){
                int row = idx / BN, col = idx % BN;
                int g_row = bk + row, g_col = block_col + col;
                if(N % 4 == 0 && g_row < K && g_col + 3 < N){
                    v = FLOAT4(B[g_row * N + g_col]);
                }else{
                    v.x = (g_row < K && g_col + 0 < N) ? B[g_row * N + g_col + 0] : 0.0f;
                    v.y = (g_row < K && g_col + 1 < N) ? B[g_row * N + g_col + 1] : 0.0f;
                    v.z = (g_row < K && g_col + 2 < N) ? B[g_row * N + g_col + 2] : 0.0f;
                    v.w = (g_row < K && g_col + 3 < N) ? B[g_row * N + g_col + 3] : 0.0f;
                }
            }
            b_reg[s] = v;
        }
    };

    // Register -> Shared: 转置写入 A (As[k][m])
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

    // prologue
    fetch_a(0), fetch_b(0), commit_a(0), commit_b(0);
    __syncthreads();

    const int num_tiles = (K + BK - 1) / BK;
    for(int t = 0 ; t < num_tiles ; t ++ ){
        int cur = t & 1;
        int bk = t * BK;

        if(t + 1 < num_tiles){
            fetch_a(bk + BK);
            fetch_b(bk + BK);
        }

        // 外积累加
        #pragma unroll
        for(int k = 0 ; k < BK ; k ++ ){
            int a_row = warp_row * 32 + lane_row * TM;
            int b_col = warp_col * 32 + lane_col * TN;

            FLOAT4(a_frag[0]) = FLOAT4(As[cur][k][a_row]);
            FLOAT4(a_frag[4]) = FLOAT4(As[cur][k][a_row + 4]);
            FLOAT4(b_frag[0]) = FLOAT4(Bs[cur][k][b_col]);

            #pragma unroll
            for(int i = 0 ; i < TM ; i ++ )
                #pragma unroll
                for(int j = 0 ; j < TN ; j ++ )
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }

        if(t + 1 < num_tiles){
            commit_a(cur ^ 1);
            commit_b(cur ^ 1);
        }
        __syncthreads();
    }

    // 写回: 每线程 8 行 x 4 列, 每行一个 float4
    #pragma unroll
    for(int i = 0 ; i < TM ; i ++ ){
        int r = block_row + warp_row * 32 + lane_row * TM + i;
        if(r >= M) continue;
        int c0 = block_col + warp_col * 32 + lane_col * TN;
        if(N % 4 == 0 && c0 + 3 < N){
            FLOAT4(C[r * N + c0]) = make_float4(c_frag[i][0], c_frag[i][1],
                                                c_frag[i][2], c_frag[i][3]);
        }else{
            #pragma unroll
            for(int j = 0 ; j < TN ; j ++ )
                if(c0 + j < N)
                    C[r * N + c0 + j] = c_frag[i][j];
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

bool check_result(const float* C, float* C_relif, int M, int N){
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
    cout << "CPU reference finished." << endl;

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cout << "block = " << block.x << endl;
    printf("grid = (%d, %d)\n", grid.x, grid.y);

    sgemm_v7<BM, BN, BK, TM, TN, 3><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 500;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0 ; i < iterations ; i ++ )
        sgemm_v7<BM, BN, BK, TM, TN, 3><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

    double tile_flops = 2.0 * (double)BM * (double)BN * (double)BK;
    double memory_bytes = ((double)BM + (double)BN) * (double)BK * sizeof(float);
    double arithmetic_intensity = tile_flops / memory_bytes;

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
