/*
M = 4096
N = 4096

Computing CPU reference...
CPU reference finished.
block = 256
grid = (4096)

============================================
                Result
============================================
Average Kernel Time = 0.181123 ms
Performance = 463.143 GFLOPS
AR = 0.625 FLOPs/Byte
Check = PASS
Row 0: first = 2.30626e-07, last = 0.000687487
Expected first = 2.30626e-07, Expected last = 0.000687486

Benchmark finished.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int M, N;
constexpr int BLOCK_SIZE = 256;

// Warp 内合并规约
__device__ void warpReduceOnline(float &m, float &d){
    for(int offset = 16 ; offset ; offset >>= 1){
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);

        float m_merge = fmaxf(m2, m);
        d = d * expf(m - m_merge) + d2 * expf(m2 - m_merge);
        m = m_merge;
    }
}

// V1: Block 级并行 + Warp Shuffle 两级合并规约
__global__ void online_softmax_v1(float* in, float* out, int M, int N) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int Warp_id = tid >> 5;
    int lane_id = tid & 31;

    float* x = in + row * N;
    float* y = out + row * N;

    // 第 1 遍: 每线程 Online 处理一段
    float local_m = -INFINITY;
    float local_d = 0.0f;
    for(int i = tid ; i < N ; i += blockDim.x){
        float xi = x[i];
        float m_new = fmaxf(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    // 第一级: Warp 内合并
    // local_m, local_d 存的是每个 Warp 内的结果
    warpReduceOnline(local_m, local_d);

    // Warp 通过 Shared Memory 进行合并
    __shared__ float warp_m[32];
    __shared__ float warp_d[32];

    if(lane_id == 0){
        warp_m[Warp_id] = local_m;
        warp_d[Warp_id] = local_d;
    }
    __syncthreads();

    // 第二级: Warp 0 做最终合并
    int num_warp = blockDim.x >> 5;
    // 填充项是 (m, d) 的中性元：d 必须为 0，m 用一个有限的极小值。
    // m 若用 -INFINITY，两个填充项相遇时会算出 (-inf) - (-inf) = NaN，
    // NaN 再顺着 shuffle 树一路污染到 lane 0。
    local_m = (lane_id < num_warp) ? warp_m[lane_id] : -1e30f;
    local_d = (lane_id < num_warp) ? warp_d[lane_id] : 0.0f;
    if(Warp_id == 0) warpReduceOnline(local_m, local_d);
    
    // 广播所有结果
    __shared__ float final_m;
    __shared__ float final_d;
    if(tid == 0){
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m, inv_sum = 1.0f / final_d;

    // 第 2 遍: 归一化输出
    for(int i = tid ; i < N ; i += blockDim.x)
        y[i] = expf(x[i] - row_max) * inv_sum;

}

void softmax_cpu(const float* in, float* out, int M, int N) {
    for (int row = 0; row < M; ++row) {
        const float* x = in + row * N;
        float max_val = -INFINITY;
        for (int i = 0; i < N; ++i)
            max_val = fmaxf(max_val, x[i]);

        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
            sum += expf(x[i] - max_val);

        float inv_sum = 1.0f / sum;
        for (int i = 0; i < N; ++i)
            out[row * N + i] = expf(x[i] - max_val) * inv_sum;
    }
}

bool check_result(const float* C, const float* C_ref, int M, int N) {
    const float tolerance = 1e-5f;
    for (int i = 0; i < M * N; ++i) {
        // NaN 参与任何比较都是 false，只判 diff > tolerance 会把满屏 NaN 判成 PASS
        if (isnan(C[i]) || isnan(C_ref[i])) {
            cerr << "NaN at index " << i << ", GPU = " << C[i]
                 << ", CPU = " << C_ref[i] << endl;
            return false;
        }
        float diff = fabsf(C[i] - C_ref[i]);
        if (diff > tolerance) {
            cerr << "Mismatch at index " << i << ", GPU = " << C[i]
                 << ", CPU = " << C_ref[i] << ", diff = " << diff << endl;
            return false;
        }
    }
    return true;
}

int main() {
    M = N = 4096;

    cout << "M = " << M << endl;
    cout << "N = " << N << endl;
    cout << endl;

    size_t size_in = (size_t)M * N * sizeof(float);
    size_t size_out = (size_t)M * N * sizeof(float);

    float* h_in = (float*)malloc(size_in);
    float* h_out = (float*)malloc(size_out);
    float* h_out_ref = (float*)malloc(size_out);

    if (!h_in || !h_out || !h_out_ref) {
        cerr << "Host memory allocation failed!" << endl;
        return EXIT_FAILURE;
    }

    // 初始化：每一行的输入值都不相同，确保 softmax 的结果有意义
    for (int i = 0; i < M * N; ++i) {
        h_in[i] = static_cast<float>((i % 17) - 8) * 0.5f + static_cast<float>((i / 17) % 7) * 0.25f;
    }

    cout << "Computing CPU reference..." << endl;
    softmax_cpu(h_in, h_out_ref, M, N);
    cout << "CPU reference finished." << endl;

    float* d_in;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size_in));
    CUDA_CHECK(cudaMalloc(&d_out, size_out));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid(M);

    cout << "block = " << block.x << endl;
    printf("grid = (%d)\n", grid.x);

    online_softmax_v1<<<grid, block>>>(d_in, d_out, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 200;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        online_softmax_v1<<<grid, block>>>(d_in, d_out, M, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

    double flops = 5.0 * (double)M * (double)N;
    double seconds = avg_ms / 1000.0;
    double performance = flops / seconds / 1e9;  // GFLOPS
    double memory_bytes = 2.0 * (double)M * (double)N * sizeof(float);
    double ar = flops / memory_bytes; // FLOPs/Byte

    CUDA_CHECK(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));

    bool pass = check_result(h_out, h_out_ref, M, N);

    cout << endl;
    cout << "============================================" << endl;
    cout << "                Result" << endl;
    cout << "============================================" << endl;
    cout << "Average Kernel Time = " << avg_ms << " ms" << endl;
    cout << "Performance = " << performance << " GFLOPS" << endl;
    cout << "AR = " << ar << " FLOPs/Byte" << endl;
    cout << "Check = " << (pass ? "PASS" : "FAIL") << endl;
    cout << "Row 0: first = " << h_out[0] << ", last = " << h_out[N - 1] << endl;
    cout << "Expected first = " << h_out_ref[0] << ", Expected last = " << h_out_ref[N - 1] << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    free(h_in);
    free(h_out);
    free(h_out_ref);

    cout << endl;
    cout << "Benchmark finished." << endl;
    return 0;
}
