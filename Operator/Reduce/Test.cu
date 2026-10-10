/*
============================================
        CUDA Reduce V7 vs V8 Benchmark
============================================

N = 134217728
Data size = 512 MiB
Block size = 256
Warmup = 5
Iterations = 100

============================================
V7: float + Grid-Stride + Warp Shuffle
============================================
Grid      Time(ms)          Bandwidth(GB/s)     Check     
------------------------------------------------------------
128       0.655166          819.442             PASS      
256       0.646802          830.039             PASS      
512       0.649910          826.070             PASS      
1024      0.647755          828.818             PASS      
2048      0.647917          828.611             PASS      
4096      0.647597          829.021             PASS      
8192      0.644417          833.111             PASS      

============================================
V8: float4 + Grid-Stride + Warp Shuffle
============================================
Grid      Time(ms)          Bandwidth(GB/s)     Check     
------------------------------------------------------------
128       0.644172          833.428             PASS      
256       0.641951          836.311             PASS      
512       0.648904          827.350             PASS      
1024      0.639937          838.943             PASS      
2048      0.643920          833.754             PASS      
4096      0.647313          829.383             PASS      
8192      0.644390          833.145             PASS      

============================================
Benchmark Finished
============================================
*/

#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;


// ============================================================
// CUDA Error Check
// ============================================================

#define CUDA_CHECK(call)                                             \
do {                                                                 \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;   \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)


// ============================================================
// Warp Reduce
// ============================================================

__device__ __forceinline__
float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}


// ============================================================
// V7
//
// Grid-Stride Loop
// + Warp Shuffle
// + Shared Memory
// ============================================================

__global__
void reduce_v7(const float* in, float* out, int N)
{
    int tid = threadIdx.x;

    // 当前线程的全局 ID
    int gid = blockDim.x * blockIdx.x + tid;

    // Grid-Stride Loop 步长
    int stride = blockDim.x * gridDim.x;

    // --------------------------------------------------------
    // 每个线程在寄存器中累加
    // --------------------------------------------------------

    float val = 0.0f;

    for (int idx = gid; idx < N; idx += stride)
    {
        val += in[idx];
    }


    // --------------------------------------------------------
    // 第一级：Warp 内 Reduce
    // --------------------------------------------------------

    val = warp_reduce_sum(val);


    // --------------------------------------------------------
    // 每个 Warp 的 lane 0 写入 Shared Memory
    // --------------------------------------------------------

    __shared__ float warp_results[32];

    int lane_id = tid % 32;
    int warp_id = tid / 32;

    if (lane_id == 0)
    {
        warp_results[warp_id] = val;
    }

    __syncthreads();


    // --------------------------------------------------------
    // 第二级：Warp 间 Reduce
    // --------------------------------------------------------

    int num_warps = blockDim.x / 32;

    if (warp_id == 0)
    {
        val = (lane_id < num_warps)
            ? warp_results[lane_id]
            : 0.0f;

        val = warp_reduce_sum(val);
    }


    // --------------------------------------------------------
    // Block 0 Thread 0 写出 Block Sum
    // --------------------------------------------------------

    if (tid == 0)
    {
        out[blockIdx.x] = val;
    }
}


// ============================================================
// V8
//
// V7
// + float4 Vectorized Load
// ============================================================

__global__
void reduce_v8(const float* in, float* out, int N)
{
    int tid = threadIdx.x;

    // 当前线程的全局 ID
    int gid = blockDim.x * blockIdx.x + tid;

    // Grid-Stride Loop 步长
    int stride = blockDim.x * gridDim.x;


    // ========================================================
    // float4 Vectorized Load
    // ========================================================

    float val = 0.0f;

    // N 中可以完整组成多少个 float4
    int N4 = N / 4;

    // 将 float* 转换成 float4*
    const float4* in4 =
        reinterpret_cast<const float4*>(in);


    // --------------------------------------------------------
    // Grid-Stride Loop
    //
    // 每次读取 4 个 float
    // --------------------------------------------------------

    for (int idx = gid; idx < N4; idx += stride)
    {
        float4 data = in4[idx];

        val += data.x;
        val += data.y;
        val += data.z;
        val += data.w;
    }


    // ========================================================
    // Tail
    //
    // 如果 N 不是 4 的倍数，处理最后几个 float
    // ========================================================

    int tail_start = N4 * 4;

    for (int idx = tail_start + gid;
         idx < N;
         idx += stride)
    {
        val += in[idx];
    }


    // ========================================================
    // 第一级：Warp 内 Reduce
    // ========================================================

    val = warp_reduce_sum(val);


    // ========================================================
    // 每个 Warp 的 lane 0 写入 Shared Memory
    // ========================================================

    __shared__ float warp_results[32];

    int lane_id = tid % 32;
    int warp_id = tid / 32;

    if (lane_id == 0)
    {
        warp_results[warp_id] = val;
    }

    __syncthreads();


    // ========================================================
    // 第二级：Warp 间 Reduce
    // ========================================================

    int num_warps = blockDim.x / 32;

    if (warp_id == 0)
    {
        val = (lane_id < num_warps)
            ? warp_results[lane_id]
            : 0.0f;

        val = warp_reduce_sum(val);
    }


    // ========================================================
    // 写出 Block Sum
    // ========================================================

    if (tid == 0)
    {
        out[blockIdx.x] = val;
    }
}


// ============================================================
// CPU Final Reduction
// ============================================================

double cpu_reduce(
    const float* h_out,
    int gridSize)
{
    double result = 0.0;

    for (int i = 0; i < gridSize; i++)
    {
        result += h_out[i];
    }

    return result;
}


// ============================================================
// Benchmark Result
// ============================================================

struct BenchmarkResult
{
    double milliseconds;
    double bandwidth;
    double result;
    bool pass;
};


// ============================================================
// Benchmark V7
// ============================================================

BenchmarkResult benchmark_v7(
    const float* d_in,
    float* d_out,
    float* h_out,
    int N,
    int gridSize,
    int blockSize,
    size_t size)
{
    const int WARMUP = 5;
    const int ITERATIONS = 100;


    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for (int i = 0; i < WARMUP; i++)
    {
        reduce_v7<<<gridSize, blockSize>>>(
            d_in,
            d_out,
            N
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // CUDA Events
    // --------------------------------------------------------

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    // --------------------------------------------------------
    // Benchmark 100 iterations
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        reduce_v7<<<gridSize, blockSize>>>(
            d_in,
            d_out,
            N
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));


    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));


    // 平均 Kernel 时间
    double milliseconds =
        static_cast<double>(total_ms) / ITERATIONS;


    // --------------------------------------------------------
    // Copy partial sums to CPU
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(
        h_out,
        d_out,
        gridSize * sizeof(float),
        cudaMemcpyDeviceToHost
    ));


    // --------------------------------------------------------
    // CPU Final Reduction
    // --------------------------------------------------------

    double result =
        cpu_reduce(h_out, gridSize);

    double expected =
        static_cast<double>(N);

    double error =
        fabs(result - expected);

    bool pass =
        error < 1e-3;


    // --------------------------------------------------------
    // Effective Bandwidth
    //
    // N * sizeof(float)
    // -----------------
    // Kernel Time
    // --------------------------------------------------------

    double bandwidth =
        static_cast<double>(size)
        / (milliseconds * 1e6);


    // --------------------------------------------------------
    // Destroy Events
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    return {
        milliseconds,
        bandwidth,
        result,
        pass
    };
}


// ============================================================
// Benchmark V8
// ============================================================

BenchmarkResult benchmark_v8(
    const float* d_in,
    float* d_out,
    float* h_out,
    int N,
    int gridSize,
    int blockSize,
    size_t size)
{
    const int WARMUP = 5;
    const int ITERATIONS = 100;


    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for (int i = 0; i < WARMUP; i++)
    {
        reduce_v8<<<gridSize, blockSize>>>(
            d_in,
            d_out,
            N
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // CUDA Events
    // --------------------------------------------------------

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        reduce_v8<<<gridSize, blockSize>>>(
            d_in,
            d_out,
            N
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));


    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));


    // 平均 Kernel 时间
    double milliseconds =
        static_cast<double>(total_ms) / ITERATIONS;


    // --------------------------------------------------------
    // Copy partial sums
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(
        h_out,
        d_out,
        gridSize * sizeof(float),
        cudaMemcpyDeviceToHost
    ));


    // --------------------------------------------------------
    // CPU Final Reduction
    // --------------------------------------------------------

    double result =
        cpu_reduce(h_out, gridSize);

    double expected =
        static_cast<double>(N);

    double error =
        fabs(result - expected);

    bool pass =
        error < 1e-3;


    // --------------------------------------------------------
    // Effective Bandwidth
    // --------------------------------------------------------

    double bandwidth =
        static_cast<double>(size)
        / (milliseconds * 1e6);


    // --------------------------------------------------------
    // Destroy Events
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    return {
        milliseconds,
        bandwidth,
        result,
        pass
    };
}


// ============================================================
// Main
// ============================================================

int main()
{
    // ========================================================
    // Problem Size
    // ========================================================

    int N = 1 << 27;

    size_t size =
        static_cast<size_t>(N)
        * sizeof(float);


    cout << "============================================\n";
    cout << "        CUDA Reduce V7 vs V8 Benchmark\n";
    cout << "============================================\n\n";

    cout << "N = " << N << endl;

    cout << "Data size = "
         << size / (1024.0 * 1024.0)
         << " MiB"
         << endl;


    // ========================================================
    // Block Size
    // ========================================================

    int blockSize = 256;


    // ========================================================
    // Grid Size List
    // ========================================================

    const int NUM_GRIDS = 7;

    int gridSizes[NUM_GRIDS] =
    {
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192
    };


    cout << "Block size = "
         << blockSize
         << endl;

    cout << "Warmup = 5" << endl;
    cout << "Iterations = 100" << endl;


    // ========================================================
    // Host Memory
    // ========================================================

    float* h_in =
        new float[N];

    for (int i = 0; i < N; i++)
    {
        h_in[i] = 1.0f;
    }


    // ========================================================
    // Device Memory
    // ========================================================

    float* d_in = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_in,
        size
    ));


    // d_out 按最大的 Grid Size 分配
    int maxGridSize = 8192;

    size_t outSize =
        static_cast<size_t>(maxGridSize)
        * sizeof(float);

    CUDA_CHECK(cudaMalloc(
        &d_out,
        outSize
    ));


    // Host partial sums
    float* h_out =
        new float[maxGridSize];


    // ========================================================
    // Copy Input
    // ========================================================

    CUDA_CHECK(cudaMemcpy(
        d_in,
        h_in,
        size,
        cudaMemcpyHostToDevice
    ));


    // ========================================================
    // Benchmark Header
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "V7: float + Grid-Stride + Warp Shuffle\n";
    cout << "============================================\n";

    cout << left
         << setw(10) << "Grid"
         << setw(18) << "Time(ms)"
         << setw(20) << "Bandwidth(GB/s)"
         << setw(10) << "Check"
         << endl;

    cout << "------------------------------------------------------------\n";


    // ========================================================
    // V7 Benchmark
    // ========================================================

    for (int i = 0; i < NUM_GRIDS; i++)
    {
        int gridSize =
            gridSizes[i];

        BenchmarkResult r =
            benchmark_v7(
                d_in,
                d_out,
                h_out,
                N,
                gridSize,
                blockSize,
                size
            );


        cout << left
             << setw(10) << gridSize
             << setw(18) << fixed << setprecision(6)
             << r.milliseconds
             << setw(20) << setprecision(3)
             << r.bandwidth
             << setw(10)
             << (r.pass ? "PASS" : "FAIL")
             << endl;
    }


    // ========================================================
    // V8 Benchmark
    // ========================================================

    cout << "\n";
    cout << "============================================\n";
    cout << "V8: float4 + Grid-Stride + Warp Shuffle\n";
    cout << "============================================\n";

    cout << left
         << setw(10) << "Grid"
         << setw(18) << "Time(ms)"
         << setw(20) << "Bandwidth(GB/s)"
         << setw(10) << "Check"
         << endl;

    cout << "------------------------------------------------------------\n";


    for (int i = 0; i < NUM_GRIDS; i++)
    {
        int gridSize =
            gridSizes[i];

        BenchmarkResult r =
            benchmark_v8(
                d_in,
                d_out,
                h_out,
                N,
                gridSize,
                blockSize,
                size
            );


        cout << left
             << setw(10) << gridSize
             << setw(18) << fixed << setprecision(6)
             << r.milliseconds
             << setw(20) << setprecision(3)
             << r.bandwidth
             << setw(10)
             << (r.pass ? "PASS" : "FAIL")
             << endl;
    }


    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    delete[] h_in;
    delete[] h_out;


    cout << "\n============================================\n";
    cout << "Benchmark Finished\n";
    cout << "============================================\n";


    return 0;
}