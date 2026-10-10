#include <torch/extension.h>
#include <cuda_runtime.h>

// ============================================================
// RMSNorm V1：一个 block 处理一行
//   1) 每线程在寄存器里累加局部 Sigma(x^2)（strided 访问 -> 合并访存）
//   2) warp 内用 __shfl_down_sync 归约
//   3) 跨 warp 走 shared memory，再由第 0 个 warp 收尾
//   4) elementwise 写回
// ============================================================

template <typename T, int BLOCK>
__global__ void rmsnorm_block_kernel(
    const T* __restrict__ x,
    const T* __restrict__ gamma,
    T* __restrict__ y,
    int M, int K, float eps)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const T* __restrict__ row_ptr = x + (long long)row * K;
    T* __restrict__ out_ptr = y + (long long)row * K;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --- 1) 线程局部累加 ---
    float ssq = 0.f;
    for (int i = tid; i < K; i += BLOCK) {
        float v = (float)row_ptr[i];
        ssq += v * v;
    }

    // --- 2) warp 内归约 ---
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ssq += __shfl_down_sync(0xffffffffu, ssq, off);

    // --- 3) 跨 warp 归约 ---
    __shared__ float smem[BLOCK / 32];
    if (lane == 0) smem[warp] = ssq;
    __syncthreads();

    if (warp == 0) {
        constexpr int NWARP = BLOCK / 32;
        float v = (lane < NWARP) ? smem[lane] : 0.f;
#pragma unroll
        for (int off = NWARP / 2; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();

    const float rstd = rsqrtf(smem[0] / (float)K + eps);

    // --- 4) elementwise 写回 ---
    for (int i = tid; i < K; i += BLOCK)
        out_ptr[i] = (T)((float)row_ptr[i] * rstd * (float)gamma[i]);
}

torch::Tensor rmsnorm_v1(torch::Tensor x, torch::Tensor gamma, double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto x2 = x.reshape({-1, x.size(-1)});          // (M, K)
    const int M = (int)x2.size(0);
    const int K = (int)x2.size(1);

    auto y = torch::empty_like(x);

    constexpr int BLOCK = 256;
    dim3 grid(M);
    dim3 block(BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rmsnorm_v1", [&] {
        rmsnorm_block_kernel<scalar_t, BLOCK><<<grid, block>>>(
            x2.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            M, K, (float)eps);
    });
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_v1", &rmsnorm_v1, "RMSNorm V1 block-reduce (CUDA)");
}
