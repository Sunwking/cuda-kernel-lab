// Operator/GEMM-TC/V0.cu
// Step 1: 单 warp 最小 Tensor Core GEMM —— 只练 fragment layout，不含 tiling / shared / 流水线
//
// 目标: 用一条 mma.sync.m16n8k16 算完 C(16x8) = A(16x16) * B(16x8)，与 CPU 参考完全一致。
// 编译: nvcc -O3 -arch=native V0.cu -o V0.exe
//
// 本文件是 Step 2 的正确性 oracle: 之后所有版本都用同一组测试数据和同一份 CPU 参考对拍。
// fragment layout 的权威来源: PTX ISA 9.7.13 "Warp-Level Matrix Instructions" 里
// mma.m16n8k16 的图。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

// 把两个 half 打进一个 32bit 寄存器。__halves2half2(lo, hi) 里 lo 落在低 16 位。
__device__ __forceinline__ uint32_t pack2(__half lo, __half hi){
    __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// A: 16x16 row-major (M x K)   B: 16x8 row-major (K x N)   C: 16x8 row-major
__global__ void mma_probe(const __half* __restrict__ A,
                          const __half* __restrict__ B,
                          float* __restrict__ C){
    const int lane  = threadIdx.x & 31;
    const int group = lane >> 2;   // 0..7  —— 控制"行"（对 B 来说是列 n）
    const int tid   = lane & 3;    // 0..3  —— 控制"列"（对 B 来说是行 k）

    // A fragment：每个线程 8 个 half = 4 个 uint32。
    //   a[0]/a[1] 差 8 行，a[2]/a[3] 差 8 行；a[0]/a[2] 差 8 列（k 方向）。
    //   即：寄存器对 = {行块, 列块}，(行块, 列块) ∈ {0,1} x {0,1}，
    //   行块靠 +8 行切换、列块靠 +8 列切换。
    uint32_t a[4];
    a[0] = pack2(A[group       * 16 + tid*2 + 0], A[group       * 16 + tid*2 + 1]);
    a[1] = pack2(A[(group + 8) * 16 + tid*2 + 0], A[(group + 8) * 16 + tid*2 + 1]);
    a[2] = pack2(A[group       * 16 + tid*2 + 8], A[group       * 16 + tid*2 + 8 + 1]);
    a[3] = pack2(A[(group + 8) * 16 + tid*2 + 8], A[(group + 8) * 16 + tid*2 + 8 + 1]);

    // B fragment：每个线程 4 个 half = 2 个 uint32。元素 B[k][n] 的下标 = k*8 + n。
    //   注意 row.col 说的是 fragment 的逻辑排布，不是内存存储顺序——
    //   B 在内存里仍然是 row-major 的 16x8，这里按 col-major 的 fragment 规则取元素。
    //   group 管 n（列），tid 管 k（行），每线程沿 k 覆盖 {2*tid, 2*tid+1, 2*tid+8, 2*tid+9}。
    uint32_t b[2];
    b[0] = pack2(B[(tid*2 + 0) * 8 + group], B[(tid*2 + 1) * 8 + group]);
    b[1] = pack2(B[(tid*2 + 8) * 8 + group], B[(tid*2 + 9) * 8 + group]);

    // D = A*B + C，操作数顺序是 {d}, {a}, {b}, {c}。
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

    // 写回 C：C 是 16x8 row-major，下标 = row*8 + col。
    //   四个累积器和四个 (row, col) 一一对应，且和 A 的 fragment 规则同一套（group 管行、tid 管列）。
    C[group       * 8 + 2*tid]     = c[0];
    C[group       * 8 + 2*tid + 1] = c[1];
    C[(group + 8) * 8 + 2*tid]     = c[2];
    C[(group + 8) * 8 + 2*tid + 1] = c[3];
}

// ================== CPU 参考 / 校验 / 计时 ==================
int main(){
    const int M=16, N=8, K=16;
    std::vector<__half> hA(M*K), hB(K*N);
    std::vector<float>  hC(M*N, 0.f), ref(M*N, 0.f);

    // 测试数据（刻意选成能手算的整数）：
    //   A[i][j] = i*16 + j    (0..255, fp16 可精确表示)
    //   B[k][n] = k*8  + n    (0..127)
    // 期望值闭式解： C[i][n] = 15360*i + 256*i*n + 9920 + 120*n
    for(int i=0;i<M;i++) for(int j=0;j<K;j++) hA[i*K+j] = __float2half((float)(i*16+j));
    for(int k=0;k<K;k++) for(int n=0;n<N;n++) hB[k*N+n] = __float2half((float)(k*8+n));

    for(int i=0;i<M;i++) for(int n=0;n<N;n++){
        double s=0;
        for(int k=0;k<K;k++) s += (double)(i*16+k) * (double)(k*8+n);
        ref[i*N+n] = (float)s;
    }

    __half *dA,*dB; float *dC;
    CK(cudaMalloc(&dA, M*K*sizeof(__half)));
    CK(cudaMalloc(&dB, K*N*sizeof(__half)));
    CK(cudaMalloc(&dC, M*N*sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), M*K*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), K*N*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0, M*N*sizeof(float)));

    mma_probe<<<1,32>>>(dA,dB,dC);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC.data(), dC, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    double maxErr=0; int bad=0;
    for(int i=0;i<M*N;i++){
        double d = fabs((double)hC[i]-(double)ref[i]);
        if(d>maxErr) maxErr=d;
        if(d>1e-3) bad++;
    }
    printf("C[ 0][0] = %10.1f   expected %10.1f\n", hC[0],        ref[0]);
    printf("C[ 7][3] = %10.1f   expected %10.1f\n", hC[7*N+3],    ref[7*N+3]);
    printf("C[15][7] = %10.1f   expected %10.1f\n", hC[15*N+7],   ref[15*N+7]);
    printf("max error = %.6g   mismatched = %d / %d\n", maxErr, bad, M*N);
    printf("Check: %s\n", bad==0 ? "PASS" : "FAIL");
    return bad==0?0:1;
}