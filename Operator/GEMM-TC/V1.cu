// Operator/GEMM-TC/V1.cu
// Step 2: 用 ldmatrix 从 shared memory 装 fragment —— 打通 global -> shared -> ldmatrix -> mma
// 结果: 数值与 V0 完全一致 (max error = 0 / 128)，且 SASS 里确实出现了
//         LDSM.16.M88.4  (x4, 非转置) -> A
//         LDSM.16.MT88.2 (x2, 转置)   -> B     (MT = Matrix Transposed)
//         HMMA.16816.F32
//       -Xptxas -v 报 14 registers / 768 bytes smem = 16*16*2(A) + 16*8*2(B)。
//
// 编译: nvcc -O3 -arch=native V1.cu -o V1.exe
// 存盘: UTF-8 with BOM，或编译时加 -Xcompiler "/utf-8"。
//       中文 Windows 下 MSVC 默认按 GBK 读无 BOM 的 UTF-8 源码，会吃掉个别换行，
//       报出"行号错位 + 标识符 undefined"的假错。
//
// ldmatrix 速查 (PTX ISA 9.7.14.4):
//   ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
//   * 一条指令搬 4 个 8x8 的 b16 子块 = 4*8*8*2 = 512B / warp = 16B / lane = 4 个 uint32
//   * 每个 lane 提供的是"一行 8 个 half 的起点"(32-bit shared 地址)，必须 16B 对齐
//   * .x4: lane 0-7 给子块0 的 8 行, lane 8-15 给子块1, lane 16-23 给子块2, lane 24-31 给子块3
//   * .x2: 只用 lane 0-15 的地址（子块0 从 lane 0-7, 子块1 从 lane 8-15）
//   * 非 .trans 的产出: 每个 8x8 块内，lane 拿 row=lane>>2、col=2*(lane&3) 及其 +1 两个元素
//   * .trans 的产出: 先把 8x8 转置，再按上面的规则分发 —— 这就是 B 的 col-major fragment 的来源
//
// 核心结论（一句话）: ldmatrix 只负责"怎么搬"，搬运的分发规则是硬件固定的；
// 你要做的只是告诉它"每个 lane 该从哪一行的行首开始搬"。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

// 通用地址 -> 32 位 shared 地址（ldmatrix 要的是后者，不是通用指针）
__device__ __forceinline__ uint32_t smem_addr(const void* p){
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// A: 16x16 row-major (M x K)   B: 16x8 row-major (K x N)   C: 16x8 row-major
__global__ void mma_ldmatrix(const __half* __restrict__ A,
                             const __half* __restrict__ B,
                             float* __restrict__ C){
    const int lane  = threadIdx.x & 31;
    const int group = lane >> 2;   // 0..7
    const int tid   = lane & 3;    // 0..3

    // __align__(16) 是必须的: ldmatrix 每行的首地址要 16 字节对齐。
    // A 行距 16 half = 32B, B 行距 8 half = 16B, 所以基址对齐则行首都对齐。
    __align__(16) __shared__ __half As[16*16];
    __align__(16) __shared__ __half Bs[16*8];

    for(int i = threadIdx.x; i < 16*16; i += 32) As[i] = A[i];
    for(int i = threadIdx.x; i < 16*8;  i += 32) Bs[i] = B[i];
    __syncthreads();

    // ---- A 的地址: x4 的四个 8x8 子块 = A(16x16) 的四个象限 ----
    //   子块0 = A[0..7 ][0..7 ] -> a[0]      子块1 = A[8..15][0..7 ] -> a[1]
    //   子块2 = A[0..7 ][8..15] -> a[2]      子块3 = A[8..15][8..15] -> a[3]
    // 每个 lane 给出自己那一行的行首（lane>>3 的低位选"上/下"，高位选"左/右"）：
    //   lane  0..7  -> A 第 0..7  行, 第 0 列       lane  8..15 -> A 第 8..15 行, 第 0 列
    //   lane 16..23 -> A 第 0..7  行, 第 8 列       lane 24..31 -> A 第 8..15 行, 第 8 列
    const int a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int a_col = (lane >> 4) * 8;
    uint32_t addrA = smem_addr(&As[a_row * 16 + a_col]);

    uint32_t a[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addrA));

    // ---- B 的地址 + 变体 ----
    // mma 要的 B fragment 是 col-major 的: b[0] = {B[2*tid][group], B[2*tid+1][group]}
    // （前一个下标是 k, 后一个是 n）。这两个元素在 B 内存里隔了整整一行，
    // 所以非 .trans 的"同一行连续两个"物理上装不出来，必须用 .trans。
    //   .x2 的两个 8x8 子块 = B 的行 0..7 和行 8..15，故 lane 0..15 给 &Bs[0..15][0]。
    //   lane 16..31 的地址被忽略，但要给合法值（用 lane & 15 重复一遍，避免越界）。
    const int b_row = lane & 15;
    uint32_t addrB = smem_addr(&Bs[b_row * 8]);

    // 反面教材: 换成下面这句会得到 {B[group][2*tid], B[group][2*tid+1]}，
    //           即转置后的 fragment，实测 mismatched = 128 / 128。
    //   "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
    uint32_t b[2];
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1])
                 : "r"(addrB));

    // ---- 对拍用: 打印 lane 0 的原始寄存器位模式，和 V0 逐位比对 ----
    // lane 0 的期望值（A[i][j]=i*16+j, B[k][n]=k*8+n）:
    //   a = 3c000000 58085800 48804800 58485840   b = 48000000 54805400
    if (threadIdx.x == 0) {
        printf("a: %08x %08x %08x %08x | b: %08x %08x\n",
               a[0], a[1], a[2], a[3], b[0], b[1]);
    }

    // ---- mma + 写回，与 V0 逐字相同 ----
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

    C[group       * 8 + 2*tid]     = c[0];
    C[group       * 8 + 2*tid + 1] = c[1];
    C[(group + 8) * 8 + 2*tid]     = c[2];
    C[(group + 8) * 8 + 2*tid + 1] = c[3];
}

// ================== CPU 参考 / 校验（与 V0 同一套） ==================
int main(){
    const int M=16, N=8, K=16;
    std::vector<__half> hA(M*K), hB(K*N);
    std::vector<float>  hC(M*N, 0.f), ref(M*N, 0.f);

    // A[i][j] = i*16+j, B[k][n] = k*8+n；闭式解 C[i][n] = 15360*i + 256*i*n + 9920 + 120*n
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

    mma_ldmatrix<<<1,32>>>(dA,dB,dC);
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