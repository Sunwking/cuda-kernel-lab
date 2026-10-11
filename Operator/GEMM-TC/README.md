# GEMM-TC — Tensor Core GEMM 学习阶梯

> 目标：把 `Operator/GEMM/` 里那套 CUDA Core 的 SIMT GEMM（24.03 TFLOPS @2048^3）
> 换成 Tensor Core 版本，达到 **≥80 TFLOPS**（cuBLAS fp16 的 80%）。
> 参考基线见 `bench/README.md`。

## 进度

| Step | 内容 | 状态 |
| --- | --- | --- |
| 0 | 测 cuBLAS fp16 锚点 | ✅ 完成，见下 |
| 1 | **单 warp 最小 mma，把 fragment layout 钉死** | ✅ 完成（`V0.cu`，`max error = 0`） |
| 2 | `ldmatrix` 从 shared memory 装 fragment（先不做 swizzle） | ✅ 完成（`V1.cu`，`max error = 0`） |
| 3 | 接进 tiling 框架（block 128x128 / warp 64x64 / K 分块 16） | ⬜ 下一步 |
| 4 | swizzle 消 bank conflict + 调 tile 尺寸 | ⬜ |
| 5 | `cp.async` 多级流水线（2 → 4 stage）+ 寄存器双缓冲 | ⬜ |
| 6 | 非对齐 shape 边界 + split-K，对比 cuBLAS fp16 | ⬜ |

## Step 0 结果：这台机器的 Tensor Core 上限

`bench/cublas_sgemm.exe`（fp16 输入 / fp32 累加输出，warmup 20，iters 100，median）：

| shape | fp32 (SIMT) | tf32 | **fp16 (TC)** | fp16 / SIMT 峰值 |
| --- | ---: | ---: | ---: | ---: |
| 2048^3 | 31.27 | 48.01 | 91.98 | 205.6% |
| 4096^3 | 35.74 | 50.12 | 97.33 | 217.5% |
| 8192^3 | 37.32 | 50.52 | **100.38** | 224.3% |

单位 TFLOPS。

- **目标线：2048^3 fp16 ≥ 80 TFLOPS**（= 91.98 的 87%）。当前 SIMT 版是 24.03，即 3.3x 空间。
- **不要做 TF32**：只有 1.35~1.54x，而 fp16 是 2.69~2.94x。
- `2048^3 fp16` 的 mean（50.30）远低于 median（91.98），是被少数慢迭代污染的，
  **结论一律用 median / min**。

## Step 1 结论：`mma.m16n8k16` fragment layout（速查）

`V0.cu` 用一条指令算完 `C(16x8) = A(16x16) * B(16x8)`，`max error = 0 / 128`。
下面是从"内存下标 → 寄存器"的完整映射，Step 2 起全部以它为准。

```cuda
const int lane  = threadIdx.x & 31;
const int group = lane >> 2;   // 0..7  管第一维（A 的行 m / B 的列 n）
const int tid   = lane & 3;    // 0..3  管第二维（A 的列 k / B 的行 k）

// A: m16k16 row-major  → 4 个 uint32（8 个 half）
a[0] = {A[group*16 + tid*2 + 0], A[group*16 + tid*2 + 1]};
a[1] = {A[(group+8)*16 + tid*2 + 0], A[(group+8)*16 + tid*2 + 1]};
a[2] = {A[group*16 + tid*2 + 8], A[group*16 + tid*2 + 9]};
a[3] = {A[(group+8)*16 + tid*2 + 8], A[(group+8)*16 + tid*2 + 9]};

// B: k16n8 row-major，但按 col-major fragment 规则取元素 → 2 个 uint32（4 个 half）
b[0] = {B[(tid*2 + 0)*8 + group], B[(tid*2 + 1)*8 + group]};
b[1] = {B[(tid*2 + 8)*8 + group], B[(tid*2 + 9)*8 + group]};

// C: m16n8 row-major，4 个 f32 累积器
C[group*8 + 2*tid]         = c[0];
C[group*8 + 2*tid + 1]     = c[1];
C[(group+8)*8 + 2*tid]     = c[2];
C[(group+8)*8 + 2*tid + 1] = c[3];
```

三个必须能答上来的问题：

1. `group = lane>>2` 管 A 的行 / B 的列，`tid = lane&3` 管 A 的列（k）/ B 的行（k）。
   A 沿 k 被切成 4 份摊到 4 个 tid 上，靠硬件 crossbar 在指令内部归约。
2. A 是 4 个寄存器、B 是 2 个：寄存器数 = 各维切块数的乘积。
   A 的 `(m,k)` = 16x16 → 每线程 `(2 行块)x(2 列块)` = 4；B 的 `(k,n)` = 16x8 → `(2 行块)x(1 列块)` = 2。
3. `row.col` 说的是 fragment 的**逻辑排布**，不是内存布局。B 在内存里仍是 row-major 的 `K x N`，
   只是取元素时按 col-major 的 fragment 规则索引。这是最高频的误区。

**测试数据**：`A[i][j] = i*16+j`，`B[k][n] = k*8+n`，闭式解 `C[i][n] = 15360*i + 256*i*n + 9920 + 120*n`。
选整数是为了：错了你能一眼看出是转置、错位还是只算了半个 tile。随机数只能告诉你"错了"。
Step 2 起继续复用这份数据和 CPU 参考，把 `V0` 当 oracle 对拍。

## Step 2 结论（`V1.cu`）：`ldmatrix` 装 fragment

`V1.cu` 打通了 global → shared → `ldmatrix` → `mma`，数值与 `V0` 逐位一致。
硬证据（`-Xptxas -v` + `cuobjdump -sass`）：

```
LDSM.16.M88.4  R4, [R0+UR4]   ; x4 非转置 -> A
LDSM.16.MT88.2 R8, [R8+UR6]   ; x2 转置   -> B   (MT = Matrix Transposed)
HMMA.16816.F32 R4, R4, R8, RZ
14 registers, 768 bytes smem = 16*16*2(A) + 16*8*2(B)
```

地址规则（本步唯一要学的东西）：

| lane | 给的地址 | 对应子块 |
| --- | --- | --- |
| 0-7 | `&As[0..7][0]` | A 左上（a[0]） |
| 8-15 | `&As[8..15][0]` | A 左下（a[1]） |
| 16-23 | `&As[0..7][8]` | A 右上（a[2]） |
| 24-31 | `&As[8..15][8]` | A 右下（a[3]） |
| 0-15（`.x2`） | `&Bs[0..15][0]` | B 的上/下半（b[0]/b[1]），`.trans` |

归纳式：`a_row = ((lane>>3)&1)*8 + (lane&7)`，`a_col = (lane>>4)*8`，`b_row = lane & 15`。

**为什么 B 必须 `.trans`**：`b[0] = {B[2*tid][group], B[2*tid+1][group]}` 的两个元素在 B 内存里
隔了整整一行；非 `.trans` 的 `ldmatrix` 只会给"同一行里连续两个"，物理上装不出来。换成非 `.trans`
实测 `mismatched = 128 / 128`。

**寄存器级对拍**（比 `PASS` 强，用来把 `mma` 从嫌疑里摘掉）：lane 0 的期望值
`a = 3c000000 58085800 48804800 58485840`、`b = 48000000 54805400`，
逐位相同即说明 `ldmatrix` 的装填完全正确。

**踩过的坑**

- `__align__(16)` 不可省，否则行首不对齐是 UB，结果是随机错。
- `.x2` 只用 lane 0-15 的地址；lane 16-31 也必须给出**合法**地址（用 `lane & 15` 重复一遍），
  不能写成 `&Bs[lane*8]`（那是越界指针）。
- `ldmatrix` 吃的是 32 位 shared 地址，必须过 `__cvta_generic_to_shared()`。

## Step 3 任务（`V2.cu`）：接进 tiling 框架

前面两版都只有 1 个 warp、1 个 tile，纯玩具。Step 3 开始才是真正的 GEMM：

- block tile **128x128**，warp tile **64x64**（即 block 内 2x2 个 warp），K 方向按 16 分块。
- 每个 warp 算 64x64 的输出，折算成 `m16n8k16` 就是 **4x8 = 32 条 mma**（4 个 M 方向 x 8 个 N 方向）。
- shared memory 放 A tile(128xK) 和 B tile(Kx128)，`ldmatrix` 按 warp 坐标切出自己那份。
- **先不做 swizzle**，接受 bank conflict；先把数值跑对，再做性能。
- 正确性 oracle：同一份随机数据下与 `bench/cublas_sgemm.exe` 的 fp16 结果比对
  （相对误差 ~1e-2 量级即可，fp16 累加顺序不同必然有小差异）。

**验收**：2048^3 fp16 跑通且数值正确，先别管性能；性能调优是 Step 4/5 的事。

## 三个必踩的坑

- **`row.col` 不是内存布局**。B 在内存里仍是 row-major 的 `K x N`，只是按 col-major 的
  fragment 规则取元素。这是最高频的误区。
- **`__halves2half2(lo, hi)` 低半字在前**。装反了会得到"接近但不对"的结果，最难查。
- **不要用 `nvcuda::wmma`**。它掩盖 layout，而 layout 正是这一步唯一要学的东西。
  它留给 Step 2/3 当正确性 oracle。

## 工程注意（这台机器）

- `nvcc -arch=native`：本机是 sm_120，直接编到原生架构，别写死 `-arch=sm_80`。
- **`.cu` 带中文注释时，必须让 MSVC 按 UTF-8 读源码**。中文 Windows 下 MSVC 默认按 GBK 读无 BOM
  的 UTF-8 源码，个别汉字的字节序列会吃掉换行，报出莫名其妙的"行号错位 + 标识符 undefined"。
  两个解法（任选，建议都用）：
  1. 存成 **UTF-8 with BOM**；
  2. 编译时加 `-Xcompiler "/utf-8"`：`nvcc -O3 -arch=native -Xcompiler "/utf-8" V1.cu -o V1.exe`。
  表现：错误行号和真实文件对不上 → 先怀疑编码，不是代码。

## 资料

- PTX ISA：https://docs.nvidia.com/cuda/parallel-thread-execution/ （§9.7.13 / §9.7.14）
- CUDA C++ Programming Guide「Warp Matrix Functions」：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA 博客 Programming Tensor Cores in CUDA C
- LeetCUDA（中文注释的 HGEMM 演进）：https://github.com/xlite-dev/LeetCUDA
- BBuf 优化笔记：https://github.com/BBuf/how-to-optim-algorithm-in-cuda
- CUTLASS 文档「Efficient GEMM in CUDA」：https://docs.nvidia.com/cutlass/
- GPU MODE lectures：https://github.com/cuda-mode/lectures
