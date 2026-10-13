# GEMM-TC — Tensor Core GEMM 学习阶梯

> 目标：把 `Operator/GEMM/` 里那套 CUDA Core 的 SIMT GEMM（24.03 TFLOPS @2048^3）
> 换成 Tensor Core 版本，达到 **≥80 TFLOPS**（= cuBLAS fp16 91.98 的 87%）。
> 参考基线见 `bench/README.md`。

## 进度

| Step | 内容 | 状态 |
| --- | --- | --- |
| 0 | 测 cuBLAS fp16 锚点 | ✅ 完成，见下 |
| 1 | **单 warp 最小 mma，把 fragment layout 钉死** | ✅ 完成（`V0.cu`，`max error = 0`） |
| 2 | `ldmatrix` 从 shared memory 装 fragment（先不做 swizzle） | ✅ 完成（`V1.cu`，`max error = 0`） |
| 3 | 接进 tiling 框架（block 128x128 / warp 64x32 / K 分块 16） | ✅ 完成（`V2.cu`，46.19 TFLOPS @2048^3） |
| 4 | swizzle 消 bank conflict + 调 tile 尺寸 | ✅ 完成（`V3.cu` / `V4.cu`，59.2 TFLOPS） |
| 5 | `cp.async` 多级流水线（shared memory 层面） | ✅ 完成（`V5.cu`，77.5 TFLOPS） |
| 5b | 寄存器双缓冲（fragment 预取） | ✅ 完成（`V5b.cu`，78.0 TFLOPS） |
| 6 | 任意 shape 边界 + split-K | ✅ 完成（`V6.cu`，瘦长 shape 上 split-K 提速 13.4×） |

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

## Step 3 结论（`V2.cu`）：tiling 框架

grid = `(N/BN, M/BM)`，每个 block 算 C 的 128x128 小块；block = 8 个 warp (256 线程)，
排成 2(行) x 4(列)，每个 warp 算 64x32；K 方向在 block 内部循环，每轮吃 16 个 k。

一个 warp 每轮发 `MF x NF = 4 x 4 = 16` 条 mma（64/16=4 个 M fragment，32/8=4 个 N fragment），
累加器 64 个 float + fragment 24 个 uint32 → **94 registers, 0 spills, 8192 B smem**
（8192 = `128*16*2 + 16*128*2`）。

实测：

```
[check] 256x256x256  max|abs diff| = 1.34e-05   mismatched = 0 / 65536   -> PASS
[check] 512x512x512  max|abs diff| = 3.05e-05   mismatched = 0 / 262144  -> PASS
[bench] 2048x2048x2048  median 0.372 ms  ->  46.19 TFLOPS
LDSM.16.M88.4 x4   LDSM.16.MT88.2 x4   HMMA x16   LDG x8   STS x8   （每个 K 迭代）
```

`LDG 8` = `BM*BK/NTHREADS = 2048/256`，`STS 8` = `BK*BN/NTHREADS`；
`HMMA 16` = `MF*NF`。这三个数能反推"结构写对了没"。

对照：SIMT 版 24.03，cuBLAS fp16 91.98。**只把 tiling 搭对就 1.9x**。

### 这一脚踩过的坑：`ldmatrix` 的地址规则 != fragment 的归属规则

写回 C 时最容易顺手把 `ldmatrix` 那套 lane 规则抄过来（`((lane>>3)&1)*8 + (lane&7)`），
那是错的。两套规则分工不同：

| 规则 | 回答的问题 | 形式 |
| --- | --- | --- |
| 地址规则 | 你这个 lane 负责**搬**哪一行的 16 字节 | 8 行一组：lane 0-7 给子块0… |
| 寄存器归属规则 | 搬完之后**谁手里拿着哪个元素** | `row = lane>>2`、`col = 2*(lane&3)` |

C 的 fragment 归属和 A 的寄存器归属是同一套（V0 验过），与地址规则无关。
写错的后果不是"稍微偏一点"，而是 lanes 0-3 会往同一格写不同值、大量位置没人写、
并且 `grow+8` 会越过 C 的行边界 → 越界写。

自检法子：一个 16x8 fragment 有 32 lane x 4 累加器 = 128 个值，必须**每格恰好写一次**。
代 `lane=0..3` 进去，应落在 (0,0)、(0,2)、(0,4)、(0,6)。

**已知限制（Step 6 再解决）**：M/N/K 必须能被 128/128/16 整除，没有边界判断。

**留给 Step 4/5 的两块**：

- shared memory 没做 swizzle，`ldmatrix` 有 bank conflict。
  粗算：A 的 tile 行距 `BK*2 = 32B` → 0,8,16,24,0,8,16,24… 号 bank → 2 路冲突；
  B 的行距 `BN*2 = 256B` → 每行都从 bank 0 开始 → **8 路冲突**，B 那边更惨。
- 没有 `cp.async`，每轮 K 都要 `__syncthreads()` 停下来等 global load。

## Step 4 任务（`V3.cu` / `V4.cu`）：消 bank conflict

本步算法一行不改，只改 shared memory 的**摆放方式**。两个文件各教一条路：

- `V3.cu`：**padding**。给行距加 8 个 half，让 8 个 lane 的 16B 窗口正好铺满 32 个 bank。
  4 个空：`AST`、`BST`，以及两处 ldmatrix 地址里的行距。
- `V4.cu`：**XOR swizzle**（CUTLASS 走的路）。不浪费存储，把"16B 块号"和"行号低 3 位"异或。
  2 个空：搬入用的 `swz_b(k,col)` 和 ldmatrix 用的 `swz_b_chunk(k,col)`。

### 先把账算清楚

shared = 32 bank x 4B，一次 ldmatrix 里 8 个 lane 各读 16B（= 4 个 bank），
所以目标是**这 8 个窗口互不重叠**。行 r 的 bank = `(r * 行距字节数 / 4) mod 32`。

| tile | 原行距 | 每行错开几个 bank | 结果 |
| --- | --- | --- | --- |
| A (BK=16) | 32B | 8 | 行 0..7 落 0,8,16,24,0,8,16,24 → **2 路冲突** |
| B (BN=128) | 256B | 64 mod 32 = 0 | 8 行全从 bank 0 开始 → **8 路冲突** |
| A (BK=64) | 128B | 32 mod 32 = 0 | 8 行全从 bank 0 开始 → **8 路冲突** |

所以 padding 的目标是让"每行错开 4 个 bank"（16B 窗口依次落在 [0-3],[4-7],…,[28-31]）。

### 为什么 A 用 padding、B 用 swizzle

swizzle 靠"行号混进列位置"来错开，**一行里的 16B chunk 数必须够多**：

- A tile 一行 `BK=16` 个 half = 32B = **2 个** chunk → 只能错开 2 种位置，不够 → 只能 padding。
- B tile 一行 `BN=128` 个 half = 256B = **16 个** chunk → 足够 → 可以 swizzle。

这也是真实 kernel 里 `BK` 通常取 64 及以上的原因之一：tile 够宽，swizzle 才玩得转。

### 实验对照表（先自己预测，再编译验证）

改顶部一个 `BK` 常量就能切换配置。**第 3 行是关键**：先写下你的预测再看。

| # | K 块 | A | B | TFLOPS |
| --- | --- | --- | --- | --- |
| 0 | 16 | 行距 16 | 行距 128 | 46.19（= V2 基线） |
| 1 | 16 | padding | padding | 50.3 |
| 2 | 64 | padding | padding | 58.5 |
| 3 | 64 | **无 padding** | **无 padding** | **39.4** |
| 4 | 16 | padding | swizzle | 50.3|
| 4 | 64 | padding | swizzle | 59.5 |
| 5 | 16 | **无 padding** | padding | 51.24 |

### 实测结论

`V3.cu`（BK=64, padding）**58.52 TFLOPS** / 108 regs / 35840 B smem；
`V4.cu`（BK=64, A padding + B swizzle）**59.24 TFLOPS** / 112 regs / 34816 B smem。
两者差距在噪声内 —— **swizzle 的收益是省下约 1 KB shared，不是更快**；它真正的价值是
"tile 做得更宽（BK>=64）时还能保持零冲突"，以及面试必问。

第 5 行（BK=16，只给 B padding、A 不管）**51.24**，比全 padding 的 50.3 还快一点点：
说明 BK=16 下 A 的 2 路冲突基本不要钱（**2 路 ≈ 免费，8 路才是灾难**），
真正要治的是 B。做优化要能分辨"哪一处冲突值钱"。

**验收**：两个文件都 `PASS`，`[bench]` 落在对照行 ±3% 内，并能解释第 3 行。

**这一步的判断（后面被 Step 5 验证了）**：BK=16 下把冲突全消掉只涨约 8%，
说明 bank conflict **不是**当时的主瓶颈 —— 主瓶颈是缺少流水线。

## Step 5 任务（`V5.cu`）：`cp.async` 多级流水线

前面每一步优化的都是 shared memory 的**布局**，这一步第一次优化**时序**：
把"搬 tile 时 Tensor Core 干等着"的时间消掉。

三个填空，都在流水线的"大脑"位置（不是抄索引，是要想清楚在飞几级）：

1. 序言要预取几级 buffer。
2. 主循环里 `cp_wait<???>()` 该等到剩几组。
3. 预取要写回哪一级 buffer。

参考解实测（2048^3，全部 PASS）：

| BK | stages | smem | TFLOPS |
| --- | --- | --- | --- |
| 16 | 2 | 21.0 KB | 68.5 |
| 16 | **3** | 31.5 KB | **77.5** |
| 16 | 4 | 42.0 KB | 77.0 |
| 16 | 5 | 52.5 KB | 68.0 |
| 16 | 6 | 63.0 KB | 66.9 |
| 32 | 3 | 55.5 KB | 68.4 |
| 32 | 4 | 74.0 KB | 68.3 |
| 64 | 2 | 70.0 KB | 61.0 |
| 64 | 3 | 105 KB | launch 失败（超 shared 上限） |

**级数不是越多越好**：shared 变大 → 每个 SM 能驻留的 block 变少 → 并行度下降，
把流水线省下的时间又赔回去。3 级刚好。

### 为什么 S=5 掉下悬崖：occupancy 账

本机（RTX 5070 Ti / sm_120）的硬约束（用 `devquery.cu` 实测）：

| 资源 | 值 |
| --- | --- |
| regs / SM | 65536 |
| shared / SM | 100 KB |
| shared / block（opt-in 上限） | 99 KB |
| threads / SM | 1536 |
| blocks / SM | 24 |

每 block 256 线程、实测 **104~106 registers**：

- 寄存器限制：`65536 / (104 * 256) = 2.46` → **最多 2 个 block/SM**（与级数无关）
- 线程限制：`1536 / 256 = 6` → 不是瓶颈
- shared 限制（100 KB/SM）：

| stages | smem/block | shared 能放几个 | 实际驻留 | TFLOPS |
| --- | --- | --- | --- | --- |
| 2 | 21.0 KB | 4 | 2（寄存器卡住） | 68.5 |
| 3 | 31.5 KB | 3 | 2 | **77.5** |
| 4 | 42.0 KB | 2 | 2 | 77.0 |
| 5 | 52.5 KB | **1** | **1** ← 掉崖 | 68.0 |
| 6 | 63.0 KB | **1** | 1 | 66.9 |

平台期是 S=3/4（都是 2 block/SM + 流水足够深）；S=5 掉到 1 block/SM，并行度减半，
流水线省下的时间全赔回去。注意 S=2 也是 2 block/SM 却只有 68.5 —— 那是**流水太浅**
（在飞的只有 1 级，藏不住内存延迟），两种瓶颈要分开看。

另外 `BK=64 / stages=3` 需要 105 KB/block，**超过 99 KB 的 opt-in 上限**，
`cudaFuncSetAttribute` 直接返回 `invalid argument` —— 那一格是启动失败，不是变慢。

**里程碑**：`BK=16 / stages=3` 的 77.5 TFLOPS = SIMT 版（24.03）的 **3.2 倍**，
达到 cuBLAS fp16（91.98）的 **84%**。77.5 还**没**摸到 `≥80 TFLOPS` 这条线（差 3%）——真正越过去的是 4096³：Step 6 的 91.6（cuBLAS 的 94%）。

## Step 5b 任务（`V5b.cu`）：寄存器双缓冲

V5 把 global → shared 的搬运藏起来了，但 shared → 寄存器的 `LDSM` 没藏：

```
cp_wait → __syncthreads → LDSM(第 i 级) → HMMA(第 i 级) → issue → commit
```

屏障之后所有 warp 一起冲出来做 LDSM，而 HMMA 必须等 LDSM 的结果。修法是把 LDSM 提前一级：

```
V5 :  LDSM(i)   → HMMA(i)   → LDSM(i+1) → HMMA(i+1)    每个 LDSM 都挡在 HMMA 前面
V5b:  LDSM(i+1) → HMMA(i)   → LDSM(i+2) → HMMA(i+1)    HMMA 一发出去，LDSM 就在后台跑
```

三个填空：序言的 `cp_wait` 参数、主循环 `cp_wait` 的参数、预取的目标 buffer 下标。
**本步真正要"想"的不是这三个数，而是 `issue/commit` 的位置变了**：它被挪到 `cp_wait`
**之前**，commit 组的编号整体前移一位，于是同一个 `cp_wait<NSTAGE-2>` 的含义从
"保证第 i 级就绪"变成"保证第 i+1 级就绪"。照抄 V5 的顺序去预取下一级，读到的是没搬完的 buffer。

参考解实测（2048³，median of 50）：

| 配置 | 寄存器（V5 / MINB=2 / MINB=1） | 驻留 | V5（无） | V5b `MINB=2` | V5b `MINB=1` |
| --- | --- | --- | --- | --- | --- |
| BK=16 / stages=3 | 104 / 128 / 151 | 2 / 2 / 1 | 77.2 | **78.0** | 70.9 |
| BK=16 / stages=4 | 106 / 127 / 152 | 2 / 2 / 1 | 76.7 | 77.7 | 66.8 |
| BK=16 / stages=5 | 106 / 128 / 150 | 1 / 1 / 1 | 68.1 | 68.4 | **70.3** |

参考解实测（4096³，median of 50）：

| 配置 | 寄存器（V5 / MINB=2 / MINB=1） | 驻留 | V5（无） | V5b `MINB=2` | V5b `MINB=1` |
| --- | --- | --- | --- | --- | --- |
| BK=16 / stages=3 | 104 / 128 / 151 | 2 / 2 / 1 | 88.8 | **91.0** | 80.0 |
| BK=16 / stages=4 | 106 / 127 / 152 | 2 / 2 / 1 | 88.3 | 91.0 | 76.9 |
| BK=16 / stages=5 | 106 / 128 / 150 | 1 / 1 / 1 | 78.2 | 78.8 | 81.0 |

（`MINB` = `__launch_bounds__` 的第二个参数，也就是"我承诺每个 SM 至少驻留几个 block"。）

**收益比预期小得多，这是本步最值得记住的一条。**
2 block/SM = 16 个 warp/SM 的时候，LDSM 的延迟**早就**被别的 warp 盖住了，寄存器双缓冲
几乎没有空间：77.2 → 78.0，+0.9%。只有 occupancy 掉到 1 block/SM（stages=5）时，warp 级
并行藏不住延迟，指令级并行才有价值 —— 但**前提是同时放开寄存器**：`MINB=1` 时 ptxas 用
150 个寄存器，68.1 → 70.3（+3.3%）；`MINB=2` 时它被钉在 128 个，只剩 +0.5%。

**`__launch_bounds__` 的第二参数是"承诺"不是"愿望"。** stages=5 时 smem 已经只能放 1 个
block，你还在承诺 2 个，于是白白丢掉 22 个寄存器、什么都没换到。反过来在 stages=3/4 上写
`MINB=1` 就是灾难：151 个寄存器直接把 occupancy 打到 1 block/SM，77.2 → 70.9。

**代价**：fragment 要两套寄存器，`a[MF][4] + b[NF][2] = 24` 个翻倍，104 → 128（钉住之后，
ptxas 是 0 spill）。

### 2048³ 为什么只有 78：尾波

同一份 `V5b`（BK=16 / stages=3）：

| shape | TFLOPS | 对比 cuBLAS fp16 |
| --- | --- | --- |
| 2048³ | 78.0 | 91.98 → 85% |
| 4096³ | 89.9 | 97.33 → 92% |

2048³ 的 grid 只有 256 个 block，而（2 block/SM × 70 SM）= 140 个槽位，
`256 / 140 = 1.83` 波 —— 第二波只装了 83% 就结束了。用 `256 / (2×140) = 91.4%`
修正尾波，`78.0 / 0.914 = 85.3`；拿 4096³ 的 89.9 当渐近值，能对上 95%。
**所以 2048³ 这个数不是 kernel 不行，是 shape 太小喂不饱机器。**

## Step 6 任务（`V6.cu`）：任意 shape 的边界 + split-K

前 5(b) 步都假设 M/N 是 128 的倍数、K 是 16 的倍数。这一步把假设拆掉，并加上 split-K。

**边界**：越界的位置补 0（zfill）。补完之后每个 tile 都是满的，`mma` 那部分一行都不用改。

补零的第二个坑在**搬运**上 —— `cp.async` 要求 global 地址 **16 字节对齐**：
`A[grow][gk]` 的字节地址是 `(grow*K + gk) * 2`，K 不是 8 的倍数时（比如 K=131）行首
只有 2 字节对齐，`cp.async` 直接报 `misaligned address`。所以搬运必须留一条逐元素拷贝的退路。
**split-K**：`grid.z = SPLITK`，把 K 方向切成 SPLITK 段，每段算一份部分和，最后用 `atomicAdd` 累加。

参考解实测（2048³，median of 50）：

| 配置 | 寄存器 | 驻留 | TFLOPS |
| --- | --- | --- | --- |
| 未加边界（= V5b） | 128 | 2 | 78.0 |
| 加边界，**不加** `__launch_bounds__` | 141 | **1** | 62.6 |
| 加边界 + `MINB=2` | 128 | 2 | **78.2** |

边界判断本身几乎不要钱（78.0 → 78.2），但会让 ptxas 多用 13 个寄存器。不管它就直接掉到
1 block/SM、掉 20% —— 又是 Step 5b 那条「寄存器决定 occupancy」的复读。

**验收口径**：`-Xptxas -v` 报 `128 registers`；四条边界 shape（`129x130x131` / `257x385x1000` /
`128x128x2048` / `512x512x2048`）全 PASS；`compute-sanitizer --tool memcheck` 在 `SPLITK=1` 和
`SPLITK=4` 两种编译下都报 `0 errors`。性能（均 splitK=1）：`2048³ = 78.2` / `4096³ = 91.6` /
`256x256x32768 = 4.23`。`4096³` 的 91.6 是 cuBLAS fp16（97.33）的 **94%**，整条阶梯的最好成绩。

### split-K 是一个权衡，不是一个优化

`256 x 256 x 32768` —— `grid.x*grid.y = 4` 个 block，70 个 SM 大部分在睡觉：

| splitK | 1 | 2 | 4 | 8 | 16 | cuBLAS fp16 |
| --- | --- | --- | --- | --- | --- | --- |
| blocks | 4 | 8 | 16 | 32 | 64 | — |
| TFLOPS | 4.23 | 8.45 | 16.3 | 31.8 | **56.7** | 68.0 |

**13.4 倍，而且几乎严格线性** —— 因为瓶颈就是"块数不够"，跟 kernel 写得好不好无关。
64 个 block 铺到 70 个 SM 上，机器才终于被喂饱，此时是 cuBLAS 的 83%。

`2048³` —— `grid.x*grid.y = 256` 个 block，已经 1.83 波，不饿：

| splitK | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| TFLOPS | 78.5 | 76.6 | 69.2 | 54.0 | 37.6 |

**只赔不赚，而且越切越亏。** 两个原因：`atomicAdd` 把 C 变成 fp32 的 read-modify-write，
访存流量放大 SPLITK 倍；K 每段的长度变短，流水线的序言/尾声占比上升。
`4096³` 同理：splitK=1 → 91.6，splitK=4 → 80.6，一样只赔不赚。

**判断标准就一句话**：基础 grid 的 block 数够不够填满 SM。不够才切 K。

### 切 K 必须按 tile 切，不能按元素切

`tbeg = (ktiles * blockIdx.z) / SPLITK` 里的乘除发生在 **tile 号**上，不是元素下标。这不是风格选择，
是正确性要求。对照实验：只把 `tbeg` / `tend` 换成按元素切（`kbeg = K * z / SPLITK`），其余一字不动 ——

| shape | splitK | 按元素切 | 按 tile 切 |
| --- | --- | --- | --- |
| 129 x 130 x 131 | 4 | **FAIL**，mismatched 16734 / 16770 | PASS |
| 257 x 385 x 1000 | 4 | **FAIL**，mismatched 98406 / 98945 | PASS |
| 128 x 128 x 2048 | 4 | PASS | PASS |

为什么：loader 一次搬的是**整块 16 宽的 tile**，`kend` 只管循环跑几次，管不住"最后一块搬多宽"。

- 按 tile 切：`kend = tend * BK` 必是 `BK` 的倍数，末块正好停在 `kend` 上，多读 0 个元素。
- 按元素切：`kend` 一般不是 `BK` 的倍数，末块会**越过 `kend`**，把本该属于下一段的 k 再读一遍，
  相邻两段同时给这段 k 记账 → 双重计数。（K=131、SPLITK=4 → `kbeg = 32`，而 32 不是 16 的倍数。）

`K = 2048` 那两条为什么侥幸过了？因为 `2048 / 4 = 512 = 32 x 16` 恰好是 `BK` 的倍数，两种切法给出同一组
边界。**这是数字的巧合，不是正确性**：测试集里混进这种"恰好对"的 shape 时，缺这个约束是查不出来的。

顺带两个只有泛化之后才暴露的点：

- **stage 轮转索引 `st` 必须和 tile 号解耦**。V5 里 `st` 和"第几个 tile"同余（起点是 0），怎么写都对；
  段起点变成 `kbeg` 之后必须用独立的局部计数器 `st = (st + 1) % NSTAGE`。
- **`cp_commit()` 必须留在 `if` 外面**（序言和主循环尾部各一处）。`cp_wait<N>` 数的是 **commit 出的**
  **group 数**，不是 issue 出的 tile 数，空 issue 也要照常 commit；否则计数错位、`wait` 提前放行，读到
  没搬完的 buffer。反过来这也是好事：段比流水线还短时照样安全 —— K=131、SPLITK=4 的段长只有 2~3 个
  tile，小于 `NSTAGE=3`，实测 PASS。

## ncu 剖析（2048^3 单次 launch）

口径：`grid = (16,16,1)`（M=N=K=2048），取第一个计时 launch。ncu 默认把时钟锁到 base（`--clock-control base`），
所以 duration 比 standalone 高约 14%，**只能同表内横向比**。百分比都是 `pct_of_peak_sustained_elapsed`。

| 版本 | duration (µs) | 张量管线 % | DRAM % | L1TEX % | regs | occupancy % | 主 stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V2 tiling | 529.5 | 37.2 | 8.2 | 40.2 | 94 | 30.5 | long_scoreboard 9.9 |
| V3 padding | 448.5 | 43.7 | 8.4 | 27.2 | 108 | 30.6 | long_scoreboard 12.3 |
| V4 swizzle | 447.3 | 43.9 | 7.7 | 27.3 | 112 | 30.6 | long_scoreboard 11.7 |
| V5 cp.async x3 | 252.3 | **77.9** | 13.1 | 33.6 | 104 | 30.2 | **math_pipe_throttle 15.1** |
| V5b 寄存器双缓冲 | 252.0 | 78.0 | 15.5 | 33.9 | 128 | 29.7 | math_pipe_throttle 12.2 |
| V6 边界 + split-K | 250.8 | **78.9** | 8.6 | 33.7 | 128 | 29.3 | math_pipe_throttle 9.0 |

stall 列的单位是 `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`：每次发射平均有几个 warp 卡在这个原因上。

### 三件 ncu 才说得清的事

1. **瓶颈搬家：从"等内存"变成"喂不动张量管线"。**
   V2~V4 的主 stall 一直是 `long_scoreboard`（全局内存延迟）9.9 → 12.3，说明 SM 大部分时间在等 global load。
   V5 上了 `cp.async` 三级流水之后同一列掉到 **0.38**（内存延迟被预取藏干净），主 stall 换成
   `math_pipe_throttle` **15.1** —— 卡在"张量管线忙不过来"。张量管线利用率 37% → 78%，
   这一步 1.7 倍加速全部来自这个结构性变化。

2. **bank conflict 是看得见的。** V2 → V4 的 L1TEX 吞吐 40.2% → 27.2%，同时 `mio_throttle` 2.11 → 0.37、
   `short_scoreboard` 2.05 → 0.18，而张量管线 37.2% → 44%。这几个数一起动才是"shared memory 冲突被消掉"的证据；
   光看耗时 529 → 448 µs 只能说"快了"，说不出为什么快。

3. **occupancy 从来不是这里的瓶颈。** 六版都在 29~31%（2 block/SM x 8 warp / 48 warp = 33% 的理论上限），
   `launch__occupancy_limit_registers = 2`、`launch__occupancy_limit_shared_mem = 2~3` —— 两个限制同时卡在 2。
   所以 V5b 把寄存器从 104 加到 128 也没掉 occupancy；而 `MINB=1` 那一版是 151 regs / limit=1，
   就是 Step 5b 里掉到 70.9 TFLOPS 的那条路。**这六版从头到尾都是 2 block/SM，性能差异全部来自 stall 结构，不是并行度。**

> 另外：`sm__throughput` 与 `sm__pipe_tensor_cycles_active` 在每一版都**完全相等** ——
> SM 侧占用率最高的单元就是张量管线本身，ALU / LSU / SFU 全都很闲。

### 复现

```powershell
cd Operator\GEMM-TC
ncu --csv --kernel-name regex:gemm_tc --launch-skip 12 --launch-count 1 `
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,`
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,`
sm__warps_active.avg.pct_of_peak_sustained_active,`
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,`
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio .\V6.exe
# --launch-skip：V2~V5b 用 12（2 个 check + 10 次 warm-up），V6 用 14（4 个 check + 10 次 warm-up）。
# 用输出里的 grid dims 核对确实落在 (16, 16, 1) = 2048^3。
```

ncu 权限（`ERR_NVGPUCTRPERM`）的一次性解决办法见 `Operator/LayerNorm/README.md` §6。

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
- **`compute-sanitizer` 是查越界的唯一可靠手段**。`nvcc -O3` 编出来的 kernel 越界读常常
  "什么都不发生"：读到的垃圾只要不写回 C，结果就是对的。Step 6 的边界处理就踩了这个：
  漏了 `grow < M` 门，三个 check 全 PASS、78.9 TFLOPS，但 sanitizer 报 6 个
  `Invalid __global__ read`（其中 4 个是 16 字节的，来自 `cp_async16`）。

  ```
  compute-sanitizer --tool memcheck .\V6.exe 0
  ```
  判据只有一个：`ERROR SUMMARY: 0 errors`。
- **改过 `-D` 开关或源码之后一定要重编再测**。磁盘上的旧 `.exe` 会让 ncu 数据说谎：
  这次做 ncu 剖析时就发现 `V5b.exe` 还是改 `MINB` 之前编的（151 regs / 1 block/SM），
  重编成 128 regs / 2 block/SM 之后才和 README 里的 bench 数字对得上。
  自查办法：`nvcc -Xptxas -v` 报的寄存器数应该和 ncu 的 `launch__registers_per_thread` 一致。

## 资料

- PTX ISA：https://docs.nvidia.com/cuda/parallel-thread-execution/ （§9.7.13 / §9.7.14）
- CUDA C++ Programming Guide「Warp Matrix Functions」：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA 博客 Programming Tensor Cores in CUDA C
- LeetCUDA（中文注释的 HGEMM 演进）：https://github.com/xlite-dev/LeetCUDA
- BBuf 优化笔记：https://github.com/BBuf/how-to-optim-algorithm-in-cuda
- CUTLASS 文档「Efficient GEMM in CUDA」：https://docs.nvidia.com/cutlass/
- GPU MODE lectures：https://github.com/cuda-mode/lectures
