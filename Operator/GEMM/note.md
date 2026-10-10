# CUDA GEMM 优化路线性能分析（base → V6）

> 本文件基于本仓库 `Operator/GEMM` 下的 12 个实现（base、V1、V2、V2F、V3、V3F、V4、V4P、V4Z、V5、V51、V6），
> 用 Nsight Compute（ncu）实测剖析后整理，不含外部资料。

## 0. 测试环境与口径

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti（sm_120 / Blackwell，70 SM，15.9 GB） |
| 硬件上限 | regs/SM = 65536，threads/SM = 1536，shared/SM = 100 KB，FP32 ≈ 44 TFLOPS |
| CUDA | 13.4（V13.4.92），host 编译器 MSVC 14.51（VS Build Tools 18） |
| 编译 | `nvcc -arch=native`（**关键**，见 §4.6） |
| 问题规模 | M = N = K = 2048，fp32，row-major |
| 计时 | 每版本 warm-up 后跑 500 次 kernel，cudaEvent 取平均 |
| 剖析 | ncu 2026.3.1，`--launch-skip 20 --launch-count 1`，sections：SpeedOfLight / Occupancy / MemoryWorkloadAnalysis / SchedulerStats / WarpStateStats |

> 说明：仓库里原有的 12 个 `.exe` 是各版本随手编出来的（迭代次数 1/100/500 不统一、架构默认）。
> 为保证可比性，本表数据是**统一重编重跑**：全部 `-arch=native`、统一 500 次迭代。
> 其中 `base.cu` 已按它自己文件头记录的 2048³ 基线重建（朴素实现 + 同样的 benchmark 框架）。

## 1. 结论速览

- **base 不是 DRAM 带宽瓶颈，而是 L1/TEX（LSU）吞吐瓶颈**：L1/TEX 吞吐 96%，DRAM 只有 4%，调度器 70% 周期"无 warp 可发射"。原因是 B 的列访问是跨步的、不合并。
- **整条路线的结构性天花板是寄存器/占用率**：除 base 外所有版本都是 `Block Limit Registers = 2`，理论占用率被钉死在 **33.33%**（512 / 1536 threads/SM）。
- **优化过程就是"把瓶颈从 L1/TEX 搬到 SM/发射"**：L1/TEX 吞吐 96% → 45%，SM 吞吐 38% → 59%，发射率 0.29 → 0.65。
- **float4 是最有效的一步之一**：V2 → V2F 让 L1/TEX 吞吐从 73.9% 掉到 55.3%，V3 → V3F 从 83.9% 掉到 59.7%。
- **V4P 加 padding 的收益很小**（918 → 897 µs）：在 float4 + 正确排布之后，bank conflict 已经不是主要瓶颈。
- **V6（cp.async 多级流水线）反而比 V5 略慢**（831 vs 770 µs）：见 §4.5。
- 最好成绩 **V51 ≈ 24.0 TFLOPS**，约到 FP32 峰值的 **55%**；相对 base 提速 **8.7×**。

## 2. 各版本做了什么

| 版本 | kernel | 核心改动 |
| --- | --- | --- |
| base | `gemm_naive` | 每线程算 1 个 C 元素，16×16 block，无 shared 复用 |
| V1 | `sgemm_block_tiling` | Block tiling：`As[128][8]` + `Bs[8][128]`，256 线程协作加载，按 K 分 8 一块 |
| V2 | `sgemm_thread_tiling` | V1 + 线程级(寄存器)tiling：每线程 TM×TN = 8×8 |
| V2F | `sgemm_thread_tiling` | V2 + float4 向量化，A 转置成 `As[BK][BM]` |
| V3 | `sgemm_warp_tiling` | V1 + warp 级分块：warp tile 32×64，warp 内 4×8 排布 |
| V3F | `sgemm_warp_tiling` | V3 + float4 + A 转置 |
| V4 | `sgemm_warp_tiling_scatter` | warp 分块 + scatter（"z" 字）排布，`As[BK][BM]` |
| V4P | `sgemm_warp_tiling_scatter` | V4 + shared padding（`+4`）消 bank conflict |
| V4Z | `sgemm_warp_tiling_scatter` | V4P + 16B 向量化搬运 |
| V5 | `sgemm_warp_tiling_scatter` | + 双缓冲 `As[2]/Bs[2]` + 寄存器预取 + `__launch_bounds__(256,2)` |
| V51 | `sgemm_warp_tiling_scatter` | V5 的精简版：去掉 A_ITERS/B_ITERS 通用循环 |
| V6 | `sgemm_async_pipeline` | cp.async + `STAGES = 4` 多级流水线（global→shared 不经过寄存器） |

公共参数：BM = BN = 128，BK = 8，TM = TN = 8，BLOCK_SIZE = 256，grid = (16, 16)。

## 3. 实测性能

| 版本 | ncu Duration (µs) | 实测 (ms) | TFLOPS | 相对 base |
| --- | ---: | ---: | ---: | ---: |
| base | 7525.1 | 6.226 | 2.76 | 1.00× |
| V1 | 1371.6 | 1.120 | 15.33 | 5.56× |
| V2 | 1269.7 | 1.110 | 15.47 | 5.61× |
| V2F | 1120.2 | 0.892 | 19.27 | 6.98× |
| V3 | 1615.8 | 1.416 | 12.13 | 4.40× |
| V3F | 1036.3 | 0.824 | 20.84 | 7.55× |
| V4 | 918.4 | 0.773 | 22.23 | 8.05× |
| V4P | 896.8 | 0.774 | 22.18 | 8.04× |
| V4Z | 912.4 | 0.783 | 21.94 | 7.95× |
| V5 | 770.5 | 0.737 | 23.32 | 8.45× |
| **V51** | 783.5 | **0.715** | **24.03** | **8.71×** |
| V6 | 831.6 | 0.742 | 23.14 | 8.39× |

编译期资源（`nvcc -Xptxas -v`，全部 0 spill）：

| 版本 | registers/thread | shared/block | 理论占用率 | 理论 occupancy 限制来源 |
| --- | ---: | ---: | ---: | --- |
| base | 32 | ~0 | 100% | threads |
| V1 | 120 | 8.0 KB | 33.33% | **registers** |
| V2 | 94 | 8.0 KB | 33.33% | **registers** |
| V2F | 93 | 8.0 KB | 33.33% | **registers** |
| V3 | 94 | 8.0 KB | 33.33% | **registers** |
| V3F | 89 | 8.0 KB | 33.33% | **registers** |
| V4 | 111 | 8.0 KB | 33.33% | **registers** |
| V4P | 111 | 8.25 KB | 33.33% | **registers** |
| V4Z | 111 | 8.25 KB | 33.33% | **registers** |
| V5 | 128 | 16.5 KB | 33.33% | **registers** |
| V51 | 128 | 16.5 KB | 33.33% | **registers** |
| V6 | 128 | 32.5 KB | 33.33% | **registers**（shared 也降到 2 block/SM） |

## 4. ncu 剖析

### 4.1 关键指标总表

| 版本 | SM 吞吐 % | Mem 吞吐 % | **DRAM %** | L1/TEX % | L2 % | 实测占用率 % | cyc/inst | 发射/warp调度器 | L2 命中 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 92.8 | 92.8 | **3.9** | **96.0** | 29.2 | 98.5 | 40.2 | 0.29 | 98.4 |
| V1 | 42.7 | 42.7 | 4.9 | 43.7 | 15.4 | 30.4 | 9.20 | 0.41 | 89.0 |
| V2 | 49.4 | 69.0 | 5.5 | 73.9 | 21.1 | 30.0 | 6.81 | 0.53 | 90.9 |
| V2F | 44.4 | 52.9 | 4.8 | 55.3 | 20.8 | 31.1 | 8.03 | 0.46 | 90.9 |
| V3 | 38.9 | 78.0 | 3.9 | 83.9 | 16.1 | 30.0 | 8.64 | 0.42 | 83.5 |
| V3F | 48.1 | 57.3 | 6.1 | 59.7 | 22.5 | 30.8 | 7.42 | 0.50 | 91.2 |
| V4 | 49.9 | 42.6 | 9.4 | 44.9 | 21.6 | 30.4 | 6.93 | 0.52 | 90.3 |
| V4P | 51.2 | 42.2 | 8.9 | 43.8 | 22.1 | 30.4 | 6.87 | 0.53 | 90.0 |
| V4Z | 50.4 | 31.1 | 6.7 | 33.2 | 21.8 | 30.3 | 6.78 | 0.54 | 89.3 |
| V5 | 59.3 | 36.3 | 8.0 | 39.7 | 27.2 | 31.8 | 5.92 | 0.65 | 89.9 |
| V51 | 58.4 | 35.7 | 6.3 | 39.6 | 25.5 | 32.6 | 5.88 | 0.65 | 91.7 |
| V6 | 57.5 | 41.3 | 6.1 | 45.2 | 24.1 | 29.8 | **5.68** | 0.63 | 89.3 |

> 全表 `Avg. Active Threads Per Warp = 32`（无分支发散），`Shared Memory Spilling = 0`。

### 4.2 base：被 L1/TEX 卡住，而不是 DRAM

- DRAM 吞吐只有 **3.9%**，但 **L1/TEX 吞吐 96%**、SM 吞吐 92.8% —— 说明瓶颈在 LSU/L1 的**事务数**，不是显存带宽。
- 朴素实现里每个线程读 A 的一行（连续，好）和 B 的一列（跨步 2048×4 B，极差），B 的访问几乎每个元素一次独立事务 → L1/TEX 被撑满。
- 调度器 **70.6% 的周期"没有可发射的 warp"**（一个 warp 平均 40 个周期才发射一条指令）。
- 占用率高达 98.5% 也没用 —— 典型的"高占用率但被内存流水线拖死"。
- 结论：**base 的问题不是"数据搬得不够快"，而是"访问模式太差"**，所以下一步必须是 tiling（复用 + 合并访存），而不是加带宽或加线程。

### 4.3 中间版本：瓶颈从 L1/TEX 慢慢搬向 SM

- **V1 → V2**（加寄存器 tiling）：SM 吞吐 42.7% → 49.4%，cyc/inst 9.2 → 6.8，发射率 0.41 → 0.53。但 L1/TEX 仍然是 73.9%（V2），说明 shared/global 的事务数还很重。
- **float4（F 后缀）是最直接的一刀**：
  - V2 L1/TEX 73.9% → **V2F 55.3%**，1.110 ms → 0.892 ms；
  - V3 L1/TEX 83.9% → **V3F 59.7%**，1.416 ms → 0.824 ms。
  一次 16B 访存替代 4 次 4B，事务数直接除以 4。
- **V3 反而是"退步"**：它只做 warp 级分块、没做向量化，L1/TEX 冲到 **83.9%**、Mem 吞吐 78%，比 V2 还慢（1.416 ms）。对比 V3F 说明：**warp 分块的价值必须配合向量化访存才能兑现**。
- **V4 系列**把 L1/TEX 压到 ~44%（V4/V4P），Mem 吞吐回到 42%，SM 吞吐 ~50% —— 瓶颈基本完成了从"访存事务"到"发射/计算"的迁移。

### 4.4 所有优化版的共同天花板：33.33% 占用率

- 从 V1 开始，`Block Limit Registers = 2`（89~128 regs/thread）→ 每 SM 只驻留 **2 个 block = 512 线程**，理论占用率 **33.33%**，实测只有 ~30%。
- 关键指标 `Active Warps Per Scheduler ≈ 3.6`（上限 12），`Eligible Warps Per Scheduler` 只有 0.74~2.08 —— **可发射的 warp 长期不足**，这正是 `No Eligible` 长期占 35~58% 的原因。
- 也因此，V4→V5 的提升主要来自 **每 warp 的 ILP / 预取**（发射率 0.52 → 0.65，cyc/inst 6.93 → 5.92），而不是占用率。
- **想再往上走，必须降低寄存器压力或提高每线程 ILP**，把 2 block/SM 提到 3~4。

### 4.5 为什么 V6（cp.async 流水线）比 V5 慢

ncu 显示 V6 的"停滞指标"其实是全线最好的（cyc/inst 5.68 最低，Mem 吞吐 41.3% 低于 V5），**但它就是更慢**，原因有 4 点：

1. **它不是带宽瓶颈，异步拷贝用不上**：全程 DRAM 只有 6~9%，L2 命中 ~90%。数据本来就在 L2 里，cp.async 想掩盖的 DRAM 延迟并不存在。
2. **shared 用量翻倍，占用率反降**：STAGES = 4 让 shared 从 16.5 KB 涨到 32.5 KB，`Block Limit Shared Mem` 从 5 掉到 **2**；虽然理论上仍是 33.33%，实测占用率从 31.8% 掉到 **29.8%**。
3. **BK = 8 太小，流水线粒度太细**：K/BK = 256 个 k-tile，每个 tile 都要 `cp.async.commit_group` + `wait_group` + `__syncthreads()`。同步/指令开销摊在只有 8 次 K 迭代的 tile 上，占比过高。
4. **A 的 cp.async 只能用 4 B**：A 要转置存放，global 侧 16 B 连续块在 shared 侧被打散，只能用 `cp.async.ca ... 4`，每个 tile 要发 1024 条 4 B 拷贝，指令数反而不比 V5 的 `LDG.128 + 4×STS.32` 少。

> **顺带一个重要坑（最容易踩）**：如果**不带 `-arch`** 编译，nvcc 默认目标是 **`sm_75`**。
> `V6.cu` 里 cp.async 的代码是 `#if __CUDA_ARCH__ >= 800` 包起来的，sm_75 下**整段异步路径被编译掉**，退化成同步拷贝。
> 实测同一份 V6.cu：
> - `nvcc V6.cu -o V6`（默认 sm_75）→ **15.87 TFLOPS**
> - `nvcc -arch=sm_120 V6.cu -o V6` → **23.28 TFLOPS**
>
> 注意 `sm_75` 的 cubin 不能在 sm_120 上跑，会走 PTX JIT；JIT 出来的还是"没有 cp.async"的代码。
> **所以本仓库编译一律要加 `-arch=native`（或 `-arch=sm_120`）。**

### 4.6 V4P / V4Z 的 padding 与 16B 搬运：收益有限

| 对比 | 时间 | L1/TEX | 说明 |
| --- | --- | --- | --- |
| V4 → V4P（加 +4 padding） | 918 → 897 µs（-2.3%） | 44.9% → 43.8% | 有收益但很小 |
| V4P → V4Z（16B 搬运） | 897 → 912 µs（+1.7%） | 43.8% → 33.2% | L1 压力降了，但没变快 |

- 在 **float4 + scatter 排布**已经就位的情况下，shared 侧的 bank conflict 已经不是瓶颈（L1/TEX 只剩 44%），所以 padding 的边际收益很小。
- V4Z 把 L1/TEX 进一步压到 33%，但**吞吐没涨**——说明此时瓶颈已经不在访存事务，而在 SM 发射/占用率（SM 吞吐仅 50.4%，占用率 30.3%）。**继续优化访存已经没用了**，这也是 V4Z→V5 转向"双缓冲 + 提高 ILP"的原因。

## 5. 下一步可以做什么

1. **打占用率**（当前 33% 是最大天花板）：把 TM/TN 调到 4×8 之类，或换 block 大小，把寄存器压到 ~80 以下，争取 3 block/SM；或用 `__launch_bounds__` 强制压寄存器并观察 spill。
2. **加大 BK 到 16**：能摊薄每个 k-tile 的同步开销。注意 V6 需要把 shared 改成**动态分配 + `cudaFuncSetAttribute`**，因为静态 `__shared__` 上限是 48 KB（BK=16、STAGES=4 需要 66.5 KB，会报 `uses too much shared data`）。
3. **让 A 也用上 16B `cp.async`**：把 A 的 fragment 改成沿 K 的 4 个连续元素（k 向量化），或换 CUTLASS 那种 swizzle 布局。
4. **别再盲目加带宽/异步**：ncu 已经证明 DRAM 只用了 6~9%，L2 命中 90%，问题在 SM 发射与占用率，不在访存带宽。
5. 继续用 ncu 盯这几个数：`Achieved Occupancy`、`Issued Warp Per Scheduler`、`Eligible Warps Per Scheduler`、`Block Limit Registers`。

## 6. 如何复现

```powershell
# 1) 编译（务必带 -arch，看寄存器/寄存器占用）
nvcc -arch=native -Xptxas -v -o V6.exe V6.cu

# 2) 跑基准（程序自带 cudaEvent 计时）
.\V6.exe

# 3) ncu 剖析（需要管理员权限，否则报 ERR_NVGPUCTRPERM）
ncu --launch-skip 20 --launch-count 1 --set full -o report V6.exe

# 或只看关键 section
ncu --launch-skip 20 --launch-count 1 ^
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis ^
    --section SchedulerStats --section WarpStateStats V6.exe
```

> Windows 上让普通（非管理员）终端也能用 ncu：NVIDIA 控制面板 → 开发者 → 管理 GPU 性能计数器 →
> 允许所有用户访问；或每次用管理员终端运行 ncu。
