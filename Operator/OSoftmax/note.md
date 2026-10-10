# CUDA Online-Softmax 优化路线性能分析（V0 → V4）

> 本文件基于本目录 \`Operator/OSoftmax\` 下当前版本的 5 个实现（V0、V1、V2、V3、V4），
> 用 Nsight Compute（ncu）实测剖析后整理，不含外部资料。
> 所有数据是**统一重编重跑**得到的：源码先快照到 \`prof/src/\`，再用 \`nvcc -arch=native -O3\` 编到 \`prof/build/\`，逐版本剖析。
> 原始报告在 \`prof/V0..V4.ncu-repz\`，文本导出的 details / raw 页面在 \`prof/V*.details.txt\`、\`prof/V*.raw.txt\`。

## 0. 测试环境与口径

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti（sm_120 / Blackwell，**70 SM**，16303 MiB，CC 12.0） |
| 驱动 / CUDA | 617.42 / CUDA 13.4（V13.4.92） |
| DRAM 峰值带宽 | ≈ **896 GB/s**（ncu 实测 756 GB/s ÷ 84.5% ≈ 895 GB/s） |
| 每 SM 资源上限 | threads 1536、warps 48、shared 100 KB 量级 |
| 编译 | \`nvcc -arch=native -O3\`（**必须带 -arch**，否则默认 sm_75 走 PTX JIT） |
| 计时 | 程序自带 cudaEvent，warm-up 后 200 次 kernel 取平均；本表取多次运行的最小值 |
| 剖析 | ncu 2026.3.1，\`--set full -k regex:online_softmax -c 1\` |
| 问题规模 | 除 **V3 = 8192×8192** 外，其余均为 **4096×4096**，fp32 |

> 说明：本次剖析期间 \`V3.cu\` 被改动过（\`reg_cache[4] → reg_cache[8]\`、规模 \`4096 → 8192\`），
> 因此**以本文件撰写时的源码快照 \`prof/src/V3.cu\` 为准**。另外 \`V3.cu\` 文件头的注释（M=4096、0.1809 ms）已经过期，
> 与当前代码不符。

## 1. 结论速览

- **V0 有两条病**：① 只有 16 个 block（\`grid = M/256\`），\`Waves Per SM = 0.04\`，70 个 SM 里绝大多数空转；② 一个线程读一整行是**跨步地址**，访存完全不合并，L1 事务数是必要的 **8 倍**。结果 L1/TEX 吞吐 91.5%、SM 吞吐 1.9%、占用率 16.7%。
- **V1 是整条路线唯一的大跃迁**：把"一行一个线程"改成"一行一个 block + warp shuffle 两级 online 合并"，立刻从"访存事务 + 并行度"瓶颈变成 **DRAM 带宽瓶颈**（DRAM 83.6%），耗时降到 1/7。
- **V2 的 float4 是有效但边际的改写**：指令数 33.2M → 26.2M、L1/TEX 21% → 19%、LSU 管线 15.5% → 7.1%，但耗时几乎不变（162 µs）——因为此时瓶颈已经在 DRAM，不在 L1。
- **V3 的"寄存器缓存 / one-pass"没有收益**：它省掉了第二遍全局读，但 ncu 显示 V2 的第二遍读本来就被 **L2 命中（L2 Hit 31%）**吃掉了，DRAM 读并没有少（V2 70.5 MB ≈ V3 按比例 71.7 MB/4096²）。one-pass 换来的只是更高的寄存器压力（34 → **51** regs、理论占用率 100% → **66.7%**）和更多次要停顿。
- **V4 的 grid-stride 持久化也没收益**：\`grid = 280\`（= 70 SM × 4）把 \`Waves Per SM\` 压到 0.67、占用率降到 64.3%，耗时与 V1/V2 持平甚至略慢。
- **V1~V4 的共同天花板就是 DRAM 带宽 ≈ 85%**：理论下界（134.2 MB ÷ 896 GB/s ≈ **150 µs**）与 V1/V2 实测 161 µs 只差 ~7%，**V2 已经基本贴脸访存 roofline**。想再快，只能**减少搬运的字节数**（融合上游/下游算子），靠调 kernel 本身已经没有空间。

## 2. 各版本做了什么

| 版本 | kernel | 核心改动 | grid × block |
| --- | --- | --- | --- |
| V0 | \`online_softmax_v0\` | 1 线程算 1 行；online 求 max/sum，再第二遍归一化 | **16** × 256 |
| V1 | \`online_softmax_v1\` | 1 block 算 1 行；每线程处理 \`N/block\` 个元素，warp shuffle + shared 两级 online 合并 | **4096** × 256 |
| V2 | \`online_softmax_v2\` | V1 + **float4 向量化**（读写各 16 B） | 4096 × 256 |
| V3 | \`online_softmax_v3\` | V2 + **寄存器缓存 \`reg_cache[8]\`**，一遍读入、一遍写回（one-pass） | 8192 × 256 |
| V4 | \`online_softmax_v4\` | V2 + **grid-stride 持久化**，一个 block 连续处理多行 | **280** × 256 |

公共参数：\`BLOCK_SIZE = 256\`；online 合并用 \`(m, d)\` 二元组，\`warpReduceOnline\` 做 shuffle 树合并。

## 3. 实测性能

| 版本 | 规模 M=N | ncu Duration | 实测 (ms) | GFLOPS | 相对 V0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V0 | 4096 | 1.40 ms | 1.247 | 67.3 | 1.00× |
| V1 | 4096 | 161.4 µs | 0.181 | 464 | **6.9×** |
| V2 | 4096 | 161.7 µs | 0.185 | 454 | **6.8×** |
| V3 | 8192 | 707.9 µs | 0.747 | 449 | （见下） |
| V4 | 4096 | 168.7 µs | 0.190 | 441 | 6.6× |

> **V3 的规模是 8192×8192（元素数是其他版本的 4 倍）**，不能直接和上面比。
> 按每元素归一化：\`0.747 ms ÷ 4 ≈ 0.187 ms\`（等价 4096²），即 **V3 与 V2 基本同速**——one-pass 没有带来提速。

编译期资源：

| 版本 | registers/thread | 理论占用率 | 理论占用率限制 | shared/block |
| --- | ---: | ---: | --- | ---: |
| V0 | 35 | 100% | warps | 16.4 KB |
| V1 | 28 | 100% | warps | 32.8 KB |
| V2 | 34 | 100% | warps | 32.8 KB |
| V3 | **51** | **66.67%** | **registers** | 16.4 KB |
| V4 | 40 | 100% | warps | 32.8 KB |

## 4. ncu 剖析

### 4.1 SpeedOfLight 关键指标

| 版本 | SM 吞吐 % | Mem 吞吐 % | **DRAM %** | L1/TEX % | L2 % | 实测占用率 % | 寄存器 | Waves/SM | Ipc(active) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | 1.9 | 28.4 | **16.9** | **91.5** | 28.4 | 16.7 | 35 | **0.04** | 0.34 |
| V1 | 29.5 | 83.6 | **83.6** | 21.1 | 35.8 | 92.7 | 28 | 9.75 | 1.28 |
| V2 | 23.4 | 84.5 | **84.5** | 18.6 | 37.0 | 89.8 | 34 | 9.75 | 0.97 |
| V3 | 16.3 | 85.6 | **85.6** | 15.5 | 25.7 | 65.8 | 51 | 29.26 | 0.70 |
| V4 | 21.3 | 84.3 | **84.3** | 17.6 | 33.5 | 64.3 | 40 | 0.67 | 1.05 |

- **V0**：唯一一个 \`Mem 吞吐 = 28%\` 却 \`L1/TEX = 91.5%\` 的版本 → 瓶颈在 **L1/TEX（访存事务）**，不在 DRAM。ncu 直接报两条：
  - \`This kernel grid is too small ... only 0.04 full waves\`；
  - 全局读写各有 \`88% excessive sectors\`（不合并）。
- **V1~V4**：ncu 给出一致的 \`SOLBottleneck\`：*"utilizing greater than 80.0% of the available memory performance ... Start by analyzing DRAM"* → 全部是 **DRAM 带宽瓶颈**。

### 4.2 访存数据（谁在搬多少字节）

| 版本 | DRAM 读 (MB) | DRAM 写 (MB) | 有效带宽 (GB/s) | L1 读 sectors | L1 写 sectors | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V0 | 145.7 | 65.1 | ~150 | **33,554,432** | **16,777,216** | 读两遍 + **8× 合并浪费** |
| V1 | 70.3 | 50.4 | 750 | 4,194,304 | 2,097,152 | 读两遍（第 2 遍吃 L2），全合并 |
| V2 | 70.5 | 51.8 | 756 | 4,194,304 | 2,097,152 | 同上，float4 |
| V3 | 286.6 | 255.8 | 768 | 8,388,608 | 8,388,608 | 8192²，读一遍（256 MB） |
| V4 | 69.2 | 58.0 | 755 | 4,194,304 | 2,097,152 | 同 V1/V2 |

关键点：

- **V0 的 L1 读 sectors 是 V1 的 8 倍**（33.5M vs 4.19M）。4096² 的输入只有 64 MiB，理想只需 \`64MiB×2 / 32B = 4.19M\` 个 sector；V0 却发了 33.5M，等于每个 32 B sector 只用了 4 B。**这就是 91.5% 的 L1 吞吐的来源**。
- **V1 的 DRAM 读只有 ~70 MB（≈ 一遍输入），而不是两遍的 134 MB** → 说明 V1 的第二遍（归一化前的重读）**被 L2 挡住了**。这直接解释了为什么 V3 的 one-pass 寄存器缓存**不可能**再省 DRAM：V2 本来就没有为第二遍付 DRAM 成本（L2 Hit 31%）。
- V1~V4 的有效带宽都在 750~770 GB/s，即 **DRAM 峰值的 84%~86%**。
- 理论下界：\`134.2 MB ÷ 896 GB/s ≈ 150 µs\`；V2 实测 161.7 µs（**≈92.6% 效率**），V1 161.4 µs。**V2 已经贴着访存 roofline 了。**

### 4.3 Warp 停顿原因（单位：cycle / 每次发射）

| 停顿原因 | V0 | V1 | V2 | V3 | V4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **long_scoreboard**（等全局访存） | **20.87** | **32.59** | **36.98** | **26.14** | **20.12** |
| short_scoreboard | 0.55 | 0.56 | 0.48 | 3.56 | 0.38 |
| wait | 1.13 | 1.59 | 0.77 | 0.86 | 0.99 |
| barrier（\`__syncthreads\`） | 0 | 1.38 | 3.79 | 3.68 | **6.04** |
| mio_throttle | 0.00 | 0.07 | 0.08 | 2.60 | 0.08 |
| lg_throttle | 0.00 | 0.00 | 0.00 | 3.09 | 0.00 |
| drain | 0.00 | 0.02 | 0.01 | 3.07 | 0.00 |
| not_selected | 0.05 | 0.79 | 0.58 | 0.56 | 0.48 |
| no_instruction | 0.02 | 0.10 | 0.14 | 0.06 | 0.75 |
| 平均 cyc/inst | 23.8 | 36.0 | 39.9 | 44.3 | 29.4 |

- 所有版本的主停顿都是 **long_scoreboard**（等 L1TEX/全局数据返回）→ 典型的"访存延迟/带宽受限"。
- **V4 的 barrier 停顿最高（6.04）**：持久化后每行之间都要经过两次 \`__syncthreads\`，而 \`Waves/SM\` 只有 0.67、并行度不足，同步开销摊不掉。
- **V3 出现 short_scoreboard / mio / lg / drain**：这是寄存器压力升高后（51 regs）调度余量变小、每条 warp 更长造成的副作用；ncu 确认 **没有 local memory 溢出**（\`Local Memory Spilling Requests = 0\`，local 访存全为 0），所以 reg_cache 确实待在寄存器里。

### 4.4 计算管线利用率（谁是最后 1% 的瓶颈）

| 管线 % | V0 | V1 | V2 | V3 | V4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FMA（FP32） | 1.26 | 15.46 | 14.32 | 11.33 | 13.39 |
| ALU（整数/地址） | 0.32 | 8.34 | 4.70 | 2.50 | 3.86 |
| **MUFU（expf）** | 1.29 | 14.06 | 14.78 | 11.51 | 13.48 |
| LSU（访存） | 1.29 | 15.46 | **7.07** | 3.75 | 6.54 |

- softmax 的核心计算是 \`expf\`（走 MUFU/SFU），各版本 MUFU 利用率 11%~15%，**远低于瓶颈**，说明计算不是问题。
- **V2 的 float4 把 LSU 从 15.5% 压到 7.1%**（一次 16 B 事务替代 4 次 4 B），但因为整体已经 DRAM 受限，这个改善没有转成时间收益。

### 4.5 为什么 V3 的 one-pass 没提速

1. **V2 的第二遍读本来就命中 L2**（L2 Hit 31%，DRAM 读 ≈ 一遍输入）→ 省不掉 DRAM 流量。
2. 8192² 下 V3 读到 286.6 MB、写 255.8 MB，DRAM 仍是 85.6%——**还是带宽瓶颈**，没有质变。
3. 代价却实打实：寄存器 34 → **51**，理论占用率 100% → **66.67%**，实测 89.8% → 65.8%，MUFU/LSU 利用率反而下降。
4. 结论：**"寄存器缓存消除第二遍读"这一招的前提是第二遍读会真正落到 DRAM**。在这个 N 和 L2 容量下前提不成立。

### 4.6 为什么 V4 的 grid-stride 没提速

- \`grid = 280 = 70 SM × 4\`，\`Waves Per SM = 0.67\`：整个 kernel 落不满一轮 wave，**尾部空转比例高**。
- 每处理完一行要 \`__syncthreads()\` 两次（合并结果 + 广播），而 block 数变少、占用率从 89.8% 降到 64.3%，**barrier 停顿从上不了榜变成最大非访存停顿（6.04）**。
- 数据量不变 → DRAM 依旧 84.3%，耗时反而比 V2 略高（168.7 vs 161.7 µs）。
- 结论：**当 grid 已经足够大（4096 行）时，再用持久化没有收益**，反而牺牲了并行度。

## 5. 下一步可以做什么

1. **V1~V4 已经是 DRAM 带宽瓶颈（~85%），调 kernel 本身收益极小**。想再快只有两条路：
   - 减少搬运字节：把 softmax 与相邻算子**融合**（如 attention 里的 PV 乘法），避免这次 128 MB 的读写往返；
   - 用更低精度存储中间结果（bf16/fp16 输入输出），把字节数直接砍半。
2. **V0 若还要留作教学对照**：至少改成"一个 block 一行"（即 V1）+ 合并访存，这一条就值 6~7×。
3. **回到 V2 作为基线**：V2 已接近 roofline（150 µs 下界 vs 162 µs 实测）。V3/V4 的尝试可以保留但不必作为主路径。
4. 想继续压时间可关注：
   - 提高 **L2 命中**（当前 V1/V2 只有 ~31%，读两遍还能再省）；
   - \`__syncthreads\` 次数（V4 的 barrier 停顿最高）；
   - 保持寄存器 ≤ 40 以维持 100% 理论占用率。
5. 继续用 ncu 盯这几个数：\`DRAM Throughput\`、\`Achieved Occupancy\`、\`Waves Per SM\`、\`smsp__average_warps_issue_stalled_long_scoreboard\`。

## 6. 如何复现

\`\`\`powershell
# 0) 让普通终端也能用 ncu（只需一次，需管理员）：
#    reg add "HKLM\SOFTWARE\NVIDIA Corporation\Global\NVTweak" /v RmProfilingAdminOnly /t REG_DWORD /d 0 /f
#    否则 ncu 会报 ERR_NVGPUCTRPERM；实在不行就用管理员终端跑 ncu。

# 1) 编译（务必带 -arch，否则默认 sm_75 走 JIT）
cd Operator\OSoftmax
nvcc -arch=native -O3 -o V2.exe V2.cu

# 2) 跑基准（程序自带 cudaEvent 计时，200 次取平均）
.\V2.exe

# 3) ncu 剖析（只抓 online_softmax 这一次 launch）
ncu --set full -k regex:online_softmax -c 1 -o prof\V2 V2.exe

# 4) 只看关键 section
ncu --set full -k regex:online_softmax -c 1 ^
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis ^
    --section SchedulerStats --section WarpStateStats V2.exe

# 5) 导出报告文本
ncu --import prof\V2.ncu-repz --page details
ncu --import prof\V2.ncu-repz --page raw
\`\`\`

> 本次剖析用到的辅助文件都在 \`prof/\` 下：\`src/\`（源码快照）、\`build/\`（统一编译产物）、
> \`V*.ncu-repz\`（原始报告）、\`V*.details.txt\` / \`V*.raw.txt\`（文本导出）、\`V*.log\`（程序 + ncu 输出）。
