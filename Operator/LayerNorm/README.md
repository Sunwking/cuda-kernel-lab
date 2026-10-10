# LayerNorm

> 目录：`Operator/LayerNorm/`（V0 → V3，每个版本一个独立可执行的 `.cu`）
> 辅助脚本：`bench_bw.cu`（纯拷贝带宽上限）、`_reduce_cost.cu`（归约开销拆解）、
> `rmsnorm_block.cu` + `bench_rms.py` + `check_rms.py`（PyTorch 扩展版 RMSNorm 与数值校验）、
> `_ncu.ps1` / `_ncu_run.ps1`（批量剖析脚本，需管理员权限）

## 0. 口径

| 项 | 值 |
| --- | --- |
| 硬件 / 工具链 | 见 `bench/README.md`（RTX 5070 Ti，sm_120，nvcc 13.4，ncu 2026.3.1） |
| 问题规模 | M = 32768 行，D = 4096 列，N = M*D = 134217728，fp32 |
| 访存流量 | 512 MiB 读 + 512 MiB 写 = 1 GiB / 次 |
| 启动配置 | block = 256 thread，grid = M（一块处理一行），eps = 1e-5 |
| 计时 | 源码内 1 次 warm-up + 100 次迭代取平均（cudaEvent） |
| 正确性 | 与 CPU double 参考实现逐元素比对，报全量校验和 + max abs error |

> ⚠️ 本目录的绝对带宽数字有 **±8% 的 run-to-run 波动**（GPU boost 频率漂移），
> 跨 session 直接比较会得出错误结论，详见 §3.1。

## 1. 问题定义与理论上限

`y = (x - mean) / sqrt(var + eps) * gamma + beta`，对每一行独立计算。

- 每一行都要先求出 mean / var，再逐元素归一化 → **每行需要一次归约 + 一次写回**。
- 每个元素参与的浮点运算只有个位数，算术强度远小于 1 FLOP/Byte → **memory-bound**，
  瓶颈在 DRAM，不在 SM。
- **roofline 锚点**：本机 `bench_bw.cu` 的纯拷贝 kernel（float4，读写各 512 MiB）
  实测 **763.5 GB/s**，`cudaMemcpy D2D` 只有 732.0 GB/s。这就是这台机器
  "读 + 写" 的可达带宽上限（GDDR7 标称 896 GB/s 的约 85%）。
- **因此 LayerNorm 的合理上限就是 760 GB/s 量级，不可能超过纯拷贝。**
  这句话是后面所有分析的基准。

## 2. 版本演进

| 版本 | kernel | 核心改动 |
| --- | --- | --- |
| V0 | `layernorm_v0_kernel` | 一块一行。均值、方差各做一次 shared memory 树形归约（`sm[256]`），第三遍读 x 做归一化写回。**x 被读 3 遍** |
| V1 | `layernorm_v1_kernel` | 用 `warpReduceSum`（`__shfl_down_sync`）做 warp 内归约，`sm[32]` 只存跨 warp 的部分，减少 shared 访问 |
| V2 | `layernorm_v2_kernel` | float4 向量化 + **单遍 merge**（sum 与 sum_sq 在同一次遍历里算完），中间结果用 `float4 sm1[1024]` 缓存在 shared |
| V3 | `layernorm_v3_kernel` | 中间结果从 shared 挪到**寄存器**（`float4 v[ITEMS_MAX]`），x 全程只读一次，省掉 16 KB shared 占用和 4 次 shared 读写 |

## 3. 实测性能

### 3.1 两次独立运行的对比（这条比数字本身重要）

| 版本 | 文件头记录的 GB/s | 本次重跑 GB/s |
| --- | ---: | ---: |
| V0 | 770.2 | 717.6 |
| V1 | 753.5 | 671.5 |
| V2 | 754.6 | 698.9 |
| V3 | 749.2 | 638.1 |

- 绝对值差 **5~8%**：本次重跑紧跟在一次带宽压测之后，GPU 更热、boost 频率更低。
- **但排序两次完全一致：V0 最快，V3 最慢。**
- 结论：**跨 session 的绝对数字不能直接比**。要比较就在同一次 session 里 A/B。

### 3.2 与可达上限对比（本次重跑）

| 版本 | ms | GB/s | vs 纯拷贝 763.5 GB/s |
| --- | ---: | ---: | ---: |
| V0 | 1.4963 | 717.6 | 94.0% |
| V1 | 1.5991 | 671.5 | 88.0% |
| V2 | 1.5363 | 698.9 | 91.5% |
| V3 | 1.6827 | 638.1 | 83.6% |

> 文件头那一次 V0 是 770.2 GB/s，**已经略微超过纯拷贝的 763.5 GB/s**（差值在噪声内）——
> 也就是说不止 V0，连"最朴素"的那版都已经打满了这台机器的可达带宽。

## 4. ncu 剖析

**TODO**（未采集：`ncu` 需要 GPU 性能计数器权限，见 §6）

| 版本 | duration | DRAM% | L1TEX% | SM% | regs/thread | occupancy | 主要 stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V0 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| V1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| V2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| V3 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

要验证的假设（已有一半证据）：

> **V0 就已经是 DRAM-bound，V1~V3 的改动并没有减少 DRAM 流量（永远都是读 512 MiB + 写 512 MiB），
> 只是把寄存器用量和 shared 占用换了个位置 → 不可能变快。**

已有的间接证据：V0 的 717.6~770.2 GB/s 与纯拷贝的 763.5 GB/s 同量级。
ncu 需要补的是 `dram__throughput.avg.pct_of_peak_sustained_elapsed` 应接近 90%+，
而 `sm__throughput` 很低，且 V3 的 `launch__registers_per_thread` 明显高于 V0。

## 5. 面试问答

- **LayerNorm 是 memory-bound 还是 compute-bound？怎么证明？**
  与纯拷贝 kernel 对比带宽，再用 ncu 看 `dram__throughput` 占比。
- **为什么你写的 V1/V2/V3 反而更慢？**
  因为优化错了方向：kernel 已经打满 DRAM，减少 shared 访问 / 减少寄存器压力
  都不影响 DRAM 流量。这是 roofline 判断的典型案例。
- **LayerNorm 还有没有提速空间？**
  单算子没有。真实收益来自**融合**：`Residual + RMSNorm` 省掉一次 round trip，
  `RMSNorm + RoPE` 同理——减少 DRAM 流量才是唯一有效方向。
- **为什么 LayerNorm 不能上 Tensor Core？**
  算术强度太低（每元素个位数 FLOPs、4 字节访存），Tensor Core 帮不上。

## 6. 复现

```powershell
cd Operator\LayerNorm
nvcc -O3 -arch=native V0.cu -o V0.exe
.\V0.exe

# 带宽上限锚点
nvcc -O3 -arch=native bench_bw.cu -o bench_bw.exe
.\bench_bw.exe

# ncu（需要管理员权限，否则报 ERR_NVGPUCTRPERM）
# 方式一：管理员 PowerShell 里跑
#   ncu --csv --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active -s 1 -c 1 .\V0.exe
# 方式二：直接跑批量脚本（会弹 UAC）
#   .\_ncu_run.ps1
```

> **一次性解决 ncu 权限**：NVIDIA 控制面板 → 桌面/开发者 → *Manage GPU Performance Counters*
> → 选 *Allow access to the GPU performance counters to all users*。之后不用再提权。

---

**待办**：`note.md` 是 0 字节的空文件，可以删掉（本 README 已取代它）。