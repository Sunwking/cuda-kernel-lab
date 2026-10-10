# RMSNorm

> 目录：`Operator/RMSNorm/`（V0 → V1，每个版本一个独立可执行的 `.cu`）
> 相关：`Operator/LayerNorm/` 下的 `rmsnorm_block.cu`（PyTorch 扩展版）、
> `bench_rms.py`（与 `torch.nn.functional.rms_norm` 对比）、`check_rms.py`（数值/梯度校验）

## 0. 口径

| 项 | 值 |
| --- | --- |
| 硬件 / 工具链 | 见 `bench/README.md`（RTX 5070 Ti，sm_120，nvcc 13.4，ncu 2026.3.1） |
| 问题规模 | M = 32768 行，D = 4096 列，N = 134217728，fp32 |
| 访存流量 | 512 MiB 读 + 512 MiB 写 = 1 GiB / 次 |
| 启动配置 | block = 256 thread，grid = M（一块处理一行），eps = 1e-6 |
| 计时 | 源码内 1 次 warm-up + 100 次迭代取平均（cudaEvent） |

> ⚠️ 与 LayerNorm 相同，绝对值有 ±8% 的 run-to-run 波动，跨 session 不要直接比。

## 1. 问题定义与理论上限

`y = x / sqrt(mean(x^2) + eps) * gamma`（LLaMA 路线，**不减均值、没有 beta**）。

- 相比 LayerNorm 少了减均值这一步，只需要一次平方和归约 → 理论上比 LayerNorm 略快。
- 但两者的 DRAM 流量完全相同（读 512 MiB + 写 512 MiB），所以差距只体现在
  归约那一小部分指令数上，**上限同样由可达带宽决定**。
- roofline 锚点：本机纯拷贝 kernel 实测 **763.5 GB/s**（见 `Operator/LayerNorm/bench_bw.cu`），
  且 `bench_bw.cu` 里已经内置了 BLOCK=256/512/1024 三档 RMSNorm 的对比。

## 2. 版本演进

| 版本 | kernel | 核心改动 |
| --- | --- | --- |
| V0 | `rmsnorm_v0_kernel` | 一块一行，`warpReduceSum`（`__shfl_down_sync`）做 warp 内归约，`__shared__ float sm[16]` 存跨 warp 部分；x 读两遍（一遍算平方和，一遍归一化） |
| V1 | `rmsnorm_v0_kernel` | 在 V0 基础上加 float4 向量化 + `float4 reg[4]` 寄存器缓存中间结果，x 只读一遍 |

> ⚠️ V1 的 kernel 名字还是 `rmsnorm_v0_kernel`（从 V0 复制后没改），
> 建议改成 `rmsnorm_v1_kernel`，否则 ncu 里按 `-k` 过滤时会和 V0 混在一起。

## 3. 实测性能

### 3.1 两次独立运行的对比

| 版本 | 文件头记录的 GB/s | 本次重跑 GB/s |
| --- | ---: | ---: |
| V0 | 752.6 | 723.6 |
| V1 | 761.4 | 680.6 |

### 3.2 与可达上限对比（本次重跑）

| 版本 | ms | GB/s | vs 纯拷贝 763.5 GB/s |
| --- | ---: | ---: | ---: |
| V0 | 1.4840 | 723.6 | 94.8% |
| V1 | 1.5777 | 680.6 | 89.1% |

- 两次运行里 V0 / V1 谁快并不稳定（文件头那次 V1 更快，本次 V0 更快），
  但两者都在纯拷贝上限的 89~95% 区间 → **同样已经吃满 DRAM**。
- 与 LayerNorm 横向比：本次 RMSNorm V0 = 723.6 GB/s > LayerNorm V0 = 717.6 GB/s，
  差约 0.8%，符合"少了减均值那次归约"的预期，量级正确。

## 4. ncu 剖析

**TODO**（未采集：`ncu` 需要 GPU 性能计数器权限，见 `Operator/LayerNorm/README.md` §6）

| 版本 | duration | DRAM% | L1TEX% | SM% | regs/thread | occupancy | 主要 stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V0 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| V1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

待验证的假设：与 LayerNorm 同因——DRAM 流量不变，V1 的 float4 + 寄存器缓存
只改变指令数和寄存器压力，不会改变可达带宽，所以收益不稳定。

### 与 PyTorch 的对比（TODO）

`Operator/LayerNorm/bench_rms.py` 已经写好（x = 1024x4096，对比
`torch.nn.functional.rms_norm` 与自研扩展 `rmsnorm_v1`），但**还没跑过**：

```powershell
cd Operator\LayerNorm
python bench_rms.py
```

> 注意该脚本用 `time.time()` 做 CPU 侧计时，且 shape 是 1024x4096（约 33 MiB 流量，
> 单次约 40 us），测量噪声会明显大于本文的 1 GiB 版本。要写进简历的话，
> 建议改成 cudaEvent 计时 + 大 shape。

## 5. 面试问答

- **RMSNorm 相比 LayerNorm 省了什么？为什么 LLaMA 用它？**
  省掉减均值和 beta（少一次归约、少两个参数张量），在训练/推理里都更便宜，
  实测质量相当。
- **RMSNorm 还能再快吗？**
  单算子不能（已到带宽上限）。收益来自融合：`Residual + RMSNorm`、
  `RMSNorm + RoPE`，把两次 DRAM round trip 合成一次。
- **为什么 x 要尽量只读一遍？**
  它不减少 DRAM 流量（还是要读一次），但减少的是 L1/L2 往返和指令开销——
  在还没打满带宽时有用，打满之后就没用了。这正是 V1 收益不稳定的原因。

## 6. 复现

```powershell
cd Operator\RMSNorm
nvcc -O3 -arch=native V0.cu -o V0.exe
.\V0.exe
```

带宽上限锚点见 `Operator/LayerNorm/bench_bw.cu`。