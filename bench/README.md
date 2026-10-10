# bench — 统一测量口径

> 目的：让仓库里所有"提升了 N 倍""达到 cuBLAS 的 X%"都有唯一、可复现的口径。
> 没有前后数字的优化，在面试里等于没做；口径不统一的数字，在面试里等于负分。

## 1. 报告任何性能数字前，必须交代的 5 件事

| 项 | 要求 | 为什么面试官会问 |
| --- | --- | --- |
| dtype / 精度 | fp32 / fp16 / bf16，两边必须一致 | fp16 TC 对比 fp32 SIMT 是"假加速"，最常见的坑 |
| 问题规模 | M/N/K 写全，至少 3 组 | 单一 shape 的好成绩会被当成过拟合 tile 参数 |
| 计时方法 | warmup ≥ 20 次，cudaEvent 计时，报 mean / **median** / min | 报最好一次 = 报运气；kernel 还要排除编译和首次分配 |
| 基线 | 同 dtype 的 cuBLAS + **理论峰值百分比** | "% of cuBLAS" 说明不了你离硬件上限多远 |
| 硬件与编译器 | GPU、SM 数、clock、CUDA 版本、`-arch` 参数 | `-arch=native` 与默认 arch 性能可能差一倍 |

## 2. 本机基线（RTX 5070 Ti，实测）

```
GPU            : NVIDIA GeForce RTX 5070 Ti (sm_120)
SM / clock     : 70 SMs @ 2.497 GHz
FP32 SIMT peak : 44.75 TFLOPS        # 70 SM x 128 FMA/SM/clk x 2 x 2.497 GHz
CUDA runtime   : 13.4
```

cuBLAS SGEMM（row-major，M=N=K，warmup 20，iters 100，cudaEvent）：

| shape | math mode | mean | median | min |
| --- | --- | ---: | ---: | ---: |
| 2048^3 | fp32 | 31.12 | **31.27** | 32.50 |
| 2048^3 | tf32 | 47.85 | 48.01 | 48.32 |
| 2048^3 | fp16 | 50.30 | **91.98** | 95.60 |
| 4096^3 | fp32 | 35.60 | **35.74** | 36.34 |
| 4096^3 | tf32 | 50.07 | 50.12 | 50.61 |
| 4096^3 | fp16 | 96.55 | **97.33** | 97.52 |
| 8192^3 | fp32 | 37.39 | **37.32** | 37.59 |
| 8192^3 | tf32 | 50.35 | 50.52 | 50.75 |
| 8192^3 | fp16 | 100.02 | **100.38** | 100.80 |

单位 TFLOPS。cuBLAS fp32 占 SIMT 峰值 72.2% / 79.9% / 83.4%。

**tf32 / fp32 倍数只有 1.35x ~ 1.54x**，说明 cuBLAS 的 fp32 路径走的是 SIMT（
如果它偷偷用了 Tensor Core，这个比值会是 2x 以上）。所以拿自研 fp32 SIMT kernel
去比 cuBLAS fp32 是**公平对比**——这句话本身就是面试答案。

**fp16（Tensor Core）的可达成绩是 92 / 97 / 100 TFLOPS，是 FP32 SIMT 峰值的 2.2 倍。**
这就是本仓库 Tensor Core GEMM 的目标基线：`Operator/GEMM-TC/` 以
**2048^3 fp16 ≥ 80 TFLOPS**（cuBLAS fp16 的 87%）为验收线。

> ⚠️ `2048^3 fp16` 那行的 mean（50.30）远低于 median（91.98）——少数慢迭代
> （时钟爬坡）把均值拉垮了。**结论一律用 median / min，不要用 mean。**

## 3. 复现

```powershell
nvcc -O3 -arch=native bench\cublas_sgemm.cu -o bench\cublas_sgemm.exe -lcublas
.\bench\cublas_sgemm.exe
```

## 4. 自研 GEMM vs 本基线

| 版本 | shape | 实测 TFLOPS | 占 SIMT 峰值 | vs cuBLAS (median) |
| --- | --- | ---: | ---: | ---: |
| V51 | 2048^3 | 24.03 | 53.7% | 74.4% |
| V51 | 8192^3 | 29.96 | 66.9% | 80.3% |

> 注意：V51 的 2048^3 数字取自 `Operator/GEMM/note.md`，8192^3 取自
> `V51.cu` 文件头。**两者需要在同一口径下重跑一遍**再写进简历。

### 4.1 "83.28%" 的来历与修正

`main.cu`（仓库根目录的历史 scratch 文件）记录了一次 cuBLAS 的单跑结果：

| shape | cuBLAS 单跑 | 耗时 |
| --- | ---: | ---: |
| 2048^3 | 30.181 TFLOPS | 0.569 ms |
| 8192^3 | 35.968 TFLOPS | 30.569 ms |

`V51 @8192^3 = 29.9565 TFLOPS`，除以 35.968 正好得到 **83.28%** —— 这就是
`Operator/GEMM/V51.cu` 文件头那句注释的出处。

问题在于那次 cuBLAS 基线是**单次运行、没有 warm-up**。本仓库用 20 次
warm-up + 100 次取 median 重测，8192^3 的 cuBLAS 是 **37.31 TFLOPS**
（最好一次 29.261 ms，对应 37.58 TFLOPS）。热机后 cuBLAS 能吃到更高的
boost 频率，所以真实比值是 **80.3%**，而非 83.28%。

结论：**不要拿单次、无 warm-up 的对手成绩当分母**。3 个百分点看着不多，
但面试官问"你怎么保证对比公平"时，这个口径答不出问题。