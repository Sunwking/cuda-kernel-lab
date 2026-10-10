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
| 2048^3 | fp32 | 31.73 | **32.32** | 32.87 |
| 2048^3 | tf32 | 46.96 | 47.99 | 48.37 |
| 4096^3 | fp32 | 35.65 | **35.76** | 36.35 |
| 4096^3 | tf32 | 49.94 | 50.15 | 50.60 |
| 8192^3 | fp32 | 36.44 | **37.31** | 37.58 |
| 8192^3 | tf32 | 50.35 | 50.51 | 50.74 |

单位 TFLOPS。cuBLAS fp32 占 SIMT 峰值 72.2% / 79.9% / 83.4%。

**tf32 / fp32 倍数只有 1.35x ~ 1.48x**，说明 cuBLAS 的 fp32 路径走的是 SIMT（
如果它偷偷用了 Tensor Core，这个比值会是 2x 以上）。所以拿自研 fp32 SIMT kernel
去比 cuBLAS fp32 是**公平对比**——这句话本身就是面试答案。

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