# cuda-kernel-lab

从零手写 LLM 推理核心算子的 CUDA 实现与性能剖析记录。

目标：把编译器/竞赛级的编码能力，兑换成**可量化的 GPU 性能产出**——
简历上只出现"实现了 / 优化了 / 加速了 N 倍"。

## 环境

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti（sm_120 / Blackwell，70 SM，16 GB） |
| FP32 SIMT 峰值 | 44.75 TFLOPS（70 SM x 128 FMA/SM/clk x 2 x 2.497 GHz） |
| 可达带宽 | 763.5 GB/s（float4 纯拷贝实测；GDDR7 标称 896 GB/s 的约 85%） |
| Tensor Core 锚点 | cuBLAS fp16：91.98 TFLOPS @2048^3 / 97.33 @4096^3 |
| CUDA | 13.4（nvcc V13.4.92） |
| 剖析 | Nsight Compute 2026.3.1（需先开性能计数器权限，见 `Operator/LayerNorm/README.md` §6） |
| 编译 | `nvcc -O3 -arch=native`（口径统一，见 `bench/README.md`） |

## 目录

```
bench/                统一测量口径 + cuBLAS 基线
Operator/
  Reduce/             V0 .. V8
  GEMM/               base, V1 .. V7（SIMT；含 float4 / 双缓冲 / cp.async 流水线）
  GEMM-TC/            V0 .. V6 + V5b（Tensor Core；该目录 README 是完整学习阶梯）
  SoftMax/            V0 .. V55（prof/ANALYSIS.md 是 ncu 剖析）
  OSoftmax/           V0 .. V4（online softmax）
  LayerNorm/          V0 .. V3
  RMSNorm/            V0 .. V1
  Fusion/             V0 .. V3（Residual+RMSNorm / +float4 / RMSNorm+RoPE / SwiGLU）
```

## 算子一览

> 每个数字的口径见 `bench/README.md`；每个算子的详细剖析见各目录的 README / note。
> **memory-bound 的算子看 GB/s（对可达带宽 763.5 GB/s），compute-bound 的看 TFLOPS。**

| 算子 | 版本数 | 最佳版本 | 关键数字 | 状态 |
| --- | ---: | --- | --- | --- |
| Reduce | 9（V0–V8） | V8 | 850.0 GB/s = 峰值 94.9%；V3 一步就到 88%，V4–V8 差异在 ±2% 噪声内 | 剖析完成 |
| GEMM（SIMT） | 13 | V51 | 24.03 TFLOPS @2048^3 = cuBLAS fp32 的 77%，8.71x vs base | 剖析完成 |
| GEMM（Tensor Core） | 8（V0–V6 + V5b） | V6 | 91.65 TFLOPS @4096^3 = **cuBLAS fp16 的 94%**；瘦长 shape 靠 split-K 拿 13.4x | 剖析完成 |
| Softmax | 7（V0–V5, V55） | V5 | 158.8 µs @4096²，769 GB/s = DRAM 峰值 86% | ncu 剖析完成 |
| OnlineSoftmax | 5（V0–V4） | V1 | 161.4 µs @4096²，6.9x vs V0；已到访存 roofline 的 92.6% | ncu 剖析完成 |
| LayerNorm | 4（V0–V3） | V0 | 717.6 GB/s = 纯拷贝上限的 94%；V1–V3 反而更慢（优化错了方向） | ncu 待补（权限） |
| RMSNorm | 2（V0–V1） | V0 | 723.6 GB/s = 纯拷贝上限的 94.8% | ncu 待补（权限） |
| Kernel Fusion | 4（V0–V3） | V2 | 字节账上限 1.25x / 2.00x / 1.67x → 实测 1.213x / 1.983x / 1.683x，全部顶到 DRAM roofline | 剖析完成 |

## 复现

```powershell
# cuBLAS 基线
nvcc -O3 -arch=native bench\cublas_sgemm.cu -o bench\cublas_sgemm.exe -lcublas
.\bench\cublas_sgemm.exe

# 单个算子（每个 .cu 都是独立可执行的 benchmark）
nvcc -O3 -arch=native Operator\GEMM\V51.cu -o Operator\GEMM\V51.exe
.\Operator\GEMM\V51.exe

# Tensor Core GEMM：V6 = 任意 shape 边界 + split-K（-DSPLITKV=n 切 K 方向段数）
nvcc -O3 -arch=native -Xcompiler "/utf-8" Operator\GEMM-TC\V6.cu -o Operator\GEMM-TC\V6.exe
.\Operator\GEMM-TC\V6.exe 0

# Kernel Fusion（V0 标量 → V3 SwiGLU；SwiGLU 的消融就是改 act() 里那一行）
nvcc -O3 -arch=native -Xcompiler "/utf-8" Operator\Fusion\V3.cu -o Operator\Fusion\V3.exe
.\Operator\Fusion\V3.exe
```

## 参考

- [AIInfraGuide](https://caomaolufei.github.io/AIInfraGuide/)
- CUDA C++ Programming Guide / Best Practices Guide
- LeetCUDA / GPU Puzzles / CUTLASS
