# cuda-kernel-lab

从零手写 LLM 推理核心算子的 CUDA 实现与性能剖析记录。

目标：把编译器/竞赛级的编码能力，兑换成**可量化的 GPU 性能产出**——
简历上只出现"实现了 / 优化了 / 加速了 N 倍"。

## 环境

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti（sm_120 / Blackwell，70 SM，16 GB） |
| FP32 SIMT 峰值 | 44.75 TFLOPS（70 SM x 128 FMA/SM/clk x 2 x 2.497 GHz） |
| CUDA | 13.4（nvcc V13.4.92） |
| 剖析 | Nsight Compute 2026.3.1 |
| 编译 | `nvcc -O3 -arch=native`（口径统一，见 `bench/README.md`） |

## 目录

```
bench/                统一测量口径 + cuBLAS 基线
Operator/
  Reduce/             V0 .. V8
  GEMM/               base, V1 .. V7（含 float4 / 双缓冲 / cp.async 流水线）
  SoftMax/            V0 .. V55
  OSoftmax/           V0 .. V4（online softmax）
  LayerNorm/          V0 .. V3
  RMSNorm/            V0 .. V1
```

## 算子一览

> 每个数字的口径见 `bench/README.md`；每个算子的详细剖析见各目录 `README.md`。

| 算子 | 版本数 | 最佳版本 | 关键数字 | 状态 |
| --- | ---: | --- | --- | --- |
| Reduce | 9 | TODO | TODO | 剖析完成 |
| GEMM | 13 | V51 | 24.03 TFLOPS @2048^3，8.71x vs base | 剖析完成 |
| Softwaremax | 8 | TODO | TODO | 剖析完成 |
| OnlineSoftmax | 5 | TODO | TODO | 剖析完成 |
| LayerNorm | 4 | TODO | TODO | **待补 TODO** |
| RMSNorm | 2 | TODO | TODO | **待补 TODO** |

## 复现

```powershell
# cuBLAS 基线
nvcc -O3 -arch=native bench\cublas_sgemm.cu -o bench\cublas_sgemm.exe -lcublas
.\bench\cublas_sgemm.exe

# 单个算子（每个 .cu 都是独立可执行的 benchmark）
nvcc -O3 -arch=native Operator\GEMM\V51.cu -o Operator\GEMM\V51.exe
.\Operator\GEMM\V51.exe
```

## 参考

- [AIInfraGuide](https://caomaolufei.github.io/AIInfraGuide/)
- CUDA C++ Programming Guide / Best Practices Guide
- LeetCUDA / GPU Puzzles / CUTLASS