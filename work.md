那你的时间安排可以明显前移，而且这台 **270K Plus + RTX 5070 Ti** 正好能成为你后面 AI Infra 学习的主力机器。

你上传的路线本身把 **2026 年 10 月上旬** 定义为“Tensor Core + Nsight + Fusion”，10 月中旬开始 Triton + FlashAttention；你现在已经完成 Reduce、GEMM、Online Softmax，所以基本可以直接进入这一阶段。

## 现在重新定位你的进度

```text
2026.09
├─ CUDA 基础 ✅
├─ Vectorization ✅
├─ Reduce ✅
├─ GEMM ✅
└─ Online Softmax ✅

2026.10.07
↓
正式开始用自己的 RTX 5070 Ti
↓
Tensor Core
↓
Nsight Compute
↓
Kernel Fusion
↓
Triton
↓
FlashAttention
```

而且这时候用自己的卡，比你之前那张远程 **RTX 2080 Ti / sm_75** 更适合作为后续主实验平台。之前的 2080 Ti 可以继续保留做兼容性对照，但从 10 月 7 日开始，**你的主要 benchmark 建议统一放到 5070 Ti**。

---

# 一、10 月 1 日～6 日：先别等装机

这几天不用闲着。

你现在已经把 Online Softmax 做完了，所以：

### 10/1～10/3

完成：

```text
LayerNorm
RMSNorm
```

这里不要投入太多时间，目标就是：

```text
公式懂
↓
CUDA 能写
↓
正确性验证
↓
简单 benchmark
```

因为后面上传计划明确要求你做：

```text
RMSNorm + RoPE
Residual + RMSNorm
SwiGLU
```

这些 Fusion Kernel。

所以 LN/RMSNorm 是**必须会，但不是主战场**。

---

### 10/4～10/6

提前学习：

```text
Tensor Core 基础
↓
mma / fragment
↓
ldmatrix
↓
Warp-level Matrix Multiply
```

同时把 NCU 的理论补一下：

```text
Memory Throughput
Compute Throughput
Occupancy
Register
L2
Warp Stall
Roofline
```

这样 7 号拿到机器之后，你就能直接开始实验，而不是：

> “电脑到了，然后又花三天配置环境。”

---

# 二、10 月 7 日：新电脑开机后第一件事

不要立刻跑大模型。

**第一天把它变成 AI Infra 工作站。**

建议：

```text
Windows
  ↓
WSL2 / Linux 环境
  ↓
NVIDIA Driver
  ↓
CUDA Toolkit
  ↓
PyTorch
  ↓
Triton
  ↓
Nsight Compute
  ↓
Nsight Systems
  ↓
Git / CMake / GCC
```

然后立刻做一套：

```text
GPU 信息
CUDA version
PyTorch CUDA
compute capability
SM 数量
显存
```

再跑你已经做过的：

```text
Reduce
GEMM
Softmax
```

作为你的 **5070 Ti baseline**。

这样你会得到一个非常好的效果：

```text
RTX 2080 Ti
      vs
RTX 5070 Ti
```

甚至以后可以写一篇：

> **我的 CUDA Kernel 从 RTX 2080 Ti 迁移到 RTX 5070 Ti 后的性能变化**

但这个属于额外产出，不要专门花很多时间。

---

# 三、10 月 7 日～10 月 13 日：Tensor Core + Profiling

这是你新电脑真正开始发挥价值的地方。

上传路线要求这一阶段重点补齐 Tensor Core 与 Nsight，并把 GEMM 能力从普通 CUDA 提升到更完整的性能优化能力。

建议你这样学：

```text
10/7
Tensor Core 原理

10/8
mma / fragment

10/9
ldmatrix

10/10
Tensor Core GEMM

10/11
NCU Memory Analysis

10/12
NCU Compute + Roofline

10/13
Tensor Core GEMM 优化总结
```

### 最终必须留下的数据

至少：

```text
Naive GEMM
Shared Memory GEMM
Register Tiling GEMM
Tensor Core GEMM
cuBLAS
```

形成：

```text
Latency
TFLOPS
TFLOPS / cuBLAS
```

对比。

上传路线给的目标是 Tensor Core GEMM 能够逐步接近 cuBLAS，同时用 NCU 做完整分析。

不过这里我要特别调整一点：

**不要为了“达到 80%+ cuBLAS”卡死自己。**

你当前第一目标是：

> **理解 Tensor Core + 能写 + 能分析。**

性能数字是第二目标。

---

# 四、10 月 14 日左右：做一次真正的 Fusion

然后做：

```text
RMSNorm + RoPE
```

或者：

```text
Residual + RMSNorm
```

推荐前者。

因为它能自然连接后面的 LLM。

做：

```text
V0
RMSNorm Kernel
+
RoPE Kernel
```

然后：

```text
V1
RMSNorm + RoPE Fusion Kernel
```

比较：

```text
Kernel 数量
Global Memory Read/Write
Latency
Bandwidth
```

你上传的路线本身就把这类融合 Kernel 作为阶段产出。

---

# 五、10 月中旬开始：进入你真正的核心阶段

这里开始，我建议**严格参考你上传的路线**。

## 10 月中旬～11 月上旬

### Triton + FlashAttention

```text
W1
Triton 基础 + Triton-Puzzles

W2
Triton FlashAttention-2

W3
CUDA FlashAttention-2

W4
Paged Attention Decode
```

这和原计划完全一致。

而你的前置知识现在正好：

```text
Reduce ✅
GEMM ✅
Online Softmax ✅
Tiling ✅
Shared Memory ✅
Warp ✅
Vectorization ✅
```

所以 FlashAttention 对你来说已经不是“突然学一个新东西”，而是：

```text
GEMM
+
Reduce
+
Online Softmax
+
Tiling
+
Memory Optimization
+
Fusion
```

的综合题。

---

# 六、11 月中旬开始：从 Kernel 转向推理系统

这时候就不要继续无止境地刷 CUDA 了。

直接进入：

```text
Prefill
Decode
KV Cache
Continuous Batching
PagedAttention
Scheduler
```

上传计划把这一阶段安排为 **mini-vLLM**，四周逐步完成单请求 → Continuous Batching → PagedAttention → Chunked Prefill / Prefix Cache。

这会是你第一个真正意义上的 **AI Infra 系统项目**。

---

# 七、你的 2026.10～2027.03，我建议最终定成这个版本

| 时间              | 核心任务                                 | 产出               |
| --------------- | ------------------------------------ | ---------------- |
| 10/1～10/6       | LayerNorm / RMSNorm + Tensor Core 理论 | 基础补齐             |
| **10/7～10/13**  | **5070 Ti 环境 + Tensor Core + NCU**   | Tensor Core GEMM |
| 10/14～10/17     | Fusion                               | RMSNorm + RoPE   |
| **10月中旬～11月上旬** | **Triton + FlashAttention-2**        | CUDA/Triton FA2  |
| 11月中旬～12月上旬     | Mini-vLLM                            | 第一个系统项目          |
| 12月中旬～1月        | vLLM / SGLang 源码 + PR                | 开源贡献             |
| 2月              | 量化 + TP/EP/PD                        | 推理优化补齐           |
| 3月              | 简历 + Kernel 面试 + 投递                  | AI Infra 实习      |

这个时间表基本就是把你上传的计划与你**已经完成的内容**结合后的版本。 

---

# 八、而且你的 5070 Ti 到这里会非常有价值

你之前问过“什么时候真正需要自己的卡”。

现在答案可以更具体：

```text
9月
远程 2080 Ti
      ↓
CUDA 学习完全够

10/7
5070 Ti
      ↓
Tensor Core / NCU
      ↓
Triton
      ↓
FlashAttention
      ↓
KV Cache
      ↓
PagedAttention
      ↓
vLLM
```

尤其从 **FlashAttention → vLLM** 开始，你会大量进行：

```text
改 kernel
↓
重新编译
↓
换模型
↓
换 batch
↓
换 sequence length
↓
测吞吐
↓
测 latency
↓
NCU / Nsight Systems
↓
继续改
```

这时候自己的电脑会比远程机舒服很多。

---

## 你现在的第一优先级已经非常明确

**10 月 7 日之前：**

```text
LayerNorm
RMSNorm
Tensor Core 理论
NCU 理论
```

**10 月 7 日之后：**

```text
5070 Ti 环境搭建
↓
Tensor Core GEMM
↓
Nsight Compute
↓
Fusion
↓
Triton
↓
FlashAttention
```

所以你这台机器买得非常适合你接下来这条路线：**不是为了拿来训练大模型，而是拿来做高性能 Kernel、推理和 Profiling 实验。**

而且从你现在的进度看，**10 月 7 日刚好可以把“新电脑到手”作为你 AI Infra 学习第二阶段的起点**。
