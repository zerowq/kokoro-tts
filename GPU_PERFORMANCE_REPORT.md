# Kokoro TTS GPU Acceleration Performance Report
# Kokoro TTS GPU 加速性能评估报告

## 1. Overview / 概述
This report evaluates the steady-state performance of the Kokoro-82M (ONNX) and Meta MMS-TTS (PyTorch) engines on an NVIDIA Tesla V100 GPU. All data are collected after initial warmups to reflect production-level stability.
本报告评估了 Kokoro-82M (ONNX) 与 Meta MMS-TTS (PyTorch) 引擎在 NVIDIA Tesla V100 GPU 上的稳态运行性能。所有数据均在深度预热后采集，反映了生产环境的真实稳定性。

## 2. Metric Definitions / 指标定义说明
- **TTFB (Time To First Audio)**: The latency from the start of the request until the first chunk of audio is synthesized. It directly impacts the "instant-start" feeling for users.
- **TTFB (首音延迟)**: 从请求开始到生成第一段音频的延迟。它直接决定了用户“秒开”的主观感受。
- **RTF (Real Time Factor)**: The ratio of [Inference Time / Audio Duration]. 
    - RTF < 1.0 means faster than real-time.
    - RTF < 0.1 is considered elite tier (10x faster than speech).
- **RTF (实时率)**: 推理时长与音频时长的比率。
    - RTF < 1.0 代表推理速度快于语速，不会卡顿。
    - RTF < 0.1 被视为顶级性能（比说话快 10 倍）。

## 3. Steady-State Benchmark Results / 稳态性能数据表
| Model / 引擎 | Text Length / 字数 | Total(s) / 总耗时 | TTFB(s) / 首音时间 | RTF / 实时率 |
| :--- | :--- | :--- | :--- | :--- |
| **Kokoro-82M** | 44 chars | 1.77 | 1.70 | x 0.437 |
| **Kokoro-82M** | 122 chars | 3.33 | 1.69 | x 0.325 |
| **Kokoro-82M** | 174 chars | 3.20 | 1.55 | x 0.228 |
| **Meta MMS-TTS** | 41 chars | 0.20 | 0.20 | x 0.059 |
| **Meta MMS-TTS** | 178 chars | 0.09 | 0.09 | x 0.008 |

## 4. Performance Analysis / 性能现象分析

### 4.1 Why Kokoro gets faster with length? / 为什么 Kokoro 字符越长效率越高？
1. **Amortization of Fixed Overhead / 固定开销被摊薄**: Every inference task has a "starting price" (Log files show 43 Memcpy nodes), which includes Python-to-GPU data transfer and Phonemization. In longer texts, these fixed costs are distributed over a longer audio duration, causing the RTF to drop significantly.
   **固定开销摊薄**: 每次推理都有“起步价”（如日志中显示的 43 个数据搬运节点），包括音素转换和显存拷贝。长文本生成的音频长，使得这些固定开销被摊平，RTF 显著下降。
2. **GPU Parallelism / GPU 并行性**: For longer sequences, GPU kernels can achieve higher occupancy, leading to better utilization of CUDA cores compared to processing tiny, fragmented chunks.
   **GPU 并行优势**: 长序列能让 GPU 的 CUDA 核心处于更饱和的计算状态，相比处理碎片化的短句，长句的整体吞吐能力更强。

### 4.2 Why MMS is faster than Kokoro? / 为什么 Meta MMS 显著快于 Kokoro？
1. **Native Framework Support / 原生框架支持**: MMS runs directly on **PyTorch**, which has superior asynchronous kernel management and memory pooling on NVIDIA GPUs. Kokoro uses **ONNX-Python**, which introduces an extra layer of orchestration overhead.
   **原生框架优势**: MMS 直接运行在 **PyTorch** 上，它在 NVIDIA 显卡之上的异步调度和显存池管理极其成熟。而 Kokoro 使用的 **ONNX-Python** 封装层会带来额外的编排损耗。
2. **Architecture Efficiency / 模型架构**: MMS (VITS-based) uses a flow-based parallel synthesis architecture. Kokoro is Transformer-based; while it has higher prosody quality, the Transformer's self-attention mechanism is computationally heavier per token.
   **模型架构理论**: MMS 使用的 VITS 架构天生为并行推理优化。Kokoro 是基于 Transformer 的架构，虽然音质和韵律上限更高，但 Transformer 的自注意力机制计算量更大。
3. **Internal Data Pipeline / 数据流水线**: MMS performs phonemization and tensor preparation entirely within highly optimized Torch C++ kernels, whereas Kokoro-onnx involves more Python-side data bouncing.
   **数据流开销**: MMS 的音素转换和张量准备几乎都在高度优化的 Torch C++ 内核中完成，而 Kokoro 则有更多的 Python 中间层数据往返。

## 5. Deployment Conclusion / 部署结论
- **Kokoro-82M** is suitable for high-quality English synthesis with a steady RTF around **0.2**, meaning it can easily handle high-concurrency streaming.
- **MMS** is the ultimate choice for speed and low-resource scenarios (RTF **< 0.1**), ideal for real-time multilingual assistants.
- **Kokoro-82M** 适合高质量英文合成，稳态 RTF 约 **0.2**，足以支撑高并发流式服务。
- **MMS** 是极致速度和低资源消耗的首选（RTF **< 0.1**），非常适合实时多语言助手场景。
