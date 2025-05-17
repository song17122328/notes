## **正式Proposal方案**

### **标题**

**EdgeLLM: A Hybrid-Precision, Fault-Tolerant Framework for Efficient LLM Inference on Edge Devices via Prefill-Decode Phase Separation**

### **结构（总12页）**

1. **Abstract** (0.5页)
   - 核心问题：边缘设备上LLM推理的高延迟、高内存消耗、受限的吞吐量与低可靠性。
   - 解决方案：阶段混合精度(解决内存消耗，降低计算量)+容错调度()+KV传输优化。
   - 结果：
2. **Introduction** (1.5页)
   - 动机：边缘场景的延迟敏感与资源受限特性（对比云端推理）。
   - 现有不足：LLM-PQ未考虑阶段间精度差异，DistServe缺乏容错机制。
   - 创新点：
     - **阶段自适应混合精度**（Prefill-FP16/Decode-INT4）。
     - **边缘协同容错调度**（双调度池+设备降级接管）。
     - **分块异步KV传输**（通信延迟隐藏+压缩优化）。
3. **Background** (1页)
   - P/D分离范式（引用DistServe）。
   - 混合精度量化技术（引用MixQ、PROGRESSIVE MIXED-PRECISION DECODING）。
   - 边缘设备算力特性（Jetson Orin vs. Nano）。
4. **System Design** (4页)
   - **Phase-Separated Architecture**
     - Prefill节点（Orin NX）：张量并行+请求级批处理。
     - Decode节点（Nano集群）：流水线并行+迭代级批处理。
   - **Hybrid-Precision Pipeline**
     - Prefill阶段：FP16权重/激活，动态分块KV生成。
     - Decode阶段：INT4权重/激活，离群值FP16重计算（MixQ优化）。
   - **Fault-Tolerant Scheduler**
     - 双调度池优先级策略（预填充队列高优先级）。
     - 设备状态监控与KV缓存热备份。
5. **Implementation & Evaluation** (3页)
   - **Baselines**: DistServe（吞吐）、MixQ（精度）、SARATHI（分块）。
   - **Metrics**: TTFT/TPOT、内存占用、容错恢复时间。
   - **硬件平台**: Orin NX（Prefill）、Nano集群（Decode）。
   - **负载场景**: 高并发对话（>100 QPS）、长上下文（>4k tokens）。
6. **Discussion & Future Work** (1页)
   - 两阶段拆分为独立论文的可能性（见下文拆分建议）。
   - 扩展至多模态边缘推理（视频+文本联合处理）。
7. **References** (1页)

------

### **论文拆分建议（两篇独立论文）**

| **论文方向**                      | **核心内容**                                       | **实验重点**                           |
| :-------------------------------- | :------------------------------------------------- | :------------------------------------- |
| **Phase-Separated Scheduling**    | 阶段分离架构、双调度池设计、容错机制               | 吞吐/延迟对比（vs. DistServe/SARATHI） |
| **Hybrid-Precision Optimization** | Prefill-Decode混合精度策略、KV分块传输、离群值处理 | 精度/内存对比（vs. MixQ/LLM-PQ）       |

------

## **阶梯式实现计划**

### **阶段1：基础框架复现与验证（4周）**

- **目标**：搭建P/D分离原型，验证基础性能优势。
- **实验设计**:
  - **复现基线系统**: DistServe（P/D分离）、VLLM（标准推理）。
  - **模型**: Llama-2-7B（FP16）。
  - **硬件**: 单台Orin NX模拟Prefill/Decode分离。
  - **指标**: TTFT（Prefill延迟）、吞吐量（Token/s）。
- **关键技术验证**:
  - KV缓存异步传输的有效性（对比同步传输的延迟差异）。
  - 请求级批处理（Prefill）与迭代级批处理（Decode）的吞吐提升。

### **阶段2：混合精度集成与优化（6周）**

- **目标**：实现Decode阶段动态降精度，测试精度-效率权衡。
- **实验设计**:
  - **复现对比算法**: MixQ（统一混合精度）、PROGRESSIVE MIXED-PRECISION DECODING（渐进降精度）。
  - **模型**: Llama-2-7B（FP16→INT4）、GPT-NeoX-3B（测试小模型适配性）。
  - **精度校准**: 使用C4数据集测量Perplexity变化（允许损失<5%）。
- **关键技术验证**:
  - Decode阶段INT4量化+离群值FP16重计算的加速比（vs. FP16）。
  - 分块KV传输的带宽占用优化（对比原始传输的通信开销）。

### **阶段3：容错调度与边缘协同（4周）**

- **目标**：实现设备故障检测与降级接管，测试系统可靠性。
- **实验设计**:
  - **硬件平台**: Orin NX（Prefill）+ 2台Nano（Decode集群）。
  - **故障注入**: 随机关闭一台Nano，测量恢复延迟与精度损失。
  - **负载测试**: 模拟高并发请求（Locust工具生成>100 QPS）。
- **关键技术验证**:
  - 双调度池的优先级策略有效性（高负载下TTFT稳定性）。
  - 设备切换时KV缓存同步的完整性（对比输出Token一致性）。

### **阶段4：全系统集成与端到端优化（4周）**

- **目标**：联调所有模块，优化端到端性能。
- **实验设计**:
  - **端到端对比**: 对比EdgeLLM与LLM-PQ在边缘场景的延迟/内存/吞吐。
  - **长上下文测试**: 输入长度4k~8k tokens，测量内存压缩率。
- **关键技术验证**:
  - 混合精度+分块传输的联合优化效果（Roofline模型分析）。
  - 边缘设备资源利用率（GPU/CPU/Memory实时监控）。

------

### **关键模型与工具**

| **模块**     | **工具/模型**                   | **用途**                  |
| :----------- | :------------------------------ | :------------------------ |
| 基础框架     | VLLM（修改调度器）、PyTorch     | 实现P/D分离与批处理逻辑   |
| 混合精度     | BitsandBytes、AWQ               | FP16/INT4量化与离群值检测 |
| 通信优化     | NCCL、gRPC                      | KV缓存分块传输与设备同步  |
| 边缘设备管理 | Kubernetes Edge（KubeEdge）     | 容器化部署与故障监控      |
| 性能分析     | NVIDIA Nsight、PyTorch Profiler | GPU利用率与延迟热点分析   |

------

### **输出规划**

- **第一篇论文（系统架构）**: 聚焦阶段分离与调度（投稿**ASPLOS/PPoPP**）。
  - 核心图表：调度策略对比、容错恢复延迟、吞吐-延迟曲线。
- **第二篇论文（优化方法）**: 深入混合精度与传输优化（投稿**ICLR/MLSys**）。
  - 核心图表：精度-速度帕累托前沿、KV分块压缩率、离群值影响分析。