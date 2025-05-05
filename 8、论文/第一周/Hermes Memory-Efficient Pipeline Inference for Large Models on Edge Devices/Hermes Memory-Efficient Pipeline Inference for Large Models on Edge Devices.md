# Hermes: Memory-Efficient Pipeline Inference for Large Models on Edge Devices

## 1. 论文基本信息

- **标题**：Hermes: Memory-Efficient Pipeline Inference for Large Models on Edge Devices
- **作者**：Xueyuan Han, Zinuo Cai, Yichu Zhang, Chongxin Fan, Junhan Liu, Ruhui Ma, Rajkumar Buyya
- **期刊/会议**：IEEE International Conference on Computer Design
- **发表年份**：2025
- **DOI/链接**：[[2409.04249\] Hermes: Memory-Efficient Pipeline Inference for Large Models on Edge Devices](https://arxiv.org/abs/2409.04249)

------

## 2. 论文总结

- **研究背景**：
  Transformer大模型在边缘设备部署时面临内存不足和推理延迟高的挑战。传统方法（如模型压缩、内存交换）会牺牲精度或增加延迟，亟需高效的内存优化和流水线执行机制。
- **研究问题**：
  1. **内存挑战**：大模型参数规模远超边缘设备内存容量（如GPT-3达1750亿参数）。
  2. **流水线停滞**：模型加载延迟显著高于推理延迟（相差1个数量级），导致计算资源闲置。
- **研究方法**：
  提出**PipeLoad机制**：
  - **动态内存管理**：推理后立即销毁已用层的内存。
  - **并行加载**：多线程加载模型层，覆盖加载与推理时间差。
    基于PipeLoad设计**Hermes框架**，包含层分析器（Layer Profiler）、流水线规划器（Pipeline Planner）和执行引擎（Execution Engine）。
- **研究结果**：
  - 在BERT/ViT模型上，相比PipeSwitch（PipeSwitch实现了模型加载和推理的并行，但模型加载延迟往往远大于推理延迟，因此需配置多个加载代理并行模型加载减少模型加载延迟），速度提升**4.24倍**，内存占用降低**86.7%**。
  - 在GPT类模型上，速度提升**2.58倍**，内存占用降低**90.3%**。
- **贡献点**：
  1. 首个同时优化内存和延迟的边缘大模型推理框架。
  2. 无需GPU支持，纯CPU实现高效流水线。

------

## 3. 研究问题

> **问题阐述**：
>
> - **内存限制**：边缘设备（如NVIDIA Jetson Nano仅4GB内存）无法直接加载大模型（如GPT-J达12GB）。
> - **延迟瓶颈**：传统流水线因加载与推理延迟不匹配，导致**60-80%时间闲置**。
> - **现有缺陷**：模型压缩降低精度，内存交换增加I/O延迟，PipeEdge等方案未优化内存。

------

## 4. 研究方法

> **研究设计**：
>
> - **实验模型**：BERT-Large、ViT-Large、GPT-2、GPT-J，覆盖NLP/CV/生成任务。
> - **测试平台**：Intel Xeon Gold CPU，Docker模拟边缘资源限制。
>
> **技术方法**：
>
> 1. **PipeLoad机制**：
>    - **多代理协作**：Loading Agents并行加载层，Inference Agent顺序推理，Daemon Agent动态释放内存。
>    - **信号机制**：通过计算就绪信号（Scomp*S*comp）和内存销毁信号（Sdest*S*dest）协调流程。
> 2. **Hermes框架**：
>    - **Layer Profiler**：分析每层的加载时间、内存占用。
>    - **Pipeline Planner**：根据内存约束规划最优Loading Agents数量。

------

## 5. 研究结果

> **实验平台**：
>
> - CPU：Intel Xeon Gold 6248R，限制8核和特定内存（通过Docker --memory）。
>
> **对比方法**：
>
> - **Baseline**：非流水线加载后推理。
> - **PipeSwitch**：标准流水线方案。
>
> **实验结果**：
>
> - **BERT-Large**：6个Loading Agents时，速度提升4.24倍，内存降至基线28.1%（表II-III）。
> - **内存约束测试**：ViT在300MB内存下延迟36ms，较60MB降低55.6%（图7a）。

------

## 6. 研究总结与个人理解

> **研究总结**：
> Hermes通过动态内存管理和并行加载，显著降低内存占用和延迟，适用于资源受限的边缘设备，且无需修改模型结构或依赖GPU。
>
> - 创新点：
>   1. 动态内存管理：通过Daemon Agent及时销毁已推理完成的模型层，显著降低内存占用（相比现有方法最高减少90.3%内存）相比PipeEdge：守护代理及时销毁使用过的内存空间，减少内存消耗
>   2. 并行加载机制：采用多个Loading Agent并行加载模型层，将多个推理时间重叠在单个加载时间内，缓解了管道停滞问题。相比PipeSwitch：并行模型加载减少加载延迟。
>   3. 分层粒度优化：专注于占用内存最多的encoder/decoder层进行优化，针对性更强
> - 局限：
>   1. 对GPT类生成式模型优化有限：由于需要逐token加载和推理，当输出token较多时延迟改善不明显
>   2. 内存与速度的权衡：增加Loading Agent数量可提升速度，但会减少内存节省幅度
>   3. 对非Transformer架构的适用性未验证
> - 改进方向：
>   1. 针对生成式模型的优化：可以探索参数预取等策略来改善多token生成的
>   3. 扩展到其他模型架构：验证对CNN、RNN等架构的适用性