

# 第1篇系统：ORCA

**[16th OSDI 2022 ] Orca: A distributed serving system for {Transformer-Based} generative models**

##### 1) 核心技术：迭代级调度，选择性批处理

**迭代级别调度**：将云端模型推理过程中的请求批处理拆分成迭代批处理。将原来的将一批请求放到GPU里面并行计算转变化将一个请求拆分为不同的迭代(迭代可以理解为生成一个token，也即一个解码过程)。通过配置并行处理不同请求的不同迭代过程，避免了短请求等长请求的现象，减少了系统延迟，增加了模型推理的并行程度。

迭代级调度图(图4)：

<img src="https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250501142853978.png" alt="Orca简要结构图" style="zoom:50%;" />

$X_i$表示第i个请求，$X_{ij}$表示第i个请求的第j个token，阴影的表示预填充的输入token，无阴影的白色块表示解码生成的token

上图中$x_1$表示第1个请求处在解码阶段，一共有4个token的上下文缓存，$x_2$也在解码阶段，但只有两个上下文缓存，$x_3$和$x_4$都在预填充阶段，分别要处理2个输入tokne和3个输入token



迭代级调度的问题：

> (1) both requests are in the initiation phase and each has different number of input tokens (e.g.,x3 and x4 in Figure 4); (2) both are in the increment phase and each is processing a token at different index from each other (x1 and x2); or (3) each request is in the different phase: initiation or increment (x1 and x3).

（1）两个请求都处于预填充阶段，且每个请求都有不同数量的输入token（例如，图 4 中的 x3 和 x4）；（2）两个请求都处于解码阶段，且每个请求都在处理不同索引处的token (上下文缓存)（x1 和 x2）；或（3）每个请求处于不同的阶段：预填充或增量(解码)（x1 和 x3）。

这种情况是ORCA迭代级调度处理不好的问题，因此ORCA提出了选择性批处理来提高这三个情况的效率，但在后面会发现，即使用了选择性批处理，整个流水线仍然会有较多的停顿和气泡，浪费了GPU周期，vLLM对此的改进是只保留(1)和 (2), 也即每次前向传递(推理) 只处理预填充或只处理解码，一定程度上减少了流水线气泡，Sarathi又对vLLM和ORCA进行了进一步的优化，以分块预填充和无停顿流水线(混合批处理技术) 极大地提高了流水线的效率。

**选择性批处理**：

对除了Attention层的其他层如FC（全连接层）、layerNorm(层增正则化)进行批处理

对Attention层不进行批处理，而是按照每个请求单独处理。

这是因为批处理需要每个处理单元的shape一致，而Attention层需要看每个处理单元之前的token，不能保证shape一致 (不同的请求有不同的前序token，即KV缓存)。

选择性批处理

<img src="https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250501144213318.png" alt="image-20250501144213318" style="zoom:50%;" />

##### 2） 系统的分布式架构：

![image-20250519175133762](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250519175133762.png)

图6展示了ORCA的并行划分策略，使用了流水线并行(层间划分) 和张量并行 (层内划分)

例如，把一个4层的transformer模型按层间划分为两个worker ——worker1 负责前两层，worker2负责后两层。

每个worker管理多个CPU线程，每个线程管理一个GPU，GPU数量取决于张量并行的并行度。

图7 完整地展示了ORCA的分布式系统执行引擎的工作流程：

**具体流程**：

一批请求到来，调度器调度这批请求给执行引擎，**Engine Master**组件把这批次请求的**相关信息发送给worker1**

> 这些相关信息包括：
>
>  1、当前批次请求上次迭代产生的token 
>
>  2、这批请求的所有id 
>
>  3、对于解码阶段，提供当前token的索引
>
>  4、对于预填充阶段，提供输入token的数量

**worker1的控制器把从Engine Master收到的信息发送给GPU控制线程，线程解析这些信息并把合适的GPU kernels (指的是GPU需要执行的核函数、算子) 发送到对应的GPU**

> 例如：Attention的kernel 使用请求的id和token的索引 可以获得 Attention K/V管理器保存KV cache的GPU内存地址

在worker1做 核函数匹配的同时，**worker1的控制器会异步把这些控制信息发送给Worker2的控制器**。

但对于**最后一个控制器(Woker2的控制器)*，它不会(直接发送)异步发送信息给Engine master，而是**同步 (等待)** 分配好GPU**核函数**在对于的GPU**完成后**，获取到输出的token，再把控制信息发送给Engine master.



**ORCA engine 最小化 CPU和GPU之间的同步开销**，

> 它们举出例子——FasterTransformer和Megatron-LM 张量并行
>
> 无论哪一个进程接收控制消息时，都有CPU-GPU的同步开销，这是因为它们使用NCCL来进行GPU通信 (NCCL是一个GPU之间的高速通信库，但将其用于控制消息的时候，会导致CPU等待GPU)
>
> 每次迭代都会有GPU-CPU的同步开销，累计起来是一个无法忽视的总的性能开销。

ORCA处理这个问题的**方法是**：分离了张量数据传输和控制信息传递的通信信道，**张量数据通信使用NCCL完成(图7中的虚线表示)**,  **控制消息使用单独的通道 (如gRPC) 在worker和worker之间， worker和engine master之间传递**。

#####  3）调度算法

ORCA调度程序决定**每次迭代**选择哪些请求，选择性批处理技术允许任何阶段的请求放到一起处理，具有高度灵活性，问题变成如何在每次迭代中选择一批请求。

ORCA调度器算法：定义了迭代级别的FCFS配置——$对于请求池中的任意一对请求(x_i,x_j) 如果i比j早，则x_i的迭代次数≥x_j$

因此：如果j更迟到达，但是j的迭代次数更少，j也可以更早return

考虑其他因素：批次大小带来的边际减少的收益 (如吞吐量随着批次大小增加而增加，但是边际收益减小)，因此设置最大批次大小。

因此调度器设置了一个最大批次大小参数，调整批次大小参数来确保满足SLO条件下的最大吞吐量。

对于内存限制，ORCA也采用了重用中间值缓冲区的技术来优化内存使用 (对于重用KV缓存技术，MoonCake在FAST2025有更精彩的工作)

具体算法：

* select阶段：

每到来一个新请求放入请求池，执行select算法从请求池中选择请求，先按请求到来顺序对每个请求池中的请求排序，然后对每个请求池中的请求做一遍for-loop，如果batch.size() (已经选的批次大小)等于max_bs，说明批次已经满了，跳出循环，select结束。

如果batch没有满，对于每一个初始状态的请求，若其需要的最大token数量req.max_token+当前给其他请求预留槽位的数量 n_rsrv<总的KV管理器的槽数 n_slot 则可以调度该请求，把请求放入bach中，n_rsrv+=req.max_token。如果不满足条件，则判断下一个。

* 调度算法阶段：

把上一个select阶段选择好的batch中所有请求送入schedule engine并把状态设置为运行态RUNNING，n_scheduled+=1，循环这个过程，直到n_scheduled=n_worker，即所有批次把流水线占满。然后等待一个scheduled batch返回，把这个批次中所有请求状态设置为增量状态INCREMENT，并检查每个请求是否完成，如果一个请求完成，则n_rsrv-=req.max_tokens。一个scheduled batch执行完后把n_scheduled batch减一。

![image-20250519200227086](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250519200227086.png)

##### 4）流水线调度

![image-20250519200200087](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250519200200087.png)

图8所示，

ORCA那样有三个worker可以同时处理三个batch的推理，气泡跟每个阶段的处理时长有关，较为合适的迭代时间可以充分利用流水线，减少气泡，但分块预处理(18th OSDI 2024: Sarathi-Serve）进一步处理了气泡问题。

FasterTransformer 也实现了流水线并行，FasterTransformer 中采用类似迭代及别调度的micro-batch(微批次)来实现流线型推理。

如FasterTransformer 将批次 AB 拆分成两个微批次：A 和 B。由于每个分区都以分批方式处理一个微批次（小于原始批次），因此分批处理带来的性能增益可能会减小。此外，当微批次过大时，此方法可能会在流水线中插入气泡，导致微批次的数量小于分区的数量。FasterTransformer 需要牺牲分批效率（大批次引入流水线气泡，所以使用小批次来减少气泡）来换取流水线效率（更少的流水线气泡），而 ORCA 则无需进行这样的权衡——这要归功于迭代级调度——并且可以轻松地将请求流水线化，而无需将批次拆分成微批次。

##### 5）实现：基于CUDA生态的C++实现

* 使用gRPC进行ORCA引擎控制平面通信，NCCL进行数据层面和层内通信
* 做了原始的编码器解码器架构的Tranformer、GPT等实现
* 做了算子融合



# 1.5 ORCA的进一步改进：Sarathi——分块预填充技术

**[ArXiv:2308.16369](https://arxiv.org/abs/2308.16369) SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills**

##### 1）**两个挑战**

Sarathi提出了两个挑战：一个是预填充阶段较小的批次大小就能跑满GPU，如Llama 13B 512的token长度，batch_size=1 就能跑满A6000 GPU，但是这个批次大小显然对解码效率非常低，针对这个问题，文章一作在后面一篇文章 Sarathi-Serve 设计了一个在线的LLM推理架构，权衡了预填充和解码阶段的吞吐量和延迟，Sarathi-Serve这篇文章也中了 18th OSDI 2024，是一篇考虑prefill 和decode不同特性的经典之作。

另外一个挑战: 像ORCA那种把不同推理长度的预填充解码请求混合，会导致不同的迭代 (微批次) 处理时间，从而导致严重的气泡的GPU浪费，针对这一问题，作者提出了**分块预填充**和 **解码搭载预填充块** 技术，优化了气泡问题，并把其引入到流水线中。这个技术也是后面 Sarathi-Serve这篇文章用到的主要技术。

##### 2）主要技术-创新点

![image-20250519210145226](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250519210145226.png)

如图所示，ORCA的三个气泡：PB1,PB2,PB3

PB1是两个连续的微批次 预填充阶段token长度不同导致的气泡

PB2是两个连续的微批次一个批次处于预填充，一个批次处于解码 ，两阶段不同的计算量导致的气泡。

PB3是两个连续的微批次 解码阶段累计上下文长度 (KV缓存长度) 不同导致的气泡。

如果能让每个微批次均匀的执行，就能减少气泡，这篇文章提出了**分块预填充**和 **解码搭载预填充块** 技术

使用单个预填充块构建批次，在剩余的槽位上搭载解码的token，可以减少每个微批次的运行时差异

![image-20250519205639582](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250519205639582.png)

如图所示，把A的预填充切两块，B的预填充切三块，C的预填充切两块，D的预填充切两块。

$B_{p1}和A_{p1}形成了较好的流水线，没有气泡，C_{p1}和B_{p2}有运行时差异，给短的C_{p1}搭载一个解码的A_{d1}来消除运行时差异，从而减少气泡  $

通过这种方法来减少流水线气泡，论文后面的内容主要讨论如何分块，如何搭载，直接跳过。



# 第二篇系统：Sarathi-Serve

**[18th OSDI 2024]  Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**

##### 1）核心技术：分块预填充技术、无停顿调度技术



Sarathi-Serve是一个平衡吞吐量和延迟，使用分块预填充和无停顿调度技术的调度器。虽然说是一个调度器，但其实没有具体的系统设计，还是跟Sarathi类似的一个算法性质的文章。

下面是几种调度框架的调度策略对比：

![image-20250519214713134](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250519214713134.png)

如上图所示，几种不同调度器的调度策略，ORCA和vLLM都是预填充优先的调度器，而FasterTransformer是解码优先的调度器。

Orca 和 vLLM 都使用 FCFS 迭代级批处理。Orca 支持由预填充和解码请求组成的混合批处理，而 vLLM 仅支持包含所有预填充或所有解码请求的批处理。尽管存在这种差异，Orca 和 vLLM 都可以通过最大化批处理大小来提高吞吐量。

vLLM 在恢复解码前会安排的预填充，这引入了气泡(停顿)，但vLLM这样做的目的是提高解码的批次大小。

ORCA使用混合批次，但是也有气泡，正如前面提到的一样。

FasterTransformer 原则上没有生成停顿，因为它是解码优先的，在安排新的预填充之前完成了所有正在进行的解码，虽然没有气泡但是解码批次小，并行度低，在吞吐量上做出了较多的妥协，因此以同样的标准来看, FasterTransformer 的GPU利用率低，性能并不太好

前两者牺牲了TBT（TPOP）的时间以换取吞吐量，而第三者考虑了TBT时间的最优，但吞吐量上不去。

这三者在吞吐量和延迟上都没有Sarathi-Serve 做的好。

相比之下，Sarathi-Serve 生成的调度可以消除生成停顿，同时提供高吞吐量。也即做了较好的吞吐量和延迟的权衡 (Taming Throughput-Latency Trade-off) 



综上来看：分块预填充很好理解，正如上文所说。无停顿流水线则是通过严格的分块设计，以较小的TBT时间增加的代价，较大的提高吞吐量。

##### 2）文章总评

这篇文章以于调度算法为主，没有明确提出系统框架和调度器。但文章做了大量的实验 ( 对比试验，消融实验，在Mistral-7B on one A100s with token budget of 256， LLaMA2-70B on four A100s with token budget of 512 等等模型 )，与前面的研究vLLM, ORCA, fasterTransformer都做了比较详细的分析，是文章的一大特色。





# 第三篇系统 SplitWise

2024 ISCA Splitwise: Efficient generative llm inference using phase splitting

##### 1)  作者认为他们的主要贡献

* 使用**生产阶段的Trace**，对Nvidia A100 和 H100 上LLM推理 **预填充和解码阶段的执行效率和利用率之间的差异**进行了 广泛的描述
* 提出了SplitWise，把两阶段分离在不同的机器上，以提高硬件利用率
* 用SplitWise做了同构和异构的集群部署探索，来优化总体成本、请求吞吐量、功率。
* 使用**生产阶段的Trace**对使用SplitWise设计的系统进行性能评估。

**我的关注点**——如何传输KV缓存：

PD分离阶段需要把即时计算生成的KVcache (缓存上下文) 低延迟地从 预填充机器传到解码机器

为了实现低延迟的KV cache上下文传输。作者使用了 **后端Infiniband 互连** 这样一个最优的方式 从而在察觉不到性能损失的情况下提高效率

后端InfiniBand互联 是一种高性能计算（HPC）、数据中心和分布式存储系统中常用的低延迟、高带宽网络技术，主要用于服务器、存储设备或计算节点之间的高速通信

在这篇文章中InfiniBand连接GPU集群 ，这些 AI 集群中的每台机器通常由 8 个旗舰 NVIDIA GPU（A100 或 H100）组成。每个 GPU 通过高带宽 Mellanox InfiniBand 互连 [10]、[13] 与集群中的所有其他 GPU 连接，形成一个高带宽数据平面网络。目前，云端提供的 InfiniBand 带宽为每对 GPU 25 至 50 GBps [7]、[10]。

##### 2）性能分析

* 指标分析

**Performance metrics for LLMs**

| **Metric**                                                   | **Importance to User**                          |
| :----------------------------------------------------------- | :---------------------------------------------- |
| End-to-end (E2E) latency                                     | Total query time that the user sees             |
| Time to first token (TTFT)                                   | How quickly the user sees the initial response  |
| Time between tokens (TBT) /Time Per Output Token (TPOT 在DistServe中) | Average token streaming latency                 |
| Throughput                                                   | Requests per second (RPS) the system can handle |

上面四个指标是LLM推理系统的优化方向，不同的任务有不同的SLO级别，如**摘要生成任务的Throughput**比TTFT和TBT更重要。

在**上下文对话中，TTFT和TBT要更重要**，SLO要求也更严格。这一点也是**MoonCake中被用作系统设计的标准**之一，如果调度器在调度一个请求前，就预测出该请求的**SLO无法被满足，则直接拒绝该请求，腾出更多资源以满足当前执行的请求的SLO**

* 批处理分析

![image-20250520110536677](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250520110536677.png)

作者分析了三种批处理方式

第一种方式是**请求级别调度的方式(静态/动态批处理)——以请求为单位调度**，就是之前提到的**FasterFransformer**，**以解码优先**，新来的**请求必须等待当前批次的请求完成才能被处理**。对于某些请求如(1,2)延迟较少，但**总体具有较高的端到端延迟**，**以及较低的吞吐量**

第二种方式是**连续批处理——以每次前向传递为单位调度 (迭代级调度)**，就是前面提到的 **ORCA 和 VLLM**，它**以预填充优先**，完成预填充**等待解码的请求可以被抢占**，把资源给到新到来需要预填充的请求。每个前向传递轮次只有预填充请求或者解码请求 （但ORCA里面一些前向传递轮次也有极少量的预填充和解码混合，但vLLM则没有这种混合）

第三中方式是**混合批处理——预填充分块和无停顿流水线技术 ，即Sarathi-Serve**，从图中也能看出提高了吞吐量，显著降低了流水线中的气泡，但还有有问题：轻微的提高了解码阶段的延迟，TBT时间，虽然在Sarathi-Serve中使用严格的预填充分块限制了TBT延迟的提高，但是SplitWise作者认为这部分提高的延迟还是存在，没有被消除。



> 我们使用 2023 年 11 月 11 日从两个 Azure LLM 推理服务获取的生产轨迹。这些轨迹代表了当今 LLM 推理中最常见的场景：编码和对话。我们在 https://github.com/Azure/AzurePublicDataset [4] 上发布了部分轨迹。



>洞察一：不同的推理服务可能具有截然不同的提示和令牌分布。
>
>洞察二：混合连续批处理大部分时间都花费在极少数活跃令牌的批处理上。
>
>洞察三：对于大多数请求，端到端 (E2E) 的大部分时间都花在令牌生成阶段。
>
>洞察四：提示阶段的批处理大小应受到限制，以确保良好的性能。相反，对令牌生成阶段进行批处理可以实现高吞吐量且不会产生任何负面影响。
>
>洞察五：提示阶段的批处理受计算能力限制，而令牌阶段受内存容量限制。
>
>洞察六：提示阶段可以高效利用 GPU 的功耗预算，而令牌阶段则不能。
>
>洞察七：令牌生成可以在计算能力较弱的硬件上运行，以获得更好的性能/功耗比 (Perf/W) 和性能/成本比 (Perf/$)。

### 3）系统架构

![image-20250520121034266](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250520121034266.png)

三个池：预填充池、解码池、混合池

预填充池里面是专责预填充的机器

解码池里面是专责解码的机器，使用连续批处理

混合池里面负责整个推理流程，包括预填充和解码，使用混合批处理

#### 4）CLS调度：

两个调度：集群级别的调度(CLS)、机器级别的调度(MLS)

CLS 负责两件事情：Machine pool management. 和 Request routing 负责机器调度和请求路由分配

1、机器池管理Machine pool management

* 静态预先分配

  机器池管理Machine pool management通过工作负载和输入/输出的token分布，来预先地分配机器到不同的pool里，具体如下：

**SplitWise会保存请求负载和输入/输出token分布**，这些**负载和token分布来自在生产过程中的Traces**，作者做了大量的实验来确定生产过程的概率分布

例如**摘要的生成**任务的概率分布：prompt的**输入token非常多 (一篇论文**) 而**输出token非常少 (摘要**长度相比论文)；

**对话任务**/编码任务：用户**输入很少的prompt**——输入token，需要**较多的输出token**——长文本回复。

通过预先估计摘要任务和对话任务所占的比例 (也即工作负载) 和输入输出token的分布情况来 管理和分配机器池

**例如摘要任务多，输入token多，则安排更多的机器到prompt pool，如果对话任务多，即少输入token，多输出token，则安排更多的机器到token pool。**

* 动态运行分配

特殊情况①：如果实际的工作负载与预先分配不匹配，例如分配了大量的prompt机器，token机器非常少，但实际的负载来了大量的对话任务，需要大量的token机器，就会出现prompt机器用不完，token机器不够用的情况。

**具体来说：**

每个机器维护各自的**待处理的请求队列**，请求队列有一个阈值超过**这个阈值则不分配机器**，CLS**以最短待处理队列优先**为请求分配机器 (这个过程叫做路由请求)。

例如一个请求到来，CLS会同时给这个请求分配一个prompt machine和一个token machine (一对机器)，这样做的目的是KV缓存的传输和prompt的计算可以并行，以减少系统延迟。

CLS找到prompt pool 从里面  最短的待处理队列的机器，判断其是否超过阈值，没超过就把这个机器分给这个请求，同时把该请求插入这个机器的待处理队列，token pool同理。

如果出现了特殊情况①，prompt machine分配完成，但是在token pool里面发现最短待处理请求队列对应的机器也超过阈值，则从prompt里面找一台机器分配给该请求完成token生成(解码)，此时把这台prompt机器移入混合池，把这个请求插入混合池请求队列，并按混合批处理方式(Sarathi-serve)处理请求，等混合请求队列处理完成，再把混合池中的机器移入原始池。

#### 5）MLS调度：

MLS调度管理每个机器内部的请求。

![image-20250520125613855](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250520125613855.png)

Prompt machine：FCFS策略，根据图6a所示，当Batch_size 到达 2048时吞吐量最大，故MLS设置limit_BathSize为2048 (这个值可以根据硬件和机器更改) 当超过这个值，就把新到来的prompt任务放入待处理队列中

token machine: 采用FCFS策略，如图6b所示，吞吐量随着BatchSize增加，因此MLS会监视机器内存，当内存接近耗尽时把新到来的token任务放入待处理队列

Mixed machines：为满足SLO和TTFT，MLS优先处理prompt阶段，并把新到来的请求立即调度待处理队列。如果machine正在解码阶段，且没有资源来运行新请求的prompt任务，则MLS将抢占正在运行的token任务，即保持以prompt优先的策略。同时为了避免token任务饥饿，随时间提高token任务的优先级并限制每个请求的抢占次数。

#### 6）KV cache传输

KV cache需要从prompt machine 传到 token machine，这个传输延迟是splitWise的主要开销。

![image-20250520130949860](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250520130949860.png)

图11 a展示了传统的序列传输 **prompt全部完成->传输KV缓存->执行token**，这增加了TBT时间和E2E延迟。

图11 b展示了SplitWise的KV传输过程，不需要prompt全部完成，prompt一层完成 ->传输一层的KV cache，同时进行下一次的计算

这种异步传输的好处是减少了传输的时间开销，token machine可以更早的工作，prompt machine也可以更早的释放传输过的KV cache。

但这样引入了prompt 每层 KV cache传输 和本身前向计算的同步开销，因此SplitWise对于非常小的输入prompt，直接使用 a)中的序列传输，对于大的输入prompt使用分层传输。



#### 7）文章总评：

文章总体写的比较好，对CLS级别调度和MLS级别调度描述都非常清楚，作为一片系统级别的文章，系统结构也非常清楚。









# 第四篇系统：DistServe

2024 OSDI {DistServe}: Disaggregating prefill and decoding for goodput-optimized large language model serving

>Batching. Current serving systems **[9, 32, 54]** utilize a batching
>technique known as continuous batching. This method
>batches the prefill of new requests with the decoding of ongoing
>ones. It boosts the GPU utilization and maximizes
>the overall system throughput – tokens generated per second  across all users and requests. However, as mentioned in §1
>and elaborated later in §2.3, this approach leads to trade-offs
>between TTFT and TPOT. An advanced variant of continuous
>**batching [9] attempts to balance TTFT and TPOT by**
>**segmenting long prefill into chunks and attaching decoding**
>**jobs with a chunked prefill – but essentially, it trades TTFT**
>**for TPOT and cannot eliminate the interference (§2.3). In**
>**summary, batching prefill and decoding invariably leads to**
>**compromises in either TTFT or TPOT.**

9是sarathi分块预填充、32是vllm、54是orca。作者认为：这里面提到sarathi通过分块预填充把预填充块搭载在解码任务中，虽然提高了吞吐量降低了延迟，但本质上还是拿TTFT换TBT，没有消除TTFT对TBT的干扰，仍然是做了TTFT和TPOT妥协 (trade-off)

#### 1）弊端/挑战

**分块预填充的弊端：**

1、分块太大，导致**mixed batch和捎带难度增大**。分块太小又会**加剧**对解码任务的**抢占和竞争**，以致于浪费GPU资源。

2、此外，分块预填充会显著**增加预填充阶段的访存任务量**，**分块预填充的访存消耗是O(N^2)**，**普通预填充任务访存是O(N)**  

这是因为生成第n个 prefill-chuncked  KVcahce 需要从HBM 加载前n-1个块的KV cache。因此第一个自己加载一次，被其他块加载N-1次，第2个块自己加载1次，被其他快加载N-2次……

总的消耗量就是 N-1 +1  +N-2+1 +……=N+N-1+N-1+……+1=**N(N+1)/2** 

**混合批次调度的弊端**：prefill 阶段的优先级高于解码，必然会影响解码的效率

#### 2）并行策略的探索

对于预填充实例的并行策略探索：同一512token长度的prompt，**在较低的请求速率到来时，使用张量并行效果更好；当请求以较高速率到来且需要排队时，流水线并行效果更好。**

预填充示例并行策略探索：对解码任务的内存制约使用模型并行来扩展解码实例，或者利用LLM KVcache管理技术(vLLM中的分页注意力机制和GQA) 来提高解码批次大小，进而将解码任务扩展至接近计算密集型。
**当 TPOT SLO 严格时，张量并行并行对于降低 TPOT 并满足延迟目标至关重要。除此之外，流水线并行对于线性提升吞吐量更为有利。**

当模型能装入单个GPU显存时，复制多个模型权重副本 (张量并行的另外一种操作方式) 可以提高吞吐量和减少排队延迟，代价是权重冗余存储。

#### 3）目标

##### 1.布局算法

作者提出了两个算法来找到一个能够最大化每个 GPU 有效吞吐量的布局

Placement for High Node-Affinity Cluster

这种算法的前提条件是实例之间的通信通过Nvidia link (节点内) 或者 infiniband ( 节点之间) 即 带宽非常高，KV cache的通信不受限制，因此只要找到单独保证 prefill阶段最优，decode最优的布局，然后复制这种布局。

Placement for Low Node-Affinity Cluster

这种算法是节点内使用 Nvidia link通信，但节点之间通信受限，作者提出尽量把模型完整放在节点内部，能放下就问题解决，放不下分层放。

##### 2.在线调度

DistServe 采用简单的 FCFS 调度策略。所有**传入请求**均到达**集中控制器**，然后被调度到**队列最短的预填充实例**进行预填充处理，之后再调度到**负载最低的解码实例**执行解码步骤。(全文就这一个非常简单结构图，其他全是各种数据图标)

![image-20250520162857567](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250520162857567.png)

作者说自己的调度策略很简单，但自己用了很多的track技巧来提高效果

技巧1 **Reducing pipeline bubbles**。作者说自己做布局的时候，假定的prompt都是统一的512长度，但实际prompt长度不一，因此会有很多流水线气泡。

处理方法：作者说设置一个最低的prompt长度限制$L_m$，如果一个请求长度低于$L_m$ 则多找几个低于$L_m$ 的请求做批处理。在解码中直接将$L_m$ 设置为最大批处理大小，也即解码阶段每次都做最大批处理。

技巧2 **Combat busrtiness.** 作者说对于突然到来的大量工作负载的情况，导致大量的KV缓存从预填充实例传送到解码实例，解码实例来不及接收就会导致内存过载。

处理方法：KV缓存生成后不立即传送，而是保存在预填充实例的GPU中，使用预填充实例的GPU作为排队缓冲区，再由解码实例以“pull”的方式从预填充实例拉取KV cache。(因为解码实例经常内存不够用，预填充实例计算密集型，内存够用)

技巧3 **Replaning.** 作者提出的并行计划和预填充解码实例的规划布局是针对一定的工作负载优化产生的。因此当工作负载发生变化 (输入数据分布发生变化) DistServe将根据历史数据重新运行布局算法，这个算法非常快几秒钟就可以了。

技巧4 **Preemption and fault tolerance.** 作者说他没实现抢占和容错，但觉得非常有用以后会做。

#### 4）系统

DistServe 是一个端到端的分布式 LLM 服务系统，它包含一个布局算法模块、一个 RESTful API 前端、一个编排层和一个并行执行引擎。

可惜作者没有给出系统架构图。

#### 5）文章总评

DistServe 在P/D分离领域名号特别响，它探索了针对两阶段不同的调度策略，并做了很多实验来调研前人工作的不足，以及支撑自己的实验结果。但作为一篇系统文章很多地方没有写到甚是可惜，文章连一张比较完整的系统结构图也没有，有些地方实现的也不完全比较可惜。













# 第五篇系统 MoonCake

FAST 2025最佳论文：Mooncake: Trading More Storage for Less Computation—A {KVCache-centric} Architecture for Serving {LLM} Chatbot

### 1) 整体架构：KVCache 为中心的分离式架构

Mooncake架构创新地提出了基于KVCache中心的分布式推理框架，将预填充和解码阶段部署在两个独立集群，并利用RDMA构建跨节点共享缓存，突破单机缓存瓶颈。

Conductor 调度程序使用离线训练的回归模型根据缓存命中长度和当前队列状态估计预填充时间。对每个请求，全局调度器（Conductor）会选择一对预填充和解码节点实例，按照以下步骤调度请求：1）将尽可能多的可重用 KVCache 转移到选定的预填充实例；2）分块/层完成预填充阶段，并将输出 KVCache 持续流式传输到相应的解码实例；3）加载 KVCache 并将请求添加到解码实例的连续批处理过程中，以生成请求输出。

![image-20250520194701006](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250520194701006.png)

图：Mooncake架构。



### 2. Overview

该设计围绕将 LLM 推理过程分解为两个不同的阶段（预填充和解码）展开，每个阶段都在独立但相互关联的资源池上运行。关键元素包括全局 KVCache（MOONCAKE Store），它聚合了未充分利用的 CPU、DRAM、SSD 和 NIC 资源以进行缓存。

MOONCAKE 的核心创新是 KVCache 为中心的分离式架构，该架构将预填充和解码节点分开，以便在请求之间重复使用缓存的K/V向量。全局调度程序 ([Conductor](https://zhida.zhihu.com/search?content_id=254597163&content_type=Article&match_order=1&q=Conductor&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDc5MDQ3NzIsInEiOiJDb25kdWN0b3IiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTQ1OTcxNjMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.PB1L8YyV6Zeetls6ODj9T1TJzfTh--Vz5JtArRI9Dbk&zhida_source=entity)) 利用预测的执行时间和缓存命中率来平衡负载并最大限度地减少延迟。该设计还集成了高性能、拓扑感知传输引擎和分块管道并行方法，以高效处理长上下文输入。

![image-20250520194727880](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250520194727880.png)

图：Mooncake推理工作流。

MOONCAKE Store管理KVCache的存储和传输。如上图所示，在完成分词后，Conductor选择一对预填充节点和一个解码节点，开始推理工作流程。

**1) KVCache Reuse:** 预填充节点接收包含原始输入和可重用前缀缓存块键的请求，加载前缀缓存以初始化请求，如果没有前缀缓存，则跳过此步骤。

**2) Incremental Prefill:** 接着，预填充节点利用前缀缓存完成预填充，若未缓存的输入令牌数量超过阈值，则将预填充阶段拆分成多个部分并流水执行。

**3) KVCache Transfer:** 同时，MOONCAKE Store在各节点中管理并传输这些缓存，异步执行与增量预填充重叠，将生成的KVCache流传至解码节点的CPU内存，减少等待时间。

**4) Decoding:** 最后，在解码节点的CPU内存接收到所有KVCache后，请求加入下一个批次，Conductor根据当前负载预选解码节点，确保不违反TBT SLO。

### 3) CPP：分块流水线并行

MOONCAKE将预填充集群中的每X个节点分组为一个流水线预填充节点组。对每个请求，其输入tokens被划分为多个块（块的个数上限由prefill_chunk参数决定）。来自同一请求的不同块可以由不同节点同时处理，从而实现处理并行化，减少TTFT。

CPP有两个主要优点：1）与训练中的流水线并行性类似，它仅在每个流水线阶段的边界处需要跨节点通信，可以与计算轻松重叠。这提高了MFU，减少了KVCache传输的网络资源竞争。2）它自然适应短和长上下文，对于短上下文预填充没有显著的开销，并避免频繁动态调整节点分配。	

### 4) KV cache 感知的预填充调度

之前的研究在选择预填充实例的时候通常考虑负载均衡，因此考虑请求的数据分布，根据最短待处理队列优先或最短负载优先的策略调度预填充实例。

在 MOONCAKE 中，预填充实例的选择不仅考虑负载，还考虑前缀缓存命中长度和可重用 KVCache 块的分布。作者提出了提出了一种缓存感知的全局调度算法，该算法同时考虑了前缀缓存引起的预填充时间和本地排队时间。

即TTFT= 请求队列的排队时间 (考虑负载均衡)  + prefix reuse后的预填充计算时间 (考虑 KVcache prefix重用)。

为最小化TTFT，需要同时考虑上面两个因素。

**prefix reuse后的预填充计算时间**：conductor根据请求长度和prefix_len （计算每个预填充实例的orefix_len），使用拟合离线数据的多项式回归模型，估算相应的执行时间。

把请求分配给计算得到的最小化TTFT对应的预填充实例，如果TTFT仍然无法达到SLO，Conductor 会直接向上层返回 HTTP 429 Too Many Requests 响应状态代码，以拒绝该请求， 减轻整个集群的负载。

### 5) KV cache 负载均衡调度

在 MOONCAKE 中，每个预填充实例都有自己的一组本地前缀缓存。这些缓存的使用频率差异很大。例如，几乎每个请求都会访问系统提示，而存储本地长文档内容的缓存可能只有一个用户使用。如 §4.1 所述，Conductor 在实现缓存匹配和实例负载之间的最佳平衡方面至关重要。

作者提出了一种基于启发式算法的自动热点迁移方案，以增强缓存负载均衡。

如果全局感知后，Prefill阶段可重用的cache比较多，但是本地比较少，需要从其他地方进行全局调度：此时 预估的额外预填充时间 (increment生成时间) 短于传输时间，Conductor 会将缓存的位置和请求转发到该实例。该实例会主动从持有者处检索 KVCache 并将其存储在本地。不仅可以减少请求的预填充时间，还可以促进热点缓存的自动复制，从而允许它们在多个实例之间进行更广泛的分布。

