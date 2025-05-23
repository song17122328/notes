#### 重点章节

- 第二章、基础介绍
- 第五章、模型架构
- 第六章、模型预训练（重点关注6.3中的训练优化技术）
- 第九章、解码与部署
- 第十一章、规划与智能体



**知识考点：**

LLM : large language model 大语言模型

预训练-微调任务范式中：

* 预训练阶段是通过大规模无标注文本建立模型的基础能力
* 微调阶段是使用现有标注数据对于模型进行特定任务适配，从而更好解决下游的自然语言处理任务



#### 前言

2017年，OpenAI初步认为基于注意力机制的Transformer模型是大模型的理想架构，并基于此开始构建GPT模型。

2018年推出第一代GPT模型-GPT-1，GPT2和GPT3随后通过扩大训练数据和模型参数规模显著提升了模型性能。

在GPT-3基础上，通过代码训练，人类对齐，工具使用等技术不断提升模型性能，并推出了强大的GPT3.5

2022年11月，chatGPT正式上线。2023年3月，GPT-4模型上线，并进而拓展至拥有多模态的GPT-4V模型。

GPT发展历程的两点启示：

* 可拓展的训练架构与学习范式：transformer能够拓展数万亿参数规模，并将预训练任务统一为“**预测下一个单词**”
* 重视数据质量和数据规模：高质量的超大规模数据是大模型语言的关键

学术界面临的困境：

* 超大算力需求使得学术界难以直接实验，开展相关研究。
* 参数众多，组件复杂，训练细节无法从论文中获取导致难以完成完整的大规模训练实验。

结果：工业界走在学界之前

转机：更多“开放”使得大模型技术愈发“透明化”，研究人员能较为顺利地完成大模型的整体训练，并取得不错模型效果。



#### chapter 1 引言

##### 1.1 四个发展阶段：

* **统计语言模型**(statistical language model, SLM) 基于统计学习方法，使用马尔可夫假设(Markov Assumption)建立语言序列的预测模型——根据固定长度前缀预测下一个单词。其中固定长度为n的模型被称为n元(n-gram)语言模型，常见，二元或三元模型，但在模型中，估计转移概率项数随n呈现指数级增长，因此带来“维度灾难”(Curse of Dimensionality)。 回退估计法(Back-off Estimation)和古德-图灵估计(Good-Turing Estimation)是语言模型平滑策略，可以缓解数据稀疏的问题，但对高阶上下文的刻画能力较弱，无法精确建模复杂的高阶语义关系。
* **神经语言模型**(Neural Language Model,NLM) 使用神经网络模型来建模文本序列生成，如循环神经网络(Recurrent Neural Networks,RNN)。Yoshua Bengio在早期引入了分布式词表示(Distributed Word Representation) 并构建基于聚合上下文特征的目标词预测函数。
  * 分布式词表示使用低维稠密向量来表示词汇含义，与基于词典空间的稀疏词典向量(one-hot)本质不同，分布词向量又称“词嵌入”(Word Embedding)。基于隐含语义特征表示的语言建模方法为自然语言处理任务提供了一种较为通用的解决途径。word2vec是一个具有代表性的词嵌入学习模型，构建了一个简化的浅层神经网络来学习分布式词表示，所学习到的词嵌入(Word Embedding)可以作为后续任务的语义特征提取器，受到广泛使用，性能显著提升。
* **预训练语言模型**(Pre-trained Language Model,PLM)相比于早期词嵌入模型，其在训练结构与训练数据两方面进行了改进和创新。ELMo(Embeddings from Language Models)是一个早期的代表预训练模型，使用大量无标注数据训练双向LSTM(Bidirectional LSTM,biLSTM)网络，预训练后得到的biLSTM可以用来学习上下文感知的单词表示，与word2vec学习固定的词显著不同。ELMo还可以根据下游任务对biLSTM进行微调(Fine-Tuning)从而实现面向特定任务的模型优化。
  * 传统序列神经网络长文本建模能力较弱，且不容易并行训练，限制了早期预训练模型(ELMo)的性能。2017年后，谷歌提出了基于自注意力机制(Selft-Attention)的Transformer模型，通过自注意力机制建模长程序列关系。
  * Transformer的一个优势：对硬件非常友好，可通过GPU或者TPU加速训练，为研发大语言模型提供了并行优化的神经网络架构。基于Transformer谷歌进一步提出了预训练模型BERT，采用了仅有编码器的Transformer架构，通过在大规模无标注数据上使用专门设计的预训练任务来学习双向语言模型。
  * 同时，OpenAI也迅速采纳了Transformer架构，将其用于GPT-1的训练，不同于bert，GPT-1采用了仅有解码器的Transformer架构及基于下一个词元预测的预训练任务进行模型训练。（通常：解码器架构更适合解决自然语言生成任务(如文本摘要)，编码器架构更适合解决自然语言理解任务(如完形填空)）
* **大语言模型(Large Language Model,LLM)** 扩展法则：通过规模扩展如增加模型参数规模或数据规模带来下游任务模型性能提升。涌现能力：通过上下文学习(In-Context Learning, ICL )的方法利用少数样本数据解决下游任务。这种大模型具有但小模型不具有的能力被称为“涌现能力”，涌现能力是区别大语言模型的标志。



##### 1.2 大语言模型的能力特点

* **具有较为丰富的世界知识**。与早期专家系统依靠逻辑和规则相比，其可以充分建模利用世界知识信息。与早期预训练模型相比，其数据量庞大可以学习到海量的世界知识。

* **具有较强的通用任务解决能力**。 大模型主要通过预测下一个词元的预训练任务进行学习，依靠较强的通用任务求解能力，大模型的提示学习方法可以代替他特定任务的解决方案。
* **具有较好的复杂任务推理能力**。 通常情况下能展现出比传统模型更强的综合推理能力。
* **具有较强的人类指令遵循能力**。任务输入与结果输出均通过自然语言进行表达。
* **具有较好的人类对齐能力**。大语言模型出色的模型性能引发人们对模型对齐与监管的重视，目前广泛采用对齐方式是基于人类反馈的强化学习技术，加强正确行为，规避错误行为，进而建立较好的人类对齐能力。
* **具有可拓展的工具使用能力**。大语言模型可借用搜索引擎和计算器等多种工具，提高模型任务解决能力。

其他能力：长程对话的语义一致性、新任务的快速适配、对于人类行为的准确模拟等



##### 1.3 大语言模型关键技术概览

* **规模扩展**。早期OpenAI从参数、数据、算力研究了规模扩展对模型性能带来的影响，并进行了一系列实践探索，发现大规模语言模型具有上下文学习能力，思维链能力等小语言模型不具备的能力。近期的工作则聚焦于高质量数据的规模扩展。实现规模扩展的关键在于模型架构的可扩展性，Transformer模型可扩展性非常强，硬件并行优化支持友好，特别适合大语言模型的研发。
* **数据工程**。技术进步简化了训练方法的复杂性，进一步凸显出优质数据的重要性。数据工程包括三个方面
  * 首先，需要**全面收集数据**，拓宽高质量数据来源；
  * 其次，需要**精细的数据清洗**，提高数据质量；
  * 再次，需要设计**有效的数据配比与数据课程**，加强模型对于数据语义信息的利用效率。

* **[高效预训练]**。训练出一个性能强大的大语言模型极具挑战性。参数规模巨大因此需要大规模分布式训练算法优化大语言模型的神经网络参数。
  * 训练时候需要用到各种**并行策略和效率优化方法**（进一步研究重点），包括**3D并行**(数据并行、流水线并行、张量并行)、**ZeRo**(内存冗余消除技术)等。 3D并行和ZeRo是技术重点。
  * 为支持分布式训练，许多研究机构发布了专用的**分布式框架来简化并行算法的实现与部署**，有DeepSpeed和Megatron-LM，能有效支持千卡甚至万卡的联合训练。
  * 具体实现上，需要搭建**全栈式的优化体系架构**，支持大规模预训练数据的**调度安排**，建立可迭代的模**型性能改进闭环**，强化效果**反馈机制**，以便能快速灵活地进行相关训练策略的调整
  * 需要开展基于小模型的沙盒测试实验，以确定大模型的最终训练策略。
* **能力激发** 。指令微调以及提示策略进行激发或诱导。
  * 指令微调：使用自然语言表达的任务描述以及期望的任务输出对于大语言模型进行指令微调。现有研究认为指令微调无法向大模型注入新知识，而是训练大模型利用已有知识进行任务求解的能力。
  * 提示学习：涉及多种高级提示策略包括上下文学习、思维链提示等，构建提示模板或表述形式来提升大语言模型对复杂任务的求解能力。提示工程已经成为利用大语言模型的重要技术途径。
  * 大语言模型还具备较好的规划能力，能对复杂任务逐步求解，降低单一步骤直接求解的难度。
* **人类对齐**。3H对齐标准(Helpfulness、Honesty、Harmlessness),OpenAI提出了基于人类反馈的强化学习算法(Reinforcement Learning from Human Feedback, RLHF)，将人类偏好引入到大模型的对齐过程中
* **工具使用**。



##### 1.4 大语言模型对于科技发展的影响

大语言模型采用与小型预训练语言模型相似的网络架构及训练方法，通过扩展模型参数、数据数量、算力资源获得了意料之外的性能跃升，首次实现了单一模型有效解决众多复杂任务。

对于科研范式的重要影响：研究人员需要深入了解大模型相关的工程技术，对于理论与实践结合提出了更高的要求。例如训练大模型具备大规模数据处理与分布式并行计算方面的实践经验。

大语言模型改变人类开发和使用人工智能的方式，使用者需要了解大语言模型的工作原理，按照其能够遵循的方式来描述需要解决的任务。

对产业应用带来技术变革，催生基于大语言模型的应用生态系统，如微软(Microsoft 365)利用大模型加强自动化办公。