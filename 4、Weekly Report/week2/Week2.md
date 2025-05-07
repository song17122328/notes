摘要和关键字：请你帮我生成

1.  【论文阅读】

- DDistServe：Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

  - **核心问题**：通过分离预填充和解码两个阶段来分别提高TTFT和TPOT，保证在SLO条件下增加有效吞吐量。
  - **方法亮点**：分别配置预填充GPU组和解码GPU组，预填充GPU组计算密集型，使用张量并行提高效率；解码组使用流水线并行，适配IO密集型来提高效率。两组沟通交流使用NVLINK减少消耗
  - **我的疑问或提升的地方**：两格不同的GPU配置组之间的沟通交流需要昂贵的NVLINK，如果没有怎么办？需要怎么处理？

  详细内容见iCloud论坛论文分析



2. 【知识学习】

- 重点概念
  - Batch Normalization
    - Batch Normalization和weights decay 一样是用来缓解梯度爆炸和梯度消失的问题。其中，weights decay主要通过对权重w的L2范数进行限制，缩小权重搜索空间，起到正则化作用，而Batch Normalization 则是尝试让把每一层神经网络都拥有相似的分布（通过线性变化来调整均值和方差）多了要学习的参数，如alpha和beta。BN层一般都放在卷积层或者全连接层后，激活层之前。
  - ResNet网络可以支持神经网络进入很深，只要内存能容得下1000层也可以。因为ResNet网络直接将下面层的输入连接到本层输出，这样就导致本层如果拟合的足够好其梯度（也是误差）较小，可能导致梯度消失，这一问题不会影响到下面层。因为越靠近输出的层约容易训练完毕（上面层）梯度比较小，通过链式法则传递到下面导致下面层梯度消失。ResNet把下面层的输入直接加到上面层的输出，这样链式法则求导加法不会造成梯度消失的问题。
- 学了神经网络LeNet，AlexNet，VGG，NiN，GoogLeNet，ResNet等。



3.【当前问题和下一步计划】

1、基础知识不够，下一周力争结束李沐的《Dive into Deep Learning》，继续学习LLM BOOK，开始进行CSAPP和并行计算(stanford)的学习。

2、论文阅读难度大，批判性和创新性阅读效率不高，需要加强论文积累，对论文分析的工作要持之以恒，熟能生巧。

3、尝试复现一些实验，可以考虑开始撰写论文，或者参与一些纵向的课题实践。

