# Q38 LC3 MoE Model Card 填写稿

队伍编号：SQ2026AAA012562

模型系列：Q38 LC3 MoE

任务：自回归语言建模

## 模型简介

Q38 LC3 MoE是一组面向固定训练计算预算的混合语言模型。模型以Qwen3.5文本模型的开源实现为基础，保留Gated DeltaNet与因果自注意力构成的混合主干，结合稀疏专家、条件记忆、深度加权平均和矩阵值长因果卷积，分别承担内容相关交互、条件特征变换、局部模式检索、跨层表示复用与按相对距离共享的序列滤波。

本模型系列按C1至C4四个计算预算档位配置宽度、深度和训练token数，使用比赛训练数据从随机初始化训练。Q38为团队内部模型名称，代码基础及技术来源见本文末尾。

## 架构设计与主要改进

### 内容相关交互与距离共享滤波的结合

主干按照三层Gated DeltaNet配一层完整因果自注意力的方式组织。Gated DeltaNet通过门控状态更新处理序列信息，自注意力根据输入内容选择交互位置。在此基础上，我们在三个深度位置加入独立的长因果卷积残差分支，将按相对距离共享的滤波规则引入混合主干。

长卷积位于所在层的Attention或Gated DeltaNet残差更新之后、MoE前馈变换之前。每个分支先将输入投影为内容与门控两部分，通过长度2048的矩阵值因果滤波器聚合历史信息，再经门控和输出投影加回残差。每个相对距离对应一个可学习的完整通道矩阵，使模块同时具有跨位置传播与跨通道组合能力。

令输入投影得到的行向量为\(u_t,g_t\in\mathbb R^c\)，该分支的核心运算为：

\[
z_t=\sum_{j=0}^{\min(K-1,t)}s_j u_{t-j}P_j,\qquad
x'_t=x_t+\bigl(z_t\odot\operatorname{SiLU}(g_t)\bigr)W_{out},
\]

其中\(K=2048\)，\(P_j\in\mathbb R^{c\times c}\)，\(s_j=[(j+1)\sum_{k=1}^{K}1/k]^{-1/2}\)。同一组距离权重在不同序列位置之间共享；随距离衰减的尺度提供平滑的初始化偏置，输入相关门控调节滤波结果的注入强度。输出投影从零初始化，使新增分支能够从保持主干输出的状态开始学习。

这一设计将内容相关的序列交互与距离共享的结构性滤波结合，为反复出现的间隔关系、局部重复和跨位置特征组合提供建模途径。长卷积设计参考SGConv、Hyena等长序列建模研究，FFT实现参考高效长卷积研究。[5–7] 本模型采用完整通道矩阵形式，并将其集成到稀疏专家混合主干中。

### 条件计算与条件记忆的分工

每个MoE层包含512个路由专家，单个token选择其中4个，同时使用共享专家。专家容量通过条件执行与单token计算量解耦，模型宽度和专家内部维度随计算预算调整。

为补充专家网络的非线性计算，模型在首层后加入Engram式条件记忆模块。该模块利用原始token ID构建哈希查表特征，并通过上下文门控和短卷积与主干表示融合。其设计参考条件记忆与可扩展查表机制。[2] 在本实现中，局部可检索模式由记忆分支表达，内容相关特征变换由专家网络承担，形成两种互补的信息处理路径。

### 负载均衡与跨层表示复用

路由器采用无辅助损失的负载均衡方式：根据专家负载更新选择偏置，以调整专家分配；专家输出的混合权重使用未加入该偏置的路由分数。该机制参考Auxiliary-Loss-Free Load Balancing，[3] 将负载调节与主任务损失的梯度优化分开，当前训练不加入路由均衡辅助损失。

模型同时采用DenseFormer的深度加权平均机制。[4] 每层对当前表示和先前层表示进行可学习的加权组合，为浅层与深层特征之间提供直接的信息传递路径。该机制与token维度上的attention、Gated DeltaNet和长卷积分别作用于不同维度，共同构成模型的信息流组织方式。

### 计算结构与执行实现协同

长卷积使用FFT执行因果滤波，在保持滤波权重参与预测的前提下，避免直接按全部lag展开长核计算。训练实现结合滤波响应复用、状态分片与梯度累积；推理实现结合FFT布局优化、低精度矩阵计算和算子融合，以控制长卷积及混合主干的执行开销。

本方案同时关注预测质量、绝对前向耗时和参数归一化效率。评测中的EvalTime按以下关系计算：

\[
\mathrm{EvalTime}=\frac{T\,10^{15}}{N_aD},
\]

其中\(T\)为累计前向时间，\(D\)为计分token数，\(N_a\)为平均非词表活跃参数数目。长卷积的算法实现改变了参数容量与实际执行工作量的关系，因此评测报告同时列出绝对时间与归一化指标，分别反映实际运行成本和该参数统计口径下的执行效率。

本方案在已有Gated DeltaNet、Engram、无辅助损失均衡和深度加权平均机制的基础上，完成与矩阵值长卷积的架构组合、预算配置和实现适配。贡献集中在这些组件在固定预算模型系列中的协同组织及其训练、推理实现。

## 模型系列配置

四档模型使用相同的组件组合，按预算调整宽度、深度和训练量。以下LC位置均采用从0开始的层索引，卷积通道数等于该档hidden维度。

| 档位 | 目标训练预算FLOPs | hidden | 层数 | 路由专家数 / top-k | LC位置 | 卷积核长度 |
|---|---:|---:|---:|---:|---|---:|
| C1 | 3×10¹⁷ | 512 | 10 | 512 / 4 | 3、6、8 | 2048 |
| C2 | 1×10¹⁸ | 640 | 12 | 512 / 4 | 4、8、10 | 2048 |
| C3 | 3×10¹⁸ | 768 | 14 | 512 / 4 | 4、9、12 | 2048 |
| C4 | 1×10¹⁹ | 768 | 16 | 512 / 4 | 5、10、14 | 2048 |

## 训练方法

模型采用统一tokenizer，词表大小为10112，训练序列长度为4096。训练数据按固定长度序列块打乱，在各rank之间划分；当预算需要跨轮读取时，按轮次重新打乱。各档训练token数依据实际结构的每token计算量重新核算，并预留0.5%的预算用于转换、校准等附加工作。

主任务为自回归下一token预测。优化器采用Muon与AdamW组合，并使用学习率预热和衰减调度。训练以BF16混合精度执行，结合分布式数据并行、优化器状态分片、FFN重计算及梯度桶复用。当前配置不启用MTP、mHC或学习式稀疏attention索引器。

## 推理与评测方法

本模型系列采用纯神经网络预测头，不接入外部统计回退头、检索缓存混合头或在线输出头适配。推理统一使用长度10240的序列设置，主干采用FP8计算路径，输出完整位置和完整词表的BF16 logits，并保留路由偏置与深度混合系数所需的FP32控制参数。

语言建模质量以BPB衡量。BPB由总负对数似然和对应的UTF-8文本字节数计算；四档模型的质量与实际训练计算量用于构建预算—性能曲线。推理报告记录硬件、精度、实际batch、计分token数、累计前向秒数、活跃参数统计和EvalTime。完整评测结果与所提交权重、配置和源码版本一一对应。

## 代码与技术来源

基础文本模型实现来自Transformers 5.19.0中的`transformers.models.qwen3_5_moe`，保留其Apache-2.0许可证和来源声明。团队初始Q38结构参考`huangzhenhua111/hzh-q38f-sparse`的提交`12862f6ded28d4cedf7de411d3a632d660ade140`；在该谱系上完成结构选择、长卷积分支、组合实验、预算核算以及训练和推理实现适配。实际采用的组件与配置以上述描述为准。

1. Yang et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. https://arxiv.org/abs/2412.06464
2. Cheng et al. *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*. https://arxiv.org/abs/2601.07372
3. Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*. https://arxiv.org/abs/2408.15664
4. Pagliardini et al. *DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging*. https://arxiv.org/abs/2402.02622
5. Li et al. *What Makes Convolutional Models Great on Long Sequence Modeling?* https://arxiv.org/abs/2210.09298
6. Poli et al. *Hyena Hierarchy: Towards Larger Convolutional Language Models*. https://arxiv.org/abs/2302.10866
7. Fu et al. *FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores*. https://arxiv.org/abs/2311.05908
