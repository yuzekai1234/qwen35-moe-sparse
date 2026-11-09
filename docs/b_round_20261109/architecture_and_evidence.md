# Q38 比赛架构与创新说明

当前方案以 Qwen3.5 的文本模型实现为基础，将条件计算、条件记忆、跨层特征复用和长程滤波结合，在固定训练预算下共同评估语言建模质量与推理成本。这里的 Q38 是团队内部版本名，不是阿里发布的 Qwen3.8。模型使用指定语料从随机初始化训练，不是继续训练 Qwen 的预训练权重。

## 架构和代码来源

基础实现来自 Transformers 5.19.0 的 `transformers.models.qwen3_5_moe`；原始文件哈希和 Apache-2.0 声明保存在 `runtime/q38/models/qwen3_8_moe/UPSTREAM.json`。团队的初始 Q38 结构参考 `huangzhenhua111/hzh-q38f-sparse` 的提交 `12862f6ded28d4cedf7de411d3a632d660ade140`。后续工作包含结构筛选、预算配置、长卷积训练与推理实现、组合实验和部署优化。个人贡献应按 Git 提交和实验记录署名，不从群聊称呼推断作者。

当前运行配置为 E512、top-4、3处 K2048 LongConv、Engram、loss-free balancing 和 DenseFormer。每四层采用三层 Gated DeltaNet、一层完整因果 attention。共享专家、RMSNorm、SwiGLU 和部分 RoPE 等基础组件继承自模型谱系，不能列为团队新发明。

| 组件 | 当前实现 | 目的和参考 | 可主张的贡献范围 |
|---|---|---|---|
| 混合 token mixer | 3层 GDN 配1层完整 attention | GDN 用门控和 delta rule 更新记忆；attention提供内容相关的显式交互。[1] | 沿用基础机制，团队选择适配比赛规模的配置 |
| 稀疏 MoE | 每层512个路由专家、top-4及共享专家；专家宽随 hidden 缩放 | 通过条件计算增加容量；E512 是当前容量、访存和吞吐之间的经验选择 | 结构配置与实验选择；不声称专家数越多必然更好 |
| Engram式记忆 | 原始 token ID 的哈希查表、上下文门控与短卷积；在首层后注入 | 将可查表的局部模式与神经计算分工，参考 conditional memory。[2] | 对原论文的任务化改造，不是完整复现；未采用原文所有 tokenizer 压缩和系统组件 |
| 无辅助损失均衡 | 专家选择分数加持久化偏置，按负载更新；混合权重使用未加偏置的分数；aux系数为0 | 用路由偏置调节负载，避免均衡辅助损失干扰主任务梯度。[3] | 实现与组合应用，原理归于引用工作 |
| DenseFormer | 学习当前层和之前层表示的加权组合 | 改善跨深度信息流和表示复用。[4] | DWA机制的集成与组合验证 |
| 矩阵值长因果卷积 | 三处 K2048，每个 lag 是完整通道矩阵；FFT执行；带输入投影、门控和残差 | 按相对距离共享滤波规则，补充内容相关交互；参考长卷积及混合模型研究。[5–7] | 当前具体组合、参数化和执行实现可以描述；尚不声称首次提出长卷积或独特插入位置 |
| 训练和推理工程 | Muon/AdamW、DDP梯度桶复用、优化器分片；FP8 trunk、完整BF16 logits、融合算子、FFT布局优化 | 在质量约束下减少计算、访存和同步开销 | 工程贡献分别报告数值对照和实测速度；组件加速不能直接当作整模加速 |

旧 README 中的 mHC、学习式 QSA/indexer、MTP 并未全部进入当前方案：当前 `mhc_enabled=0`、`qsa_mode=full`、`mtp=0`、`kl_coef=0`。这些组件可以写入探索历史，不能写成当前系统正在使用的创新。

## 长卷积的真实位置和数学形式

源码 `Qwen3_8MoeTextModel.trunk` 的顺序是：

```text
输入 → Attention/GDN残差更新 → 可选LongConv残差更新 → MoE/FFN残差更新
     →（首层后注入Engram）→ DWA → 下一层
```

因此当前实现是 **token mixer之后、MoE/FFN之前**，不是FFN后置。不能将另一份“FFN后生成特征再传播”的讨论直接作为当前实现的机制解释。

省略归一化细节，LongConv可写为：

\[
(u_t,g_t)=W_{in}\operatorname{Norm}(x_t),\quad
z_t=\sum_{j=0}^{\min(K-1,t)}s_j u_{t-j}P_j,\quad
x'_t=x_t+W_{out}(z_t\odot\operatorname{SiLU}(g_t)).
\]

这里 \(P_j\in\mathbb R^{c\times c}\)，\(s_j=[(j+1)\sum_{k=1}^K 1/k]^{-1/2}\)。本实现有跨通道混合，不是逐通道卷积；每处 lag 矩阵的参数量为 \(Kc^2\)。输出投影从零初始化，使新增残差分支从不改变主干输出的状态开始学习。因果边界和评测序列长度决定各 lag 实际参与的比例。

可采用的设计解释是：内容相关的 attention/GDN 与按相对距离共享的滤波规则形成互补；长卷积在专家非线性变换之前整合近期到较远的序列信息。距离共享、衰减初始化和门控为重复、间隔和局部结构提供归纳偏置。这是可检验的机制假设，不能直接写成模型已经学到了某种语义通道或在特定领域必然获益。

SGConv研究长卷积核的结构和衰减；Hyena研究长卷积与数据相关门控；FlashFFTConv研究FFT的硬件执行效率。[5–7] 它们提供方法背景，不证明当前完整通道矩阵、插入位置及预算配置的优越性。`Convolution for Large Language Models`研究短核深度卷积和17个插入位置，其最优结果主要在attention前的Q/K/V投影上，不能作为本实现K2048长卷积有效的直接证据。[8]

## 正面且可验证的效率表述

本方案探索“可参与预测的参数容量”与“实际执行工作量”的分离：长卷积的滤波权重真实参与输出，FFT避免逐lag展开执行全部通道矩阵乘。其效率动机包括改善比赛的参数归一化耗时指标，而不只是追求更短的绝对前向时间。

\[
\mathrm{EvalTime}=\frac{T\,10^{15}}{N_aD}.
\]

这里 \(T\) 是累计前向秒数，\(D\) 是计分token数，\(N_a\) 是按实际执行统计的平均非词表活跃参数。FFT可以改变实际运算与活跃参数之间的比例；因此较低EvalTime并不自动代表绝对秒数更短。报告必须同时列出 **BPB、T、D、Na、EvalTime、显存、硬件与精度**。不用闲置专家、训练专用分支或未执行权重扩大Na；评测计数应与实际前向匹配。

长卷积仍有频域跨通道乘法、滤波器变换、通信和优化器开销。不能套用逐通道长卷积的复杂度来描述本实现，也不能把“FFT算法更省运算”直接写成“整模更快”。

可用于创新说明的表述：

> 我们面向固定训练计算预算和参数归一化推理指标，构建融合条件专家、条件记忆、跨层表示复用与矩阵值长程滤波的混合语言模型。长卷积通过相对距离共享的因果滤波补充内容相关的序列交互，并采用FFT实现减少显式长核展开的执行成本。我们分别考察其对语言建模质量、绝对前向时间和参数归一化效率的影响，以完整预算、统一精度的多尺度实验检验设计收益。

## 已有实验结果的正确边界

| 实验 | 质量结果 | 耗时结果 | 可以得出的结论 |
|---|---|---|---|
| 原E1024、两处LC的combined C1 | raw BPB 1.00421054；hybrid 0.99090686 | 历史raw前向128.30秒；hybrid253.53秒，有其他作业干扰 | 混合预测有质量收益，耗时需独占复测；不是长卷积单项消融 |
| 历史C4 MoE与K2048 LC版本，S4096 | 0.90382160 → 0.90620114，增加0.00237954 | 此处不混入其他口径时间 | 这组结果未显示整体BPB改善；固定预算下训练token和容量变化也构成混杂因素 |
| 当前E512、三处LC、batch×1.5的C1 | raw BPB 1.01100120 | EvalTime 0.56475571，gpu1完整A记录 | 新候选已有完整端点；多个配置同时变化，不能称为任何单项结构的因果收益 |
| 当前同族C4 | 正在训练 | 训练吞吐约13.7–14.6万token/秒 | 是训练速度记录，不是推理EvalTime，也不是最终BPB |
| 配套C2/C3 | 配置与CPU核验阶段 | 完成时间需新机实测 | 四档曲线和最终Performance尚未齐全 |

## 提升样本集和消融安排

可以从逐样本配对结果中导出“LongConv提升样本集”，用于观察机制和展示案例，但这是按结果选择的事后集合。该集合上的平均提升本身不能证明泛化收益。

建议将文档按稳定ID或内容哈希预先分为发现集和独立验证集。仅在发现集查找改善样本，提出可计算的特征规则，例如长度、跨距离重复n-gram、标识符复现、列表/表格结构。规则在查看验证集效果前冻结，然后在全部验证样本及规则命中的验证子集中报告字节加权BPB、样本量和覆盖率，同时保留退化样本。不能把“按提升挑出的样本”再充当验证集。

统一输入格式为每行 `id, split, utf8_bytes, baseline_nll, longconv_nll, features`。两个NLL必须来自同一文本、计分token掩码、精度和上下文；BPB按 `sum(nll)/(ln(2)*sum(utf8_bytes))` 计算，不能简单平均样本BPB。现有汇总result.json不能反推出逐样本提升集，需要保存配对评测的逐文档NLL和字节数。

最有解释力的对照包括：无LC/短LC/K2048；相同插入数量下的位置对照；相同预算的LC容量与FFN容量对照；Engram、DWA、loss-free逐项去除。长短卷积比较还应区分等参数和等计算预算两种设定。预先固定主要指标、样本划分和选择规则，记录所有候选，而不是只保留获益的设置。

## 参考资料

1. Yang et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule*, 2024. https://arxiv.org/abs/2412.06464
2. Cheng et al. *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*, 2026. https://arxiv.org/abs/2601.07372
3. Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*, 2024. https://arxiv.org/abs/2408.15664
4. Pagliardini et al. *DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging*, 2024. https://arxiv.org/abs/2402.02622
5. Li et al. *What Makes Convolutional Models Great on Long Sequence Modeling?*, 2022. https://arxiv.org/abs/2210.09298
6. Poli et al. *Hyena Hierarchy: Towards Larger Convolutional Language Models*, 2023. https://arxiv.org/abs/2302.10866
7. Fu et al. *FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores*, 2023. https://arxiv.org/abs/2311.05908
8. Tian et al. *Convolution for Large Language Models*, 2026. https://arxiv.org/abs/2607.18413

## 复核入口

- 当前主干源码：`campaigns/b_round_20261109/runtime/q38/models/qwen3_8_moe/modeling_qwen3_8_moe.py`；SHA256 `6e22307c27b8e7b4273a1923c234b763d8599ba15b7ffd4bde4fb59274063acc`。
- 参考C4配置：`campaigns/b_round_20261109/provenance/c4-lc3-e512-b15.json`。
- 原C1证据：gpu1迁移包 `evidence/full-A-{raw,hybrid}-v1/`。
- 新C1结果：gpu1 `b-round-20261109/runs/combined-lc3-e512-b15-c1-w8-20261109/c1/evaluations/raw-s10240-auto/result.json`。
- 历史LC对照：共享日志 `20261104-nostalgiaKevin-longconv-training-convergence-readonly.md`。
- 2026-11-09另一对话《查找 FFN 后接长卷积先例》作为文献线索；实际插入位置以上述源码为准。


## 官方 Model Card 填写正文

以下正文面向官方评审，适用于当前E512、三处K2048长卷积、纯神经网络预测头的模型系列，可用于Model Card的模型简介、架构、创新设计、训练方法和技术来源栏目。它保留正式表述，不包含前文面向开发者的讨论旁白。另存的 `model_card_submission.md` 提供便于复制的独立版本。当前未取得最新版官方固定模板，栏目映射不冒充已完成官方格式校验。

队伍编号：SQ2026AAA012562

模型系列：Q38 LC3 MoE

任务：自回归语言建模

### 模型简介

Q38 LC3 MoE是一组面向固定训练计算预算的混合语言模型。模型以Qwen3.5文本模型的开源实现为基础，保留Gated DeltaNet与因果自注意力构成的混合主干，结合稀疏专家、条件记忆、深度加权平均和矩阵值长因果卷积，分别承担内容相关交互、条件特征变换、局部模式检索、跨层表示复用与按相对距离共享的序列滤波。

本模型系列按C1至C4四个计算预算档位配置宽度、深度和训练token数，使用比赛训练数据从随机初始化训练。Q38为团队内部模型名称，代码基础及技术来源见本文末尾。

### 架构设计与主要改进

#### 内容相关交互与距离共享滤波的结合

主干按照三层Gated DeltaNet配一层完整因果自注意力的方式组织。Gated DeltaNet通过门控状态更新处理序列信息，自注意力根据输入内容选择交互位置。在此基础上，我们在三个深度位置加入独立的长因果卷积残差分支，将按相对距离共享的滤波规则引入混合主干。

长卷积位于所在层的Attention或Gated DeltaNet残差更新之后、MoE前馈变换之前。每个分支先将输入投影为内容与门控两部分，通过长度2048的矩阵值因果滤波器聚合历史信息，再经门控和输出投影加回残差。每个相对距离对应一个可学习的完整通道矩阵，使模块同时具有跨位置传播与跨通道组合能力。

令输入投影得到的行向量为\(u_t,g_t\in\mathbb R^c\)，该分支的核心运算为：

\[
z_t=\sum_{j=0}^{\min(K-1,t)}s_j u_{t-j}P_j,\qquad
x'_t=x_t+\bigl(z_t\odot\operatorname{SiLU}(g_t)\bigr)W_{out},
\]

其中\(K=2048\)，\(P_j\in\mathbb R^{c\times c}\)，\(s_j=[(j+1)\sum_{k=1}^{K}1/k]^{-1/2}\)。同一组距离权重在不同序列位置之间共享；随距离衰减的尺度提供平滑的初始化偏置，输入相关门控调节滤波结果的注入强度。输出投影从零初始化，使新增分支能够从保持主干输出的状态开始学习。

这一设计将内容相关的序列交互与距离共享的结构性滤波结合，为反复出现的间隔关系、局部重复和跨位置特征组合提供建模途径。长卷积设计参考SGConv、Hyena等长序列建模研究，FFT实现参考高效长卷积研究。[5–7] 本模型采用完整通道矩阵形式，并将其集成到稀疏专家混合主干中。

#### 条件计算与条件记忆的分工

每个MoE层包含512个路由专家，单个token选择其中4个，同时使用共享专家。专家容量通过条件执行与单token计算量解耦，模型宽度和专家内部维度随计算预算调整。

为补充专家网络的非线性计算，模型在首层后加入Engram式条件记忆模块。该模块利用原始token ID构建哈希查表特征，并通过上下文门控和短卷积与主干表示融合。其设计参考条件记忆与可扩展查表机制。[2] 在本实现中，局部可检索模式由记忆分支表达，内容相关特征变换由专家网络承担，形成两种互补的信息处理路径。

#### 负载均衡与跨层表示复用

路由器采用无辅助损失的负载均衡方式：根据专家负载更新选择偏置，以调整专家分配；专家输出的混合权重使用未加入该偏置的路由分数。该机制参考Auxiliary-Loss-Free Load Balancing，[3] 将负载调节与主任务损失的梯度优化分开，当前训练不加入路由均衡辅助损失。

模型同时采用DenseFormer的深度加权平均机制。[4] 每层对当前表示和先前层表示进行可学习的加权组合，为浅层与深层特征之间提供直接的信息传递路径。该机制与token维度上的attention、Gated DeltaNet和长卷积分别作用于不同维度，共同构成模型的信息流组织方式。

#### 计算结构与执行实现协同

长卷积使用FFT执行因果滤波，在保持滤波权重参与预测的前提下，避免直接按全部lag展开长核计算。训练实现结合滤波响应复用、状态分片与梯度累积；推理实现结合FFT布局优化、低精度矩阵计算和算子融合，以控制长卷积及混合主干的执行开销。

本方案同时关注预测质量、绝对前向耗时和参数归一化效率。评测中的EvalTime按以下关系计算：

\[
\mathrm{EvalTime}=\frac{T\,10^{15}}{N_aD},
\]

其中\(T\)为累计前向时间，\(D\)为计分token数，\(N_a\)为平均非词表活跃参数数目。长卷积的算法实现改变了参数容量与实际执行工作量的关系，因此评测报告同时列出绝对时间与归一化指标，分别反映实际运行成本和该参数统计口径下的执行效率。

本方案在已有Gated DeltaNet、Engram、无辅助损失均衡和深度加权平均机制的基础上，完成与矩阵值长卷积的架构组合、预算配置和实现适配。贡献集中在这些组件在固定预算模型系列中的协同组织及其训练、推理实现。

### 模型系列配置

四档模型使用相同的组件组合，按预算调整宽度、深度和训练量。以下LC位置均采用从0开始的层索引，卷积通道数等于该档hidden维度。

| 档位 | 目标训练预算FLOPs | hidden | 层数 | 路由专家数 / top-k | LC位置 | 卷积核长度 |
|---|---:|---:|---:|---:|---|---:|
| C1 | 3×10¹⁷ | 512 | 10 | 512 / 4 | 3、6、8 | 2048 |
| C2 | 1×10¹⁸ | 640 | 12 | 512 / 4 | 4、8、10 | 2048 |
| C3 | 3×10¹⁸ | 768 | 14 | 512 / 4 | 4、9、12 | 2048 |
| C4 | 1×10¹⁹ | 768 | 16 | 512 / 4 | 5、10、14 | 2048 |

### 训练方法

模型采用统一tokenizer，词表大小为10112，训练序列长度为4096。训练数据按固定长度序列块打乱，在各rank之间划分；当预算需要跨轮读取时，按轮次重新打乱。各档训练token数依据实际结构的每token计算量重新核算，并预留0.5%的预算用于转换、校准等附加工作。

主任务为自回归下一token预测。优化器采用Muon与AdamW组合，并使用学习率预热和衰减调度。训练以BF16混合精度执行，结合分布式数据并行、优化器状态分片、FFN重计算及梯度桶复用。当前配置不启用MTP、mHC或学习式稀疏attention索引器。

### 推理与评测方法

本模型系列采用纯神经网络预测头，不接入外部统计回退头、检索缓存混合头或在线输出头适配。推理统一使用长度10240的序列设置，主干采用FP8计算路径，输出完整位置和完整词表的BF16 logits，并保留路由偏置与深度混合系数所需的FP32控制参数。

语言建模质量以BPB衡量。BPB由总负对数似然和对应的UTF-8文本字节数计算；四档模型的质量与实际训练计算量用于构建预算—性能曲线。推理报告记录硬件、精度、实际batch、计分token数、累计前向秒数、活跃参数统计和EvalTime。完整评测结果与所提交权重、配置和源码版本一一对应。

### 代码与技术来源

基础文本模型实现来自Transformers 5.19.0中的`transformers.models.qwen3_5_moe`，保留其Apache-2.0许可证和来源声明。团队初始Q38结构参考`huangzhenhua111/hzh-q38f-sparse`的提交`12862f6ded28d4cedf7de411d3a632d660ade140`；在该谱系上完成结构选择、长卷积分支、组合实验、预算核算以及训练和推理实现适配。实际采用的组件与配置以上述描述为准。

1. Yang et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. https://arxiv.org/abs/2412.06464
2. Cheng et al. *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*. https://arxiv.org/abs/2601.07372
3. Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*. https://arxiv.org/abs/2408.15664
4. Pagliardini et al. *DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging*. https://arxiv.org/abs/2402.02622
5. Li et al. *What Makes Convolutional Models Great on Long Sequence Modeling?* https://arxiv.org/abs/2210.09298
6. Poli et al. *Hyena Hierarchy: Towards Larger Convolutional Language Models*. https://arxiv.org/abs/2302.10866
7. Fu et al. *FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores*. https://arxiv.org/abs/2311.05908

### 提交冻结时填写的字段

此表属于组内提交准备，不与上方架构正文混排。描述性正文可以直接采用；下列运行产物由最终提交包补齐，不能用目标值代替实际完成值。

| 字段 | 填写内容 | 依据 |
|---|---|---|
| 模型版本与代码 | 最终源码提交、四档模型路径与权重SHA256 | 冻结提交包及文件清单 |
| 实际训练量 | 各档完成token数、实际训练FLOPs和附加计算记录 | 训练summary、配置及计算账本 |
| 参数量 | 各档总参数、非词表参数及评测Na | 导出模型与实际前向统计 |
| 实验环境 | 实际训练/评测硬件、依赖版本、精度与batch | 环境清单和run-metadata |
| 四档质量与时间 | C1–C4完整BPB、前向秒数、D、Na、EvalTime、拟合Performance | 同一设置的完整评测结果 |
| 复现材料 | 最终包内训练/评测入口、依赖及Docker命令 | 干净环境复现记录 |
| 消融与样本分析 | 仅填写已完成的对照、数据划分和统计结果 | 完整配对NLL或实验输出 |

架构正文没有填写尚未完成的性能提升数值，也不以按收益筛选的样本集合替代整体或独立验证结果。若最终改变专家数、LC位置、预测头或精度，应同时更新正文、配置表与评测记录。
