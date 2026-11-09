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
