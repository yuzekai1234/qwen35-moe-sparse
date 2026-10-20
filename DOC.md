# Qwen3.5-MoE Sparse 基线：模型来源、等比例缩放与训练代码

> 一句话：把 **Qwen3.5-35B-A3B 的文本部分**按固定规则缩小，用组里统一的 EXAONE 154k 词表、seq 8192、同一份数据和官方评测训练。C1（3e17）底 **1.1859**，C2（1e18）底 **1.0949**；同算力下 Ling 是 1.2885 / 1.1660，dense 是 1.3056 / 1.2097。

## 1. 模型来源

- 参照模型：https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- 原版配置：https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json
- 线性注意力 kernel：https://github.com/fla-org/flash-linear-attention
- 模型实现（`Qwen3_5MoeForCausalLM`）：https://github.com/huggingface/transformers

只用结构不用权重（从零初始化）；只取文本部分，去掉视觉塔和 MTP；词表换成 EXAONE 154k，嵌入绑定。

## 2. 怎么等比例缩放

原版结构：40 层、宽 h=2048，每 4 层 3 层 Gated DeltaNet（GDN，线性注意力）+ 1 层门控 GQA 全注意力；每层 MoE 是 256 个路由专家选 8 + 1 个共享专家。我们固定"结构比例"，只调层数 L 和宽 h，其余按下表随 h 走。

### 2.1 缩放规则

| 结构项 | 原版（h=2048，40 层） | 我们（给定 h、L） |
|---|---|---|
| 层型 | 每 4 层 `[lin, lin, lin, full]`（30 GDN + 10 全注意力） | **同，3:1**；L 不是 4 的倍数时余下接 lin（5 层 = `[lin,lin,lin,full,lin]`） |
| 路由专家 | 256 选 8 | **64 选 4**（同 Ling C1；256 个在小模型上太碎） |
| 专家宽 `moe_ffn` | 512 = h/4 | **= h** |
| 共享专家 | 1 个，宽 512 = h/4，带 sigmoid 门 | 1 个，**宽 h/2**，带 sigmoid 门 |
| 全注意力 head_dim | 256 | **64** |
| 全注意力头数 q/kv | 16 / 2（GQA + 输出门） | **≈h/64 / 1**（h96→2/1，h192、h256→4/1），保留输出门 |
| GDN 头 k/v、头维 | 16 / 32，128 | **2 / 4，64**（h96 用 1/2） |
| 不变的 | aux loss 0.001、partial_rotary 0.25、rope_theta 1e7、conv 4、RMSNorm 1e-6 | 同 |
| 词表 / 嵌入 | Qwen 248k，不绑定 | EXAONE 154k（153600），**绑定** |

### 2.2 尺寸怎么选

- **C1（3e17）**：跑小 / 中 / 大三点把 U 形曲线的底夹住，小点取 Ling C1 的尺寸（5×96），中点 8×192，大点 12×256。
- **C2（1e18）**：用 dense 线拟合的 N∝C^0.32–0.40 从 C1 的底（88M）外推，预测总参数（含嵌入）130–143M，在两侧各放一点：8×256（143M）和 12×192（118M）。
- token 数 = 算力 ÷ 每 token 训练 FLOPs；lr 3.5e-3–4e-3。

### 2.3 5 个配置的参数与结果

参数单位 M；FLOPs/token 为训练（前向+反向）口径，FLOPs 按实际训练 token 核算。

| run | 算力 | 层×宽 | 头 q/kv | GDN k/v | 激活非词表 | 总非词表 | 总参数（含嵌入） | 训练 FLOPs/token | 训练 token | lr | **bpb** | EvalTime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q35_c1_s1 | C1 3e17 | 5×96 | 2/1 | 1/2 | 0.91 | 9.20 | 23.9 | 101.8M | 2.95B | 4e-3 | 1.2703 | 2107 |
| **q35_c1_s2** | C1 3e17 | 8×192 | 4/1 | 2/4 | 5.63 | 58.7 | 88.2 | 240.6M | 1.25B | 4e-3 | **1.1859** | 516 |
| q35_c1_s3 | C1 3e17 | 12×256 | 4/1 | 2/4 | 13.9 | 155.5 | 194.8 | 364.2M | 0.82B | 3.5e-3 | 1.2134 | 304 |
| **q35_c2_m** | C2 1e18 | 8×256 | 4/1 | 2/4 | 9.27 | 103.6 | 143.0 | 321.4M | 3.11B | 3.5e-3 | **1.0949** | 297 |
| q35_c2_b | C2 1e18 | 12×192 | 4/1 | 2/4 | 8.45 | 88.1 | 117.6 | 272.4M | 3.67B | 4e-3 | 1.0982 | 578 |

公共设置：batch 524288 token、seq 8192、warmup ≈1% 步数、cosine 降到 0.1×lr、AdamW(0.9, 0.95) wd 0.1、clip 1.0、bf16、seed 0。

### 2.4 与 Ling / dense 对比

| 算力 | Qwen3.5-MoE 底 | Ling | dense（Qwen3 等比例）底 |
|---|---|---|---|
| C1 = 3e17 | **1.1859**（8×192） | 1.2885 | 1.3056 |
| C2 = 1e18 | **1.0949**（8×256） | 1.1660 | 1.2097 |

- 两档都比 Ling 低 0.07–0.10、比 dense 低 0.11–0.12；C2 底的 EvalTime 297，约为 Ling 的 1/4。
- C2 两点只差 0.003（单种子噪声约 0.006 以内），底落在 118–143M，与外推一致。

## 3. 训练代码

- 共享目录（主）：`4090b:/home/shared/hzh_q35_sparse/`（代码、依赖与建环境脚本、单卡/双卡启动样例、5 个配置的 json、词表；见其中 README.md）
- 原始位置：gamma-4090 `/local/hzh-agent/scicomp26/`
- 技术栈：自研单文件训练脚本，HF transformers 5.19 的 Qwen3.5-MoE 文本模型 + flash-linear-attention 0.5.2 的 GDN kernel + Liger 融合 CE，torch 2.7.1+cu126，torchrun DDP。

## 4. 踩坑

- tokenizer 加载：用 `AutoTokenizer` 加载导出的模型目录，transformers 会按 qwen3_5_moe 强制换成 Qwen 分词器，多切约 14% token，bpb 虚高到 1.46–1.52；评测一律用 `PreTrainedTokenizerFast` 原样加载。

## 结果图
