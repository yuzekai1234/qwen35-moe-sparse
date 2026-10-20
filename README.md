# Qwen3.5-MoE Sparse 预训练基线

把 Qwen3.5-35B-A3B 的**文本结构**按固定规则缩小后随机初始化预训练（不下载原权重）：3:1 Gated DeltaNet / 全注意力、64 选 4 路由专家 + 带门共享专家、绑定嵌入，无视觉塔、无 MTP。默认词表为 gzy 30K（实际 30080，BOS=1、EOS=53、PAD=0），默认语料为共享目录里用它分好的训练集与独立验证集。源码最初来自 `/home/shared/hzh_q35_sparse`（原作者黄振华）。

## 目录结构

```
q35/                        核心包（在仓库根目录用 python -m q35.<模块> 运行）
  models/qwen3_5_moe/       可编辑的 Qwen3.5-MoE 文本模型（源自 transformers 5.19.0，仅保留文本 LM）
  presets.py                读取 configs/*.json，构造模型配置，按算力预算核算 token 数
  flops.py                  每 token 参数 / 训练 FLOPs 核算
  data.py                   语料准备（uint32 token 文件）与加载
  train.py                  训练入口（单卡 / torchrun DDP）
  evaluate.py               官方口径评测：bpb / EvalTime
configs/                    五组预设：模型形状、学习率、micro batch、算力预算、历史参考结果
scripts/                    setup_env.sh / train.sh / eval.sh / smoke.sh（env.sh 为公共环境）
tests/                      单元测试：预设与 FLOPs、语料准备、模型与原版一致性
tokenizer/gzy_30k/          默认词表（30080）及其来源记录、对应的共享语料
tokenizer/gzy_10k/          10K 词表（10112），供对照实验
```

`data/`、`runs/`、`logs/`、`.venv_q35/`、`libcuda/` 不进 Git。

## 安装

```bash
bash scripts/setup_env.sh
```

Python 3.11，torch 2.7.1 / CUDA 12.6、transformers 5.19.0、flash-linear-attention 0.5.2、liger-kernel 0.8.4（完整锁定见 `requirements.txt`）。默认清华 PyPI 源，可用 `PIP_INDEX_URL` 覆盖；Triton 的 `libcuda.so` 软链建在 `libcuda/`。

## 数据

默认直接使用共享目录里用 gzy 30K 分好的语料，无需本地准备（详见 `tokenizer/gzy_30k/README.md`）：

| 用途 | 路径 | 规模 |
|---|---|---|
| 训练 `--data` | `/home/shared/llm_pt/cache/ling-no-mtp-c1-global-pruned30k-isoflops-v1/train.bin` | 7.84B token，全局随机抽取文档，无清洗 |
| 验证 `--val_data` | `/home/shared/llm_pt/cache/ling-no-mtp-c1-holdout-pruned30k-isoflops-v1/valid.bin` | 78M token，按文本哈希独立留出 1% 文档 |

`train.bin` 全部用于训练，过程验证取 `--val_data` 的前 `--eval_tokens`（默认 2M）；`--val_data ""` 时改从 `train.bin` 末尾留出。训练启动时按语料目录的 `meta.json` / `manifest.json` 校验词表 SHA256，拒绝用其他词表分好的语料。10K 词表的同类共享语料：`/home/shared/llm_pt_ablation/cache/ling-no-mtp-c1-global-pruned10k-scale-isoflops-v1/train.bin`（5.62B token，配 `--tokenizer tokenizer/gzy_10k`，验证集在同前缀的 `-holdout-` 目录）。

为其他词表或抽样规则自行准备语料：

```bash
.venv_q35/bin/python -m q35.data corpus --out data/corpus_<名称> --tokenizer <词表目录> --data_dir /home/shared/data --every 12 --sub 6 --workers 8
```

从八个官方训练分片抽取约 1/12 文档，文档间插 EOS，生成 little-endian uint32 的 `train.bin`、`val.bin` 与 `meta.json`（含中间分片，可续跑）。更换词表或抽样规则须用新输出目录。

## 训练

```bash
GPUS=1 bash scripts/train.sh q35_c2_m                  # 单卡
GPUS=1,2 bash scripts/train.sh q35_c2_m                # 双卡，全局 batch 不变，梯度累积自动调整
GPUS=1,2 bash scripts/train.sh q35_c1_s2 --max_steps 100 --run_dir runs/try
```

`scripts/train.sh <预设> [参数...]` 运行 `q35.train --preset <预设>`，追加参数覆盖预设（全部参数见 `python -m q35.train -h`），日志追加到 `logs/<预设>.log`。先用 `nvidia-smi` 选空闲 GPU。

- **预算**：不传 `--tokens` 时按预设算力（C1=3e17，C2=1e18）和实际词表、`--seq`、`--batch_tokens` 重新核算，向下取整到完整全局 batch；warmup 默认 1% 步数（至少 10）。
- **公共设置**：seq 8192、全局 batch 524288 token、AdamW(0.9, 0.95)、wd 0.1、clip 1、bf16、seed 0、cosine 降到 0.1×lr。全局 batch 须整除 `micro_bs × seq × GPU 数`。
- **续训**：默认 `--resume 1`，重跑原命令从 `ckpt.pt` 续训（FP32 主权重、优化器、token 数、各 rank 随机状态）。改了模型、数据、词表、预算或学习率策略会拒绝续训，请用新 `--run_dir`。相同 GPU 数和 micro batch 可恢复采样流；BF16/FLA 下同种子训练仍有微小数值差异。`--max_steps N` 只把本次运行截止到第 N 步，不改变学习率曲线。
- **输出** `runs/<预设>/`：`ckpt.pt`、`training_metrics.csv`、TensorBoard 日志 `tb/`（`--tensorboard 0` 关闭）、`run_config.json`（含模型源码 SHA256）、`summary.json`（`completed_budget` 区分短程与完成）、`flops.json`、BF16 HF 导出 `hf/`。
- **监控**：`.venv_q35/bin/tensorboard --logdir runs --port 6006`，远程用 `ssh -L 6006:localhost:6006 <服务器>` 转发后打开 `http://localhost:6006`。曲线：`train/{loss,aux,lr,grad_norm,tokens}`（每步）、`perf/{tok_per_s,max_mem_gib}`、`val/loss`（每 `--log_every` / `--eval_every` 步），横轴为优化步。

## 评测

```bash
GPUS=1 bash scripts/eval.sh runs/q35_c2_m/hf [/home/shared/data/dolma3_mix_100M_a.jsonl] [seq=8192] [batch=2]
```

输出 bpb / EvalTime 到 `runs/<run>/eval_<数据集>_seq<seq>.json`。评测导入仓库内模型实现，并用 `PreTrainedTokenizerFast` 原样加载 tokenizer（`AutoTokenizer` 会被替换成 Qwen 分词器，导致 bpb 虚高）。

FLOPs 核算：`python -m q35.flops q35_c2_m`（按默认 30K 词表核算预算）或 `python -m q35.flops runs/<run>/hf/config.json`。

## 自检

```bash
.venv_q35/bin/python -m unittest -v      # 需要 GPU 的用例在无 GPU 时跳过
GPUS=1 bash scripts/smoke.sh             # 完整 C2-m 模型、seq 256、两步训练 + checkpoint + HF 导出
```

## 修改模型

| 位置 | 内容 |
|---|---|
| `q35/models/qwen3_5_moe/modeling_qwen3_5_moe.py` | `Qwen3_5MoeGatedDeltaNet`（GDN）、`Qwen3_5MoeAttention`（GQA + 输出门）、`Qwen3_5MoeTopKRouter` / `Experts` / `SparseMoeBlock`、`DecoderLayer`、`TextModel`、`ForCausalLM` |
| `q35/models/qwen3_5_moe/configuration_qwen3_5_moe.py` | 配置字段与默认值 |
| `q35/presets.py` | 命令行形状参数 → 模型配置（层型、头数、专家等） |
| `q35/train.py` `LossModel` | 训练 loss：直接调用 backbone + Liger 融合 CE + 路由辅助 loss，不走 `ForCausalLM.forward()` |
| `q35/flops.py`、`q35/evaluate.py` `active_parameter_count()` | 参数 / FLOPs 口径，改路由或结构后需同步 |

流程：改代码 → 新 `--run_dir` 短训与评测 → 提交。训练启动会断言 `flops.py` 的参数统计与实际模型一致。HF 导出携带模型源码与 `auto_map`（`model_type=qwen35_moe_sparse`），他处加载用 `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`。`tests/test_model.py` 校验本地实现与原版 transformers 初始化逐位一致、GPU 前向对齐。

## 缩放规则与历史结果

| 结构项 | 原版（h=2048，40 层） | 本基线（给定 h、L） |
|---|---|---|
| 层型 | 每 4 层 `[lin, lin, lin, full]` | 同；L 非 4 的倍数时余下接 lin |
| 路由专家 / 宽 | 256 选 8，宽 h/4 | 64 选 4，宽 h |
| 共享专家 | 1 个，宽 h/4，sigmoid 门 | 1 个，宽 h/2，sigmoid 门 |
| 全注意力 | head_dim 256，q/kv 16/2 | head_dim 64，q/kv ≈ h/64 / 1 |
| GDN 头 k/v、头维 | 16/32，128 | 2/4，64（h96 用 1/2） |
| 其余 | aux 0.001、partial_rotary 0.25、rope_theta 1e7、conv 4 | 同；绑定嵌入 |

下表 bpb 为 **EXAONE 153600 词表**的历史结果（`configs/*.json` 的 `reference` 字段），不代表 30K 词表成绩。

| 预设 | 算力 | 层 × 宽 | 总参数（153600 词表） | 训练 token | lr | bpb | EvalTime |
|---|---|---:|---:|---:|---:|---:|---:|
| q35_c1_s1 | 3e17 | 5 × 96 | 23.9M | 2.95B | 4e-3 | 1.2703 | 2107 |
| q35_c1_s2 | 3e17 | 8 × 192 | 88.2M | 1.25B | 4e-3 | **1.1859** | 516 |
| q35_c1_s3 | 3e17 | 12 × 256 | 194.8M | 0.82B | 3.5e-3 | 1.2134 | 304 |
| q35_c2_m | 1e18 | 8 × 256 | 143.0M | 3.11B | 3.5e-3 | **1.0949** | 297 |
| q35_c2_b | 1e18 | 12 × 192 | 117.6M | 3.67B | 4e-3 | 1.0982 | 578 |

同算力对比：Ling 1.2885 / 1.1660，dense 1.3056 / 1.2097（C1 / C2）。

## 来源

- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)、[Transformers](https://github.com/huggingface/transformers)（模型源码 Apache-2.0，见 `q35/models/qwen3_5_moe/LICENSE` 与 `UPSTREAM.json`）
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)、[Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- 词表来源见 `tokenizer/gzy_30k/README.md`、`tokenizer/gzy_10k/README.md`
