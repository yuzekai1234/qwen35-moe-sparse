# Qwen3.5-MoE Sparse 预训练基线

按《Qwen3.5-MoE Sparse 基线：模型来源、等比例缩放与训练代码》部署，源码来自服务器共享目录 `/home/shared/hzh_q35_sparse`，原作者黄振华。独立工作目录：`4090-8:/home/yzk/qwen35-moe-sparse`。

只复用 Qwen3.5-35B-A3B 的文本结构，**随机初始化、不下载原模型权重**。保留 3:1 Gated DeltaNet/全注意力、64 选 4 MoE、共享专家和门；本分支默认使用 **gzy 10K 词表（实际 10112）**，绑定输入/输出嵌入，无视觉塔、无 MTP。

`10k词表` 分支将合作者 guozeyuan 的词表完整保存到 `tokenizer/gzy_10k/`，来源及裁剪记录见该目录 README 和 `pruning.json`。BOS=1、EOS=53、PAD=0，训练配置使用分词器自己的特殊 token ID。新语料统一使用 little-endian uint32，即使词表小于 65536，也不读取为 uint16。

默认语料目录为 `data/corpus_gzy10k/`，运行目录为 `runs/<预设>_gzy10k/`。五组模型形状不变，启动器按实际词表和序列长度重新计算每 token FLOPs，在原 C1/C2 预算内向下取整完整全局 batch，并重新设置约 1% warmup。更换词表必须重新分词，旧 EXAONE token 缓存与 checkpoint 不能作为 10K 训练输入。

## 模型源码与魔改入口

训练和评测实际导入仓库内的 `models/qwen3_5_moe/`，启动时会打印源码位置：

| 文件 / 类 | 用途 |
|---|---|
| `models/qwen3_5_moe/modeling_qwen3_5_moe.py` | 模型主体，直接在此修改 |
| `Qwen3_5MoeGatedDeltaNet` | GDN 线性注意力 |
| `Qwen3_5MoeAttention` | 全注意力 / GQA / 输出门 |
| `Qwen3_5MoeTopKRouter`、`Qwen3_5MoeExperts`、`Qwen3_5MoeSparseMoeBlock` | 路由、专家、共享专家与组合 |
| `Qwen3_5MoeDecoderLayer`、`Qwen3_5MoeTextModel` | 层内结构与层间连接 |
| `Qwen3_5MoeForCausalLM` | 语言模型输出与标准 forward/loss |
| `models/qwen3_5_moe/configuration_qwen3_5_moe.py` | 配置字段与默认值 |
| `code/train_q35.py` | 五组缩放参数构造、训练循环和融合 loss |

源码固定来自安装的 Transformers 5.19.0，保留版权和 Apache-2.0 许可，原始文件哈希见该目录 `UPSTREAM.json`。本仓库直接维护这些文件，不运行上游 modular 生成器。为便于核对原版，完整保留上游同文件中的视觉/多模态类，但训练只实例化文本 LM。Transformers 的通用框架、FLA 和 Liger 仍作为版本固定的依赖。

训练为了节省显存，使用 `code/train_q35.py` 的 `LossModel` 调用文本 backbone（`lm.model`）并用 Liger 计算融合 CE；不会调用 `Qwen3_5MoeForCausalLM.forward()` 的标准 loss 路径。若改输出头、训练 loss 或路由辅助 loss，也要修改该训练包装器。

魔改流程：修改模型文件或配置 → 使用新的 `--run_dir` 做短训与评测 → `git add` / `git commit` 记录改动。新运行的 `run_config.json` 会保存模型源码 SHA256。改变路由、专家数或结构后，现有 `count_flops_q35.py` 和评测里的 `active_parameter_count()` 也要同步调整统计口径。

新 HF 导出自动携带模型和配置的 Python 源码及 `auto_map`，使用独立 `model_type=qwen35_moe_sparse`。本仓库评测直接导入当前 Git 版本的实现；在其他环境加载导出目录时，显式使用自己导出的代码：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/run/hf", trust_remote_code=True, local_files_only=True
)
```

模型移入仓库时未改计算结构。原版/本地参数兼容、GPU 前向对齐、导出源码与再次加载测试见 `tests/test_local_model.py`。不要修改 `.venv_q35/site-packages` 下的模型副本。

## 五组原始配置

| 预设 | 层 × 宽 | 总参数 | 训练 token | 峰值学习率 | 文档 bpb（历史结果） |
|---|---:|---:|---:|---:|---:|
| q35_c1_s1 | 5 × 96 | 23.9M | 2.95B | 0.004 | 1.2703 |
| q35_c1_s2 | 8 × 192 | 88.2M | 1.25B | 0.004 | 1.1859 |
| q35_c1_s3 | 12 × 256 | 194.8M | 0.82B | 0.0035 | 1.2134 |
| q35_c2_m | 8 × 256 | 143.0M | 3.11B | 0.0035 | 1.0949 |
| q35_c2_b | 12 × 192 | 117.6M | 3.67B | 0.004 | 1.0982 |

上述表格与 `run_configs/<预设>/` 均保留为 **153600 词表的历史记录**，不代表 10K 成绩。本分支只从历史配置复用模型形状和训练超参，默认 token 预算及 warmup 在启动时重新核算；用 `--dry-run` 查看命令。公共设置为 seq=8192、全局 batch=524288 token、AdamW(0.9, 0.95)、wd=0.1、clip=1、bf16、seed=0、cosine 降到峰值学习率的 0.1 倍。

## 安装

```bash
ssh 4090-8
cd /home/yzk/qwen35-moe-sparse
bash env/setup_env.sh
```

Python 3.11，torch 2.7.1 / CUDA 12.6、transformers 5.19.0、flash-linear-attention 0.5.2、liger-kernel 0.8.4；完整依赖见 `env/requirements.txt`。默认使用清华 PyPI 源，可用 `PIP_INDEX_URL` 覆盖。安装脚本不会改动其他项目环境。Triton 的 `libcuda.so` 软链建在本项目 `libcuda/`。

## 完整训练语料（首次准备一次）

```bash
cd /home/yzk/qwen35-moe-sparse
TOKENIZERS_PARALLELISM=false .venv_q35/bin/python code/prepare_corpus_u32.py \
  --tokenizer tokenizer/gzy_10k --data_dir /home/shared/data \
  --out data/corpus_gzy10k --every 12 --sub 6 --workers 8
```

从八个官方训练分片按原脚本规则抽取约 1/12 文档，以 EOS 分隔，生成 uint32 `train.bin`、`val.bin` 和元数据；处理会读取大量数据并占用约 30GB（包含中间分片）。已有相同配置的分片可继续处理；更换词表、抽样规则或语料目录时必须使用新输出目录。训练脚本与原版相同，从 `train.bin` 末尾留出 `eval_tokens`（默认 2M）作训练过程验证；`val.bin` 单独保留。官方评测语料不会用于训练。

## 启动训练

先用 `nvidia-smi` 选择空闲 GPU，以下卡号只是示例。

```bash
# 单卡：默认 C2 最优点，8 层 × 256
GPUS=1 bash launch/train.sh q35_c2_m
# 双卡：全局 batch 不变，梯度累积自动调整
GPUS=1,2 bash launch/train.sh q35_c2_m
# C1 最优点
GPUS=1,2 bash launch/train.sh q35_c1_s2
# 查看命令；不会启动训练
.venv_q35/bin/python code/launch.py q35_c2_m --nproc 2 --dry-run
```

全部五组预设均可启动。通过 `DATA=/path/to/train.bin` 指定现有 EXAONE uint32 语料，通过 `PY` 指定解释器。额外训练参数可直接追加，例如 `--micro_bs 2` 或 `--run_dir /path/to/new-run`；全局 batch 必须整除 `micro_bs × seq × GPU 数量`。命令错误和训练错误会通过 `pipefail` 正常返回失败。

默认 `--resume 1`，重跑原命令会从最新 `ckpt.pt` 续训。checkpoint 保存 FP32 主权重、优化器、训练 token 数与各 rank 随机状态，正常结束也会保存。相同 GPU 数和 micro batch 可恢复采样流；当前 BF16/FLA/MoE 运行环境的独立同种子训练仍观察到数值差异，不保证最终权重逐位相同；更改 GPU 数/micro batch 可恢复状态和全局 token 计数，但采样顺序会改变。修改模型结构、数据路径、token 预算或学习率策略时使用新 `run_dir`。`--max_steps N` 仅限制此次运行到绝对第 N 步，不改变原预算的学习率曲线；可用于短程测试后继续原预算训练。新增运行记录 tokenizer SHA256，恢复时拒绝原路径下被替换的词表；新语料元数据也参与词表一致性检查。

输出在 `runs/<预设>_gzy10k/`：`ckpt.pt`、`training_metrics.csv`、`run_config.json`、`summary.json`、HF 导出 `hf/`；终端日志在 `logs/`。`summary.json` 的 `completed_budget` 区分短程测试和预算完成。

## 评测与实际 FLOPs

```bash
PY=$PWD/.venv_q35/bin/python bash code/eval_local_q35.sh \
  /home/shared/data/dolma3_mix_100M_a.jsonl "$PWD/runs/q35_c2_m_gzy10k/hf" 8192 2 1
bash code/q35_flops.sh q35_c2_m_gzy10k
```

评测从 HF 导出目录原样读取 tokenizer，兼容 gzy 10K 与历史 EXAONE，输出 bpb / EvalTime。训练与评测分别启动，避免短程测试完成后意外跑完整评测。FLOPs 按实际 `tokens_trained` 与本次 `seq` 核算，写入 `runs/<run>/flops.json`。

## 快速自检

```bash
.venv_q35/bin/python -m unittest discover -s tests -v
GPUS=1 bash launch/smoke.sh
```

smoke 从训练分片读取约 262k token，使用 gzy 10K 词表和完整 C2-m 模型，写入 `data/smoke_gzy10k/` 和 `runs/smoke_gzy10k/`；seq=256、训练两步，验证前后向、FLA、验证损失、checkpoint 和 HF 导出。它不会准备完整训练集，也不代表完成 C1/C2 训练。完整 seq=8192 测试可在 smoke 后运行：

```bash
DATA=$PWD/data/smoke_gzy10k/train.bin GPUS=1 bash launch/train.sh q35_c2_m \
  --run_dir "$PWD/runs/smoke_8192_gzy10k" --tokens 8192 --batch_tokens 8192 \
  --micro_bs 1 --eval_tokens 16384 --warmup 1 --log_every 1 --resume 0
```

双卡中断恢复对比测试：`GPUS=1,2 bash tests/gpu_resume.sh`。它会逐项检查恢复时模型/优化器/各卡随机状态与 checkpoint 完全一致，并比较续训与连续训练的 token 计数和采样状态；最终权重差异仅报告数值。

本次部署验证见 `docs/VALIDATION.md`。语料、环境、训练日志、checkpoint、模型权重均被 Git 忽略；仓库包含代码、词表、五组历史配置和核算记录。

## 来源

- 共享源码：`/home/shared/hzh_q35_sparse`，原始 README 见 `docs/UPSTREAM_README.md`，说明见 `DOC.md`。
- [Qwen 官方模型与配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)

本仓库保留共享源码作者归属，未自行赋予其新的开源许可。新增部署、启动和健壮性修复详见 `docs/CHANGES.md`。

服务器仓库已配置专用 GitHub deploy key（仅此仓库，可推送），通过 SSH 443 端口连接；私钥在 `/home/yzk/.ssh/`，不进入仓库。服务器上可直接运行 `git pull --ff-only` / `git push`。
