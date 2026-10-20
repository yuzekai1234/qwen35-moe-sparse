# Qwen3.5-MoE Sparse 预训练基线

按《Qwen3.5-MoE Sparse 基线：模型来源、等比例缩放与训练代码》部署，源码来自服务器共享目录 `/home/shared/hzh_q35_sparse`，原作者黄振华。独立工作目录：`4090-8:/home/yzk/qwen35-moe-sparse`。

只复用 Qwen3.5-35B-A3B 的文本结构，**随机初始化、不下载原模型权重**。保留 3:1 Gated DeltaNet/全注意力、64 选 4 MoE、共享专家和门；使用 EXAONE 153600 词表，绑定输入/输出嵌入，无视觉塔、无 MTP。

## 五组原始配置

| 预设 | 层 × 宽 | 总参数 | 训练 token | 峰值学习率 | 文档 bpb（历史结果） |
|---|---:|---:|---:|---:|---:|
| q35_c1_s1 | 5 × 96 | 23.9M | 2.95B | 0.004 | 1.2703 |
| q35_c1_s2 | 8 × 192 | 88.2M | 1.25B | 0.004 | 1.1859 |
| q35_c1_s3 | 12 × 256 | 194.8M | 0.82B | 0.0035 | 1.2134 |
| q35_c2_m | 8 × 256 | 143.0M | 3.11B | 0.0035 | 1.0949 |
| q35_c2_b | 12 × 192 | 117.6M | 3.67B | 0.004 | 1.0982 |

精确 token 预算和 warmup 步数从 `run_configs/<预设>/run_config.json` 读取。公共设置为 seq=8192、全局 batch=524288 token、AdamW(0.9, 0.95)、wd=0.1、clip=1、bf16、seed=0、cosine 降到峰值学习率的 0.1 倍。表中结果来自文档，并非本次重新训练得到。

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
  --tokenizer tokenizer/exaone_154k --data_dir /home/shared/data \
  --out data/corpus_exaone154k --every 12 --sub 6 --workers 8
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

默认 `--resume 1`，重跑原命令会从最新 `ckpt.pt` 续训。checkpoint 保存 FP32 主权重、优化器、训练 token 数与各 rank 随机状态，正常结束也会保存。相同 GPU 数和 micro batch 可恢复采样流；当前 BF16/FLA/MoE 运行环境的独立同种子训练仍观察到数值差异，不保证最终权重逐位相同；更改 GPU 数/micro batch 可恢复状态和全局 token 计数，但采样顺序会改变。修改模型结构、数据路径、token 预算或学习率策略时使用新 `run_dir`。`--max_steps N` 仅限制此次运行到绝对第 N 步，不改变原预算的学习率曲线；可用于短程测试后继续原预算训练。

输出在 `runs/<预设>/`：`ckpt.pt`、`training_metrics.csv`、`run_config.json`、`summary.json`、HF 导出 `hf/`；终端日志在 `logs/`。`summary.json` 的 `completed_budget` 区分短程测试和预算完成。

## 评测与实际 FLOPs

```bash
PY=$PWD/.venv_q35/bin/python bash code/eval_local_q35.sh \
  /home/shared/data/dolma3_mix_100M_a.jsonl "$PWD/runs/q35_c2_m/hf" 8192 2 1
bash code/q35_flops.sh q35_c2_m
```

沿用共享目录的官方评测适配版（EXAONE tokenizer 与 MoE 激活参数统计），输出 bpb / EvalTime。训练与评测分别启动，避免短程测试完成后意外跑完整评测。FLOPs 按实际 `tokens_trained` 与本次 `seq` 核算，写入 `runs/<run>/flops.json`。

## 快速自检

```bash
.venv_q35/bin/python -m unittest discover -s tests -v
GPUS=1 bash launch/smoke.sh
```

smoke 从训练分片读取约 262k token，使用真实 EXAONE 词表和完整 C2-m 模型，seq=256、训练两步，验证前后向、FLA、验证损失、checkpoint 和 HF 导出。它不会准备完整训练集，也不代表完成 C1/C2 训练。完整 seq=8192 测试可在 smoke 后运行：

```bash
DATA=$PWD/data/smoke/train.bin GPUS=1 bash launch/train.sh q35_c2_m \
  --run_dir "$PWD/runs/smoke_8192" --tokens 8192 --batch_tokens 8192 \
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
