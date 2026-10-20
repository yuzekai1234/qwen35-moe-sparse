# Qwen3.5-MoE Sparse 基线：训练代码与配置

Qwen3.5-35B-A3B 文本部分等比例缩小后的预训练代码（黄振华）。说明文档：https://alidocs.dingtalk.com/i/nodes/nYMoO1rWxaqnQ6lMCRGv1qzbV47Z3je9 （本目录 DOC.md 为同内容副本）

| 文件 | 干嘛 |
|---|---|
| `code/train_q35.py` | 训练（单卡 / torchrun DDP，ckpt 续跑，结束导出 HF 格式到 `runs/<run>/hf`） |
| `code/count_flops_q35.py`、`code/q35_flops.sh` | 参数量与训练 FLOPs（按实际训练 token 核算，写 `runs/<run>/flops.json`） |
| `code/eval_q35.py`、`code/eval_local_q35.sh` | 官方 eval.py，只改了 tokenizer 加载（`PreTrainedTokenizerFast`），单卡评测 |
| `code/prepare_corpus_u32.py` | 从 `/home/shared/data` 的 8 个分片抽 1/12 文档分词成 uint32 `train.bin`/`val.bin`（约 3.8B token，15 GB） |
| `tokenizer/exaone_154k/` | 训练用的 EXAONE 154k 词表（tokenizer.json 同 `/home/shared/tokenizer`） |
| `env/requirements.txt`、`env/setup_env.sh` | 依赖清单（torch 2.7.1+cu126、transformers 5.19、fla 0.5.2、liger 0.8.4）与建环境命令 |
| `launch/train_c2_m_1gpu.sh`、`launch/train_c2_m_2gpu.sh` | C2 最优点 8×256 的单卡 / 双卡启动样例 |
| `run_configs/` | 5 个跑过的点（C1 s1–s3、C2 m/b）的 run_config / flops / summary json，及原版 HF config |
| `fig/q35_c1c2.png` | 结果图 |

怎么起（数据和权重不在这里，太大）：
```bash
cp -r /home/shared/hzh_q35_sparse ~/q35 && cd ~/q35
bash env/setup_env.sh                                   # 建 .venv_q35 + libcuda 软链
cd code && ../.venv_q35/bin/python prepare_corpus_u32.py --tokenizer ../tokenizer/exaone_154k \
    --data_dir /home/shared/data --out ../data/corpus_exaone154k --every 12 --sub 6 && cd ..   # 建语料，一次即可
bash launch/train_c2_m_2gpu.sh                          # 或 GPUS=3 bash launch/train_c2_m_1gpu.sh
```
换尺寸就改启动脚本里的 `--layers/--hidden/--heads/--moe_ffn/--shared_ffn/--tokens/--lr`，规则见 DOC.md 第 2 节；`batch_tokens` 必须能被 micro_bs×8192×卡数 整除。
原始位置：gamma-4090 `/local/hzh-agent/scicomp26/`（C2 runs），4090a `/home/hzh/scicomp26/runs/q35_c1_s*`（C1）。
