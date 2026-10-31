# Agent 协作约定

多名成员及其 agent 在 H100（zkrh-54）上并行优化 C4 推理。以下约定适用于全部推理优化工作。

## 共享优化日志

每次优化尝试的做法和结果都记录在 H100 的 `/data/q38/optimization-log/`，失败和放弃的尝试也要记录。目录结构、记录模板和比较口径见该目录的 `README.md`。

- 开始前：阅读 `LEDGER.md`、`running/` 和最近的 `entries/`，避免与他人重复。
- 运行时：在 `running/` 登记 GPU、tmux 会话和停止条件，任务结束后删除登记。
- 结束后：在 `entries/` 写详细记录，并在 `LEDGER.md` 追加一行。

## 主办方补充说明（2026-10-31 转录，原文见 H100 上的 `/data/q38/optimization-log/OFFICIAL_FAQ.md`）

- 评测为单卡 NVIDIA H200 141 GB HBM3e，不是 H100 80 GB。H200 的 Tensor Core 峰值与 H100 SXM 相同，显存带宽约 4.8 TB/s（H100 约 3.35 TB/s）。H100 上的结论要注明这一差别，最大 batch 约为 H100 的 1.8 倍。
- 四个档位的评测设置和数值精度必须一致，Performance 与 EvalTime 使用同一套评测参数；不能只给 1e19 档用 FP8。推理改动需在 C1–C4 上都能运行。
- EvalTime 的常规下界：BF16 > 2，FP8 > 1（F_forward > 2·N_a·D）。官方说明已有队伍通过架构设计逼近或突破该下界，符合规则的设计不受限制。
- 词表参数（nn.Embedding、输出 lm_head）不计入 N、N_a；其中的普通矩阵乘参数照常计入。

## GPU

H100 由多人共用。启动前查看 `nvidia-smi` 和 `running/`。每人默认使用 2 张卡，最多 4 张，不停止他人的进程。

## Baseline

当前 C1/C4 baseline 以 H100 上的 `/data/q38/experiments/C1_C4_ACTIVE_BASELINES.json` 为准。
