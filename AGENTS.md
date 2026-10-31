# Agent 协作约定

多名成员及其 agent 在 H100（zkrh-54）上并行优化 C4 推理。以下约定适用于全部推理优化工作。

## 共享优化日志

每次优化尝试的做法和结果都记录在 H100 的 `/data/q38/optimization-log/`，失败和放弃的尝试也要记录。目录结构、记录模板和比较口径见该目录的 `README.md`。

- 开始前：阅读 `LEDGER.md`、`running/` 和最近的 `entries/`，避免与他人重复。
- 运行时：在 `running/` 登记 GPU、tmux 会话和停止条件，任务结束后删除登记。
- 结束后：在 `entries/` 写详细记录，并在 `LEDGER.md` 追加一行。

## GPU

H100 由多人共用。启动前查看 `nvidia-smi` 和 `running/`。每人默认使用 2 张卡，最多 4 张，不停止他人的进程。

## Baseline

当前 C1/C4 baseline 以 H100 上的 `/data/q38/experiments/C1_C4_ACTIVE_BASELINES.json` 为准。
