# 比赛协作入口

本目录是新H100服务器上供同学和agent共同维护的文档主源。项目根目录为 `/models/q38-b-round-20261109`，仓库为其中的 `team-repo`。Mac上的导出文件只用于阅读；修改在此仓库保留Git记录。

## 文档

| 文件 | 用途 |
|---|---|
| [architecture_and_evidence.md](architecture_and_evidence.md) | 开发讨论、架构来源、证据边界，以及附后的Model Card正文 |
| [model_card_submission.md](model_card_submission.md) | 可直接填写Model Card的独立正式正文 |
| [c2c3_training_plan.md](c2c3_training_plan.md) | C2/C3配置、迁移与训练安排 |
| [COORDINATION.md](COORDINATION.md) | 分工、认领、反馈与避免覆盖的约定 |
| [handoffs](handoffs/) | 双方交接、问题、确认及实验结果，每次独立文件 |

## 先读实时状态

1. 项目根目录 `SETUP_C2C3_STATUS.json`：当前自动队列阶段。
2. `b-round-20261109/coordination.sqlite`：训练/评测认领及心跳。
3. `b-round-20261109/BOARD.md`：带核验时间的运行摘要。
4. 本目录 `handoffs/` 最新记录，以及对方尚未确认的问题。

运行日志和状态比旧文档中的进度快照更新。启动任何GPU作业前重新查看nvidia-smi和任务登记。项目根目录的 `START_HERE.md`、`docs` 为本目录的快捷入口，指向同一份文档。
