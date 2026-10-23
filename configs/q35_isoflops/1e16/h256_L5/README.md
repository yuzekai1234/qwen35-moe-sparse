# 1e16 / h256_L5

模型：5 层 × hidden 256（宽深比 51）；全注意力 q/kv 头 8/1，GDN k/v 头 4/8（头维均 64）；64 专家 top-4，专家/共享专家宽 128；GDN/full attention=3:1；词表 gzy 10K（10112）。层内比例由 shape_for 统一生成。

预算 10,000,000,000,000,000 FLOPs；训练 134,348,800 token，2,050 步；每 token 74,423,808 FLOPs。

核算口径：`megatron_b92c3ed_plus_router_shared_gate_v1`；GDN 采用比赛指定 Megatron 核心公式，另计 router 和共享专家门投影；参数量与 FLOPs 分开统计。

非词表总参数 34,591,168，激活非词表参数 5,099,968（Na/N=14.7%）；D/Na=26。

全局 batch=65536 token（8 条序列），seq=8192；micro_bs 从 8 起自动探测（auto_micro_bs=1），梯度累积 = 8/(卡数×micro_bs)；warmup=102 步；cosine 降到峰值的 10%。

峰值学习率 0.003118（0.3118·C^-0.125，同一预算内统一）；model seed=0，data seed=0。所有预算/模型共用同一无放回打乱块序列。验证分区按完整文档冻结，见 ../../validation.json。

输出：`runs/1e16/h256_L5/`（日志、checkpoint、TensorBoard、final_eval.json、HF 导出）。

仅检查（无 GPU 分配，不训练）：

```bash
GPUS=0 bash scripts/train_isoflops.sh 1e16 h256_L5 --dry-run
```

以后启动：

```bash
GPUS=0 bash scripts/train_isoflops.sh 1e16 h256_L5
```

CLI 参数覆盖 JSON；修改预算、学习率或 seed 后须使用新 run_dir。
