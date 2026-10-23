# 1e17 / h320_L6

模型：6 层 × hidden 320（宽深比 53）；全注意力 q/kv 头 10/1，GDN k/v 头 5/10（头维均 64）；64 专家 top-4，专家/共享专家宽 160；GDN/full attention=3:1；词表 gzy 10K（10112）。层内比例由 shape_for 统一生成。

预算 100,000,000,000,000,000 FLOPs；训练 884,604,928 token，6,749 步；每 token 113,038,080 FLOPs。

核算口径：`megatron_b92c3ed_plus_router_shared_gate_v1`；GDN 采用比赛指定 Megatron 核心公式，另计 router 和共享专家门投影；参数量与 FLOPs 分开统计。

非词表总参数 64,842,468，激活非词表参数 9,546,468（Na/N=14.7%）；D/Na=93。

全局 batch=131072 token（16 条序列），seq=8192；micro_bs 从 8 起自动探测（auto_micro_bs=1），梯度累积 = 16/(卡数×micro_bs)；warmup=337 步；cosine 降到峰值的 10%。

峰值学习率 0.002338（0.3118·C^-0.125，同一预算内统一）；model seed=0，data seed=0。所有预算/模型共用同一无放回打乱块序列。验证分区按完整文档冻结，见 ../../validation.json。

输出：`runs/1e17/h320_L6/`（日志、checkpoint、TensorBoard、final_eval.json、HF 导出）。

仅检查（无 GPU 分配，不训练）：

```bash
GPUS=0 bash scripts/train_isoflops.sh 1e17 h320_L6 --dry-run
```

以后启动：

```bash
GPUS=0 bash scripts/train_isoflops.sh 1e17 h320_L6
```

CLI 参数覆盖 JSON；修改预算、学习率或 seed 后须使用新 run_dir。
