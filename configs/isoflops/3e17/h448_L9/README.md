# 3e17 / h448_L9

模型：9 层 × hidden 448（宽深比 50）；全注意力 q/kv 头 14/1，GDN k/v 头 7/14（头维均 64）；64 专家 top-4，专家/共享专家宽 224；GDN/full attention=3:1；词表 gzy 10K（10112）。层内比例由 shape_for 统一生成。

预算 300,000,000,000,000,000 FLOPs；训练 1,030,356,992 token，7,861 步；每 token 291,156,096 FLOPs。

核算口径：`megatron_b92c3ed_plus_router_shared_gate_v1`；GDN 采用比赛指定 Megatron 核心公式，另计 router 和共享专家门投影；参数量与 FLOPs 分开统计。

非词表总参数 190,289,796，激活非词表参数 27,719,556（Na/N=14.6%）；D/Na=37。

全局 batch=131072 token（16 条序列），seq=8192；micro_bs 从 8 起自动探测（auto_micro_bs=1），梯度累积 = 16/(卡数×micro_bs)；warmup=393 步；cosine 降到峰值的 10%。

峰值学习率 0.002038（0.3118·C^-0.125，同一预算内统一）；model seed=0，data seed=0。所有预算/模型共用同一无放回打乱块序列。验证分区按完整文档冻结，见 ../../validation.json。

输出：`runs/3e17/h448_L9/`（日志、checkpoint、TensorBoard、final_eval.json、HF 导出）。

仅检查（无 GPU 分配，不训练）：

```bash
GPUS=0 bash scripts/train_isoflops.sh 3e17 h448_L9 --dry-run
```

以后启动：

```bash
GPUS=0 bash scripts/train_isoflops.sh 3e17 h448_L9
```

CLI 参数覆盖 JSON；修改预算、学习率或 seed 后须使用新 run_dir。
