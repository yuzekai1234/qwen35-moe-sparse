# 1e16 / h192_L5

模型：5 层 × hidden 192（宽深比 38）；全注意力 q/kv 头 6/1，GDN k/v 头 3/6（头维均 64）；64 专家 top-4，专家/共享专家宽 96；GDN/full attention=3:1；词表 gzy 10K（10112）。层内比例由 shape_for 统一生成。

预算 10,000,000,000,000,000 FLOPs；训练 190,054,400 token，2,900 步；每 token 52,607,520 FLOPs。

非词表总参数 19,483,056，激活非词表参数 2,894,256（Na/N=14.9%）；D/Na=66。

全局 batch=65536 token（8 条序列），seq=8192；micro_bs 从 8 起自动探测（auto_micro_bs=1），梯度累积 = 8/(卡数×micro_bs)；warmup=145 步；cosine 降到峰值的 10%。

峰值学习率 0.003118（0.3118·C^-0.125，同一预算内统一）；model seed=0，data seed=0。所有预算/模型共用同一无放回打乱块序列。验证分区按完整文档冻结，见 ../../validation.json。

输出：`runs/1e16/h192_L5/`（日志、checkpoint、TensorBoard、final_eval.json、HF 导出）。

仅检查（无 GPU 分配，不训练）：

```bash
GPUS=0 bash scripts/train_isoflops.sh 1e16 h192_L5 --dry-run
```

以后启动：

```bash
GPUS=0 bash scripts/train_isoflops.sh 1e16 h192_L5
```

CLI 参数覆盖 JSON；修改预算、学习率或 seed 后须使用新 run_dir。
