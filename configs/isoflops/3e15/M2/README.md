# 3e15 / M2

模型：4 层 × hidden 96，64 专家 top-4，GDN/full attention=3:1。

预算 3,000,000,000,000,000 FLOPs；训练 73,793,536 token，1,126 步；每 token 40,646,664 FLOPs。

总参数 12,164,780，激活非词表参数 724,652。

全局 batch=65536 token（8 条序列），seq=8192；micro_bs=8（每卡取 min(8, 8/卡数)，A100-40G 上 M7 单卡峰值约 28G），梯度累积 = 8/(卡数×micro_bs)；显存不够时加 --auto_micro_bs 1 自动往小探测；warmup=56 步；cosine 降到峰值的 10%。

初始峰值学习率 0.004，尚未校准；model seed=0，data seed=0。所有预算/模型共用同一无放回打乱块序列。验证分区按完整文档冻结，见 ../../validation.json。

输出：`runs/3e15/M2/`（日志、checkpoint、TensorBoard、final_eval.json、HF 导出）。

仅检查（无 GPU 分配，不训练）：

```bash
GPUS=0 bash scripts/train_isoflops.sh 3e15 M2 --dry-run
```

以后启动：

```bash
GPUS=0 bash scripts/train_isoflops.sh 3e15 M2
```

CLI 参数覆盖 JSON；修改预算、学习率或 seed 后须使用新 run_dir。

最低预算另有 lr_2e-3.json、lr_8e-3.json；0.004 对应 config.json。

```bash
GPUS=0 VARIANT=lr_2e-3 bash scripts/train_isoflops.sh 3e15 M2 --dry-run
```
