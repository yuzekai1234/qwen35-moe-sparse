# 1e17 / h256_L7

模型：7 层 × hidden 256（宽深比 37）；全注意力 q/kv 头 8/1，GDN k/v 头 4/8（头维均 64）；64 专家 top-4，专家/共享专家宽 128；GDN/full attention=3:1；词表 gzy 10K（10112）。层内比例由 shape_for 统一生成。

预算 100,000,000,000,000,000 FLOPs；训练 1,072,562,176 token，8,183 步；每 token 93,229,632 FLOPs。

非词表总参数 48,470,112，激活非词表参数 7,182,432（Na/N=14.8%）；D/Na=149。

全局 batch=131072 token（16 条序列），seq=8192；micro_bs 从 8 起自动探测（auto_micro_bs=1），梯度累积 = 16/(卡数×micro_bs)；warmup=409 步；cosine 降到峰值的 10%。

峰值学习率 0.002338（0.3118·C^-0.125，同一预算内统一）；model seed=0，data seed=0。所有预算/模型共用同一无放回打乱块序列。验证分区按完整文档冻结，见 ../../validation.json。

输出：`runs/1e17/h256_L7/`（日志、checkpoint、TensorBoard、final_eval.json、HF 导出）。

仅检查（无 GPU 分配，不训练）：

```bash
GPUS=0 bash scripts/train_isoflops.sh 1e17 h256_L7 --dry-run
```

以后启动：

```bash
GPUS=0 bash scripts/train_isoflops.sh 1e17 h256_L7
```

CLI 参数覆盖 JSON；修改预算、学习率或 seed 后须使用新 run_dir。
