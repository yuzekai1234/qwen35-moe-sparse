# 最新 dense C1–C4 顺序训练

`scripts/run_dense_levels.py` 固定读取 `configs/dense_isoflops/{c1,c2,c3,c4}.json`，先校验其与最新 17 点拟合结果一致，再依次执行八卡训练、完整预算验收、BF16 HF 导出校验和完整 A 集单卡评测。不读取历史 `train_dense.sh c1` 等别名。

四档均使用 dense SwiGLU、关闭 mHC、将原 QSA 换为因果全注意力，保留 GDN。训练采用现有 BF16 混合精度及 FP32 主权重/优化器状态，不启用 FP8 或量化。数据、预算、形状和优化器均取自规范配置；运行时只修改输出目录及 `resume=0`、`max_steps=0`。

## 执行

在已激活项目环境、配置好 CUDA 库路径和离线缓存的干净源码目录中运行：

```bash
python scripts/run_dense_levels.py \
  --run-root /data/q38/experiments/dense-c1-c4-run \
  --evaluation-jsonl /data/q38/datasets/evaluation-a-v1/dolma3_mix_100M_a.jsonl \
  --expected-commit "$(git rev-parse HEAD)" \
  --gpus 0,1,2,3,4,5,6,7
```

默认仅输出计划，不访问大数据、不创建结果目录或使用 GPU。确认计划后加 `--execute`，并用 tmux 保持远端执行。父目录须已存在，运行目录必须全新；同目录重复启动或任何失败均不自动续跑，不覆盖模型。每档训练上限 12 小时、评测上限 3 小时、资源等待上限 30 分钟，可用对应 `--*-timeout` 秒数参数调整。启动每阶段前重新核查所用卡；不停止其他作业。

## 指标与验收

- 训练验收包含实际步骤/token/FLOPs、八卡、固定最终 16M-token 纯 CE 分区、运行配置/数据身份和 HF 架构/权重精度。
- 评测使用 `q38.evaluate_single_gpu` 的官方样例适配口径：单卡、序列 4096、固定 batch16、完整词表 logits 前向及预热/同步计时。完整 A 原文 SHA256 固定为 `4689036c6d9b413b03b68f0042509e0f3ea7b4d98cb210da4ef5b157a18c9f5c`；计分 token 为 120,142,976、UTF-8 文本字节为 445,910,984，EOS/padding 不计。
- 每档评测一遍，报告 BPB、forward 秒数、EvalTime；不是多次运行统计量。batch16 是固定比较口径，不声称是每档最大稳定 batch。H100 计时不等同于 4090 或官方 H200 成绩。
- `EvalTime = forward_seconds / [(Na/1e9) × (scored_tokens/1e6)]`，Na 排除全部 Embedding、词表输出头和未使用的 MTP。

`plan.json` 绑定源码 commit、规范配置 SHA 和数据身份，`environment.json` 记录安全白名单运行环境；每档保存运行配置、训练/评测命令、日志、权重、验收和原始评测 JSON。`summary.json` 原子更新，`completion.json` 仅在成功或失败终态产生，可供外部监控器通知；该文件本身不等于通知已送达。目录 755、普通输出文件 644，凭据不进入结果目录。完成要求四档均通过训练与完整 A 评测，不能用短测或最终 CE 替代 BPB。
