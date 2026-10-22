# IsoFLOPs 配置与运行约定

本目录位于服务器 `/home/yzk/qwen35-moe-sparse/configs/isoflops`。

目录结构：

```text
configs/isoflops/
  3e15/M1/config.json       主配置（峰值 lr=0.004）
  3e15/M1/README.md         参数、预算与使用说明
  3e15/M1/lr_2e-3.json      最低预算的学习率候选
  3e15/M1/lr_8e-3.json
  ... M2–M7
  1e16/M1 ... M7
  3e16/M1 ... M7
  1e17/M1 ... M7
  validation.json          按完整文档冻结的验证分区
  manifest.json            28 个主实验的参数量、token、步数和输出路径
```

共 28 个主配置、14 个额外学习率候选，初轮均为 model seed=0、data seed=0。
0.004 只是尚未校准的初始峰值学习率，最低预算的 0.002/0.008 候选用于后续校准。
选出学习率后应在更高预算训练前修改同一模型的配置；已有训练应换输出目录。
C1=3e17 的预测模型要等四档结果拟合后再确定，本次没有预先指定 C1 最优结构。

在仓库根目录，仅检查配置（CPU，无模型训练、无输出文件创建）：

```bash
GPUS=0 bash scripts/train_isoflops.sh 3e15 M1 --dry-run
GPUS=0 VARIANT=lr_2e-3 bash scripts/train_isoflops.sh 3e15 M1 --dry-run
# 也可直接指定配置文件或目录
.venv_q35/bin/python -m q35.train --config configs/isoflops/3e15/M1 --dry-run
```

未来启动单个主实验：

```bash
GPUS=0 bash scripts/train_isoflops.sh 3e15 M1
# 两卡共同训练同一个实验，仍保持全局 batch=65536
GPUS=0,1 bash scripts/train_isoflops.sh 3e15 M1
```

本次配置准备不启动训练，也不安装自动调度。已有输出目录为空。

输出与配置一一对应：

```text
runs/3e15/M1/
  train.log
  run_config.json
  training_metrics.csv
  ckpt.pt
  tb/
  final_eval.json
  summary.json
  flops.json
  hf/
  lr_2e-3/                 该学习率候选的独立结果
  lr_8e-3/
```

JSON 的 model/train 字段可由 CLI 覆盖。更改 lr、seed、数据或预算时使用新的
`--run_dir`；默认 resume=1，断点续训核对超参数、数据身份与采样顺序。

训练数据指向共享的 `isoflops-gzy10k-16b-v1/train.bin`（gzy 10K 词表，约 16B token；最大配置 1e17/M1 用约 9.6B），tokenizer 使用该池的冻结快照。
早先基于 50K 词表（`isoflops-gzy50k-16b-v1`）的一轮结果已清除，本轮全部按 10K 重新生成并重跑。
新采样器把数据划为不重叠的 8192-target 块，再用 data seed=0 全局打乱一次。
不循环、不放回；相同数据 seed 下所有模型读取相同序列前缀。
各 rank/micro batch 分割同一全局 batch，改变卡数或 micro batch 不改变全局数据块顺序。
块间允许共享一个预测上下文 token，预测 target 不重复。恢复位置由已完成全局步数确定。

全局 batch=65536、seq=8192；micro_bs=8 为每卡上限，多卡时取 8/卡数（A100-40G 上 M7 单卡 micro_bs=8 峰值约 28G）。显存不够时加 `--auto_micro_bs 1`：训练前从 micro_bs 往小探测能放下的最大值（显存 ≤ probe_mem_frac=0.92），结果写入 `runs/<预算>/<模型>/micro_bs_probe.json`，卡型/卡数/模型不变时续训直接复用。
允许 1/2/4/8 卡，梯度累积 = 8/(卡数×micro_bs)，默认均为 1。预算按实际 10112 词表和每 token FLOPs 向下取整。
warmup 约占 5%（至少 10 步），完整 cosine 周期在最后一步降至峰值的 10%。
`--max_steps` 只是短跑截止点，不改变完整周期，不能用其中间结果代替低预算正式实验。

验证使用同一独立留出文件，按文档边界分成三个连续、互不共享文档的区域：

- dev：约 2M target，用于监控，默认每 100 步与本次运行结束时评测。
- curve：紧接 dev 的约 16M target，写入 `final_eval.json`，用于 IsoFLOPs 拟合。
- report：剩余部分，保留给之后的最终报告，本次训练入口不使用。

精确 offset、target 数、文档数和文件指纹见 validation.json。
固定顺序逐段评分，包含末尾不足 8192 的窗口；主指标是纯 CE，不含路由辅助损失。
final_eval 在 FP32 主权重/BF16 前向下执行，然后导出 BF16 HF 模型。
多卡最终评测分摊固定窗口并汇总 NLL/token 数。短跑结果会标记 completed_budget=false。
训练 FLOPs 不包含验证、编译和优化器开销，实际耗时分别记录。

配置生成器：`scripts/make_isoflops_configs.py`。它不会启动训练，默认拒绝覆盖手改过的 JSON。
`--force` 明确要求重新生成配置；冻结验证分区仍复用同一身份文件。
CPU 检查：`CUDA_VISIBLE_DEVICES='' .venv_q35/bin/python -m unittest tests.test_presets tests.test_isoflops -v`。
