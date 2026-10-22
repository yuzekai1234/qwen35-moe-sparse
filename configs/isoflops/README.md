# IsoFLOPs 配置与运行约定

本目录位于服务器 `/home/yzk/qwen35-moe-sparse/configs/isoflops`。
目的：为比赛四档（3e17 / 1e18 / 3e18 / 1e19）确定 N、D 配比。扫描预算 1e16 / 3e16 / 1e17 / 3e17，
拟合 N_opt(C)、D_opt(C) 后按同一缩放规则外推；3e17 档的最优扫描点本身即可作为该档提交模型。

目录结构（形状以 hidden 和层数命名，同名形状在各预算中结构相同；每个预算 6 个）：

```text
configs/isoflops/
  1e16/h128_L4 h192_L4 h192_L5 h256_L5 h320_L5 h384_L6
  3e16/h192_L4 h192_L6 h256_L6 h320_L6 h320_L9 h384_L10
  1e17/h256_L4 h256_L7 h384_L5 h448_L6 h448_L9 h512_L11
  3e17/h256_L8 h320_L8 h384_L9 h512_L8 h576_L10 h640_L12
  <预算>/<形状>/config.json, README.md
  validation.json          按完整文档冻结的验证分区
  manifest.json            24 个实验的参数量、Na/N、token、D/Na、步数、lr、batch 和输出路径
```

## 缩放规则（`scripts/make_isoflops_configs.py:shape_for`）

层内比例只由 hidden 决定，扫描与四档提交模型共用。仿照 Chinchilla（Table A9：ffw 固定 4×d_model、头维固定，
d_model 与层数同时变化），扫描在宽深比 hidden/层数 ∈ [32, 80] 内同时改变 hidden 和层数，使相邻规模更细；
四档提交模型取默认层数 max(4, round(hidden/51.2))：

| 维度 | 规则 | 与 Qwen3.5-35B-A3B 原版 |
|---|---|---|
| 层型 | 每 4 层 `[lin, lin, lin, full]`，余下接 lin | 同 |
| 层数 | 扫描：hidden/层数 ∈ [32, 80]，至少 4 层；提交：max(4, round(hidden / 51.2)) | 原版宽深比 2048/40 = 51.2 |
| 全注意力 | q 头 hidden/32、kv 头 ≈ hidden/256（整除 q），头维 64（q 宽 2h、kv 宽 h/4） | 宽度比例同；头维 256→64 |
| GDN | k 头 hidden/64、v 头 hidden/32，头维 64（k 宽 h、v 宽 2h） | 宽度比例同；头维 128→64 |
| MoE | 64 专家 top-4，专家宽 h/2；共享专家宽 h/2（sigmoid 门） | 原版 256 选 8、宽 h/4：为满足 Na/N≥10% 并兼顾小规模 GPU 效率而调整 |
| 其余 | 输出门、partial_rotary 0.25、rope_theta 1e7、aux 0.001、绑定嵌入 | 同 |

非词表口径 Na/N 在所有规模约 14.5%–15.0%（比赛要求 ≥10%），生成器对每个形状断言。

每个预算取 6 个形状：以 D/Na（训练 token / 激活非词表参数）= 40 为中心，在 log 空间均匀取目标 ≈304、135、60、27、12、5，
各取最接近的形状（接近程度相同时偏向宽深比 51.2），覆盖最优点两侧，供每个预算拟合抛物线找最低点。

## 学习率与 batch

按 DeepSeek LLM（2024）拟合的计算最优超参随预算缩放，同一预算内所有形状一致：

| 预算 | 峰值 lr = 0.3118·C^-0.125 | 全局 batch = 0.2920·C^0.3271，取 2^k 条序列 |
|---|---|---|
| 1e16 | 0.003118 | 65536（8 × 8192） |
| 3e16 | 0.002718 | 65536 |
| 1e17 | 0.002338 | 131072（16 × 8192） |
| 3e17 | 0.002038 | 131072 |

model seed=0、data seed=0。

在仓库根目录，仅检查配置（CPU，无模型训练、无输出文件创建）：

```bash
GPUS=0 bash scripts/train_isoflops.sh 1e16 h256_L5 --dry-run
# 也可直接指定配置文件或目录
.venv_q35/bin/python -m q35.train --config configs/isoflops/1e16/h256_L5 --dry-run
```

启动：

```bash
bash scripts/run_isoflops_all.sh            # 全部 24 个，8 卡队列，小预算/小模型优先
GPUS=0 bash scripts/train_isoflops.sh 1e16 h256_L5
GPUS=0,1 bash scripts/train_isoflops.sh 1e16 h256_L5   # 两卡训练同一实验，全局 batch 不变
```

输出与配置一一对应：

```text
runs/1e16/h256_L5/
  train.log
  run_config.json
  training_metrics.csv
  ckpt.pt
  tb/
  final_eval.json
  summary.json
  flops.json
  hf/
```

JSON 的 model/train 字段可由 CLI 覆盖。更改 lr、seed、数据或预算时使用新的
`--run_dir`；默认 resume=1，断点续训核对超参数、数据身份与采样顺序。

训练数据指向共享的 `isoflops-gzy10k-16b-v1/train.bin`（gzy 10K 词表，约 16B token；最大配置 3e17/h256_L8 用约 2.4B），tokenizer 使用该池的冻结快照。
早先 50K 词表及旧形状网格（3e15–1e17、M1–M7）的结果均已清除。
新采样器把数据划为不重叠的 8192-target 块，再用 data seed=0 全局打乱一次。
不循环、不放回；相同数据 seed 下所有模型读取相同序列前缀。
各 rank/micro batch 分割同一全局 batch，改变卡数或 micro batch 不改变全局数据块顺序。
块间允许共享一个预测上下文 token，预测 target 不重复。恢复位置由已完成全局步数确定。

seq=8192；micro_bs=8 为每卡上限，不超过 batch 序列数/卡数。配置默认 auto_micro_bs=1：训练前从该上限往小探测能放下的最大值（显存 ≤ probe_mem_frac=0.92），结果写入 `runs/<预算>/<形状>/micro_bs_probe.json`，卡型/卡数/模型不变时续训直接复用。
允许 1/2/4/8 卡，梯度累积 = 序列数/(卡数×micro_bs)。预算按实际 10112 词表和每 token FLOPs 向下取整。
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
