# IsoFLOPs 配置与运行约定

本目录位于服务器 `/home/yzk/qwen35-moe-sparse/configs/isoflops`。
目的：为比赛四档（3e17 / 1e18 / 3e18 / 1e19）确定 N、D 配比。扫描预算 1e16 / 3e16 / 1e17 / 3e17，
拟合 N_opt(C)、D_opt(C) 后按同一缩放规则外推；3e17 档的最优扫描点本身即可作为该档提交模型。

目录结构（形状以 hidden 和层数命名，同名形状在各预算中结构相同；每个预算 4–6 个，共 19 个）：

```text
configs/isoflops/
  1e16/h128_L4 h192_L4 h256_L5 h320_L6
  3e16/h192_L4 h256_L5 h320_L6 h384_L8
  1e17/h256_L5 h320_L6 h384_L8 h448_L9 h512_L10
  3e17/h320_L6 h384_L8 h448_L9 h512_L10 h576_L11 h640_L12
  <预算>/<形状>/config.json, README.md
  validation.json          按完整文档冻结的验证分区
  manifest.json            19 个实验的参数量、Na/N、token、D/Na、步数、lr、batch 和输出路径
```

## 缩放规则（`scripts/make_isoflops_configs.py:shape_for`）

hidden 是唯一自由参数，层数和层内比例都由它决定；扫描与四档提交模型共用这一条单参数规则。
因此拟合出的 N_opt 就落在提交模型所属的同一族上，不混入宽深比的影响，3e17 扫描的最优点本身就是合规的提交模型
（若扫描在宽深比带内另选层数，最优点可能不在声明规则上，外推的 N_opt 也会混入宽深比效应）：

| 维度 | 规则 | 与 Qwen3.5-35B-A3B 原版 |
|---|---|---|
| 层型 | 每 4 层 `[lin, lin, lin, full]`，余下接 lin | 同 |
| 层数 | max(4, round(hidden / 51.2))，扫描与提交相同 | 原版宽深比 2048/40 = 51.2 |
| 全注意力 | q 头 hidden/32、kv 头 ≈ hidden/256（整除 q），头维 64（q 宽 2h、kv 宽 h/4） | 宽度比例同；头维 256→64 |
| GDN | k 头 hidden/64、v 头 hidden/32，头维 64（k 宽 h、v 宽 2h） | 宽度比例同；头维 128→64 |
| MoE | 64 专家 top-4，专家宽 h/2；共享专家宽 h/2（sigmoid 门） | 原版 256 选 8、宽 h/4：为满足 Na/N≥10% 并兼顾小规模 GPU 效率而调整 |
| 其余 | 输出门、partial_rotary 0.25、rope_theta 1e7、aux 0.001、绑定嵌入 | 同 |

非词表口径 Na/N 在所有规模约 14.5%–15.0%（比赛要求 ≥10%），生成器对每个形状断言。

hidden 以 64 为步长（GDN k 头数 hidden/64 须为整数），规则上相邻形状的 Na 约差 2 倍。
每个预算训练 D/Na（训练 token / 激活非词表参数）落在 [4, 400] 内的全部规则形状（4–6 个，D/Na 相邻约差 3 倍），
覆盖最优点两侧，供每个预算拟合抛物线找最低点；点较疏，拟合时也可把四个预算合并拟合 L(N, D)。

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

下面队列命令注释中的耗时是 FLOPs 校准前估计，须按更新后的 token 数重估。

```bash
bash scripts/run_isoflops_all.sh            # 全部 19 个：8 卡分 4 组、每组 2 卡 DDP；大预算先跑，同预算内小模型（最耗时）先跑；按实测吞吐估算约 2.5 小时（不含评测）
PER_JOB=1 bash scripts/run_isoflops_all.sh  # 每任务 1 卡（约 2.9 小时，受最长任务 3e17/h320_L6 限制）
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

训练数据指向共享的 `isoflops-gzy10k-16b-v1/train.bin`（gzy 10K 词表，约 16B token；最大配置 3e17/h320_L6 用约 2.65B），tokenizer 使用该池的冻结快照。
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

FLOPs 采用 `megatron_b92c3ed_plus_router_shared_gate_v1`：GDN 核心按比赛指定
Megatron 提交 `b92c3ed70029005fd5769d2cf4c65bc3d061a74b` 的 `8·nv·dv²` 前向公式，
全注意力按 causal 减半，前反向按参考约定乘 3，另计实际 router 与共享专家门投影。
参数量独立统计，norm 等参数不再当作 GEMM 权重。配置中分别记录原函数参考值及额外投影值。
19 个配置已按新口径重算 token、步数与 warmup；模型、lr、batch、数据与随机种子未改。
当前口径预算向下取整误差 <0.066%；若直接按未补充额外投影的 Megatron 原函数复核，
预算偏差为 -0.732% 至 -0.455%，也在 ±1% 内。详细交叉验证见 `audits/flops_audit.json`。
旧口径 checkpoint 不自动混用续训，历史实验数值保留原口径；此处采用模型 FLOPs 约定，非 FLA 物理指令计数。

配置生成器：`scripts/make_isoflops_configs.py`。它不会启动训练，默认拒绝覆盖手改过的 JSON。
`--force` 明确要求重新生成配置；冻结验证分区仍复用同一身份文件。
CPU 检查：`CUDA_VISIBLE_DEVICES='' .venv_q35/bin/python -m unittest tests.test_presets tests.test_isoflops -v`。
