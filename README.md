# Q38 dense baseline

当前主线为 **dense FFN + 无 mHC + 因果全注意力**，入口配置是 [`configs/baseline.json`](configs/baseline.json)。
C1 固定 h512/L10/F768：8 层 Gated DeltaNet、2 层全注意力，使用单路 FP32 Pre-Norm 残差；保留 N-gram、GQA、Q/K norm、attention 输出门及训练专用 MTP。主路径不构造 mHC 或 indexer，也不计算 indexer KL。

```bash
# 训练新的 C1 baseline；默认允许同配置续训
GPUS=0 bash scripts/train.sh --run_dir /path/to/c1
# 同一配置的 dense 入口，支持 dry-run
GPUS=0 bash scripts/train_dense.sh baseline --dry-run
# 复核配置及精确预算
CUDA_VISIBLE_DEVICES='' .venv_q35/bin/python scripts/make_baseline_config.py --check
# GPU 短测、独立进程恢复及 portable HF 导出核验；输出目录必须是新的
GPUS=0 bash scripts/smoke_dense.sh runs/new-baseline-qualification baseline
```

训练预算为 C1=3e17 FLOPs，预留 0.5%；新结构按每 token 366,904,320 FLOPs 重算为 813,432,832 token、6206 步。评测激活参数量为 33,776,160，排除全部 Embedding、输出头和未调用的 MTP。
默认读取 `/home/shared/llm_pt_ablation/cache/ling-no-mtp-c1-global-pruned10k-scale-isoflops-v1/` 的现成训练/验证集及对应 10112-token tokenizer；可通过命令行覆盖数据路径。配置固化已验证的八卡设置：每卡 micro4、关闭自动 micro-batch 探测、BF16 混合精度。默认输出到 `runs/q38_baseline/c1/`，保留 `resume=1` 的同配置续训语义；从头复现应显式 `--resume 0` 并指定全新目录，不能从旧结构 checkpoint 跨配置恢复。

组合模型已在八张 A100 40GB 上完成完整 C1 预算；FP32 主权重/优化器、BF16 混合精度，不使用 INT8/FP8。同一张 4090 48GB、seq4096、batch16、完整 A 三次评测中位数：BPB **1.121291**、累计 forward **216.442 秒**、EvalTime **53.337563**；同次复测的 mHC + learned QSA dense 对照为 1.131918 / 380.524 秒 / 86.159052。BPB 降低 0.010627，forward 耗时下降 43.12%，EvalTime 下降 38.09%。这是固定 batch 的本地对照，不是官方 H200 成绩。准确训练命令、原始配置身份和共享产物见 [组合 C1 训练与评测](docs/experiments/q38_baseline.md)。

模型字段 `mhc_enabled=0`、`qsa_mode="full"` 明确选择新结构；旧配置和旧 HF 导出缺少这两个字段时仍按 mHC + learned QSA 解释。历史 `q38_c1`–`q38_c4` 与 `configs/dense/` 均保留原结构和预算；`train_dense.sh c1` 仍指历史 FFN-only 对照，主线请使用默认入口或显式 `baseline`。

## 模型与实验

dense 实现、参数口径、配置生成与历史验证结果见 [Q38 dense 基线与 IsoFLOPs 验证](docs/experiments/q38_dense.md)。当前 [`configs/dense_isoflops/c1.json`](configs/dense_isoflops/c1.json) 至 `c4.json` **全部使用无 mHC、因果全注意力和 BF16**，与当前 C1 使用同一 5.617B-token 训练池。新结构已完成 3e15、1e16、3e16、1e17 四档全部 17 点训练与 N/D 拟合；默认生成器绑定[原始结果](docs/experiments/q38_combined_isoflops_results.json)，C2–C4 更新为 **h704/h896/h1152**。C1 保留已验证的 h512；新拟合的自由选择为 h576，未另行训练，不能称 h512 已被新拟合验证为最优。C2–C4 仍是未训练的外推配方，不预测 CE/BPB 或保证最优性。历史 `configs/dense/q38_c1`–`q38_c4` 仅作保留 mHC/QSA 的 FFN-only 对照，不是当前推荐配置。

新结构的实测曲线、拟合敏感性、数据身份和复现接口见 [当前 dense IsoFLOPs 方法与结果](docs/experiments/q38_combined_isoflops.md)。这四份配置的数据位置为已鉴定的 `/data/q38/datasets/c1-pruned10k-v1/`；异机迁移须重新核验身份，不会静默回退旧路径。`selection.json` 绑定原始拟合和最终配置哈希；配置生成不代表 C2–C4 已训练。

八卡依次完整训练上述最新 C1–C4、逐档自动导出并评测完整 A 集的入口为 `scripts/run_dense_levels.py`，默认仅打印计划。配置身份、预算验收、单卡评测口径及终态结果格式见 [顺序训练说明](docs/experiments/dense_levels.md)。

同一 dense checkpoint 的 BF16 / INT8 / FP8 推理质量对照见 [低精度 BPB 实验](docs/experiments/q38_low_precision.md)。入口为 `python -m q38.compare_precision`，默认只量化主干 FFN，关闭 EvalTime 计时。

### 历史模型

从 `ef25716ff031345d156abf791010ed2d0a5c381c` 创建的 `q38` 版本，破坏式替换原 Q35 代码。
架构参考 [hzh-q38f-sparse](https://github.com/huangzhenhua111/hzh-q38f-sparse)，固定参考提交 `12862f6ded28d4cedf7de411d3a632d660ade140`。

模型包含每四层三层 Gated DeltaNet、一层 QSA；QSA 使用 64-token 块、默认选 8 块、4 头 MQA indexer 和 KL 蒸馏。
MoE 默认 64 专家选 5，加一个共享专家，专家宽 h/4；其余组件为第二层前注入的 bigram/trigram 哈希嵌入、4 路 mHC/Sinkhorn 残差、训练专用一层 MTP。
训练使用 Muon + AdamW，损失为主 CE + 0.001 router auxiliary + 0.3 MTP CE + 0.05 indexer KL。保留 DDP、断点恢复、数据采样和 TensorBoard。

Q38 是这里的本地缩小版实现，继承的 GDN/MoE 基础代码来自 Transformers 5.19 的 Qwen3.5。
QSA 的块选择使用每块首个 query 的分数，避免参考实现中块内后续 query 影响前面 token 的选择。
N-gram 不跨 EOS；序列开头不足的前文使用独立占位 ID。模型只支持无 cache 的 packed prefill；评估用主头，MTP 不参与评估。

## 使用

```bash
# 环境；已有 .venv_q35 可直接复用，脚本会自动回退
bash scripts/setup_env.sh

# 单卡；GPUS=0,1 可用 DDP
GPUS=0 bash scripts/train.sh q38_c2 --run_dir runs/q38_c2

# 模型参数及预算
.venv_q35/bin/python -m q38.flops q38_c2 --seq 4096

# 评估 HF 导出
GPUS=0 bash scripts/eval.sh runs/q38_c2/hf /path/to/eval.jsonl 4096 1
```

历史 MoE 预设、FFN-only 配置和旧 IsoFLOPs 实验使用 seq=4096、GZY 10K tokenizer（实际 10112 token）及共享 16B token 训练池，无需重新分词；当前组合 baseline 和当前 `configs/dense_isoflops/` 四配置统一使用前述 5.617B-token 训练池。
4096 是序列长度；`batch_tokens` 为每步全局 token 数，`micro_bs` 为每卡序列数。
训练和数据入口为 `python -m q38.train`、`python -m q38.data`。按官方单卡样例适配的评测入口为 `python -m q38.evaluate_single_gpu`；原多卡评测入口为 `python -m q38.evaluate`。
配置新增 `n_branch`、`mhc_iters`、`qsa_blocks`、`indexer_heads`、`indexer_head_dim`、`ngram_rows`、`mtp`；`ngram_rows=-1` 自动按非词表参数规模选择行数，`0` 关闭哈希表。
训练参数可设置 `--muon 0`、`--mtp_coef`、`--kl_coef` 和 `--dense_warmup`。QSA 在学习率 warmup 阶段使用 dense attention。

HF 导出包含全部自定义模型代码、N-gram 表和 MTP 权重；可用 `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` 加载，前向请传 `use_cache=False`。
旧 Q35 checkpoint 不兼容。IsoFLOPs 配置逐字保留指定提交的 Q35 内容，目录改为 `configs/q35_isoflops/`，仅作 Q35 历史配置；请在原 Q35 提交上运行或重新生成。

## 核算与验证

`q38/flops.py` 在原 Megatron GDN 约定上计入 QSA 实际选块、indexer 投影/打分、KL 目标、mHC、N-gram 门和训练 MTP；优化器与 dense warmup 开销不计。
Indexer 输入已 detach，投影的前反向矩阵计算按前向的 2 倍计；打分矩阵计算仍按 3 倍计。
历史核算字段 `total_nonvocab_params` 包含完整 N-gram 表与 MTP；`active_nonvocab_params` 采用主头评估口径，只计每 token 访问的两条 N-gram 行，排除训练专用 MTP。这两项仅排输入/输出词表。新增 `active_params_excluding_embeddings` 则排除全部 Embedding，也不含训练专用 MTP；dense 对照说明中的 \(N_a\) 使用后者。
C1–C4 预设采用同一 64 选 5 缩放规则，非词表激活比例按包含完整 N-gram 表和 MTP 的分母计算，均大于 10%。
原 C1–C4 MoE 预设的参数/数据配比依据 Q35 IsoFLOPs 结果外推，再按 Q38 的计算开销重算 token 预算；dense 四档拟合与独立 C1 结果另见上述说明。
预设显式固定 `tokens` 并预留 0.5% 预算，改模型形状或序列长度请重新运行生成器；`--tokens 0` 会按完整预算重算且不保留该预留。
四档参数、数据配比与拟合依据见 [配置推导](audits/q38_config_plan.md)，可用 `python scripts/make_q38_configs.py --check` 复核。
`audits/flops_audit*.json` 与 `scripts/audit_flops.py` 保留继承的 Q35 backbone 核算历史，不代表 Q38 全模型核算。

快速检查：

```bash
.venv_q35/bin/python -m unittest tests.test_model tests.test_presets tests.test_isoflops -q
```

模型 smoke 覆盖小模型前后向、Muon 更新和自定义 HF 导出加载；不进行完整训练或评测。

性能复现（使用一张 GPU，首次调用会触发编译；稳定吞吐与首次编译耗时分开记录）：

```bash
GPUS=0 bash -c 'source scripts/env.sh; "$PY" scripts/profile_q38.py --preset q38_c1 --seq 4096 --micro-bs 4 --output audits/q38_c1_local.json --profile'
```

性能及显存适用范围见 [GPU profiling 报告](audits/q38_performance.md)。C2–C4 的质量为外推起点，C4 大模型的训练显存需求还需结合目标 GPU 评估；普通 DDP 不分摊权重和优化器状态。
