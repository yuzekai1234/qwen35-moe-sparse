# Q38 sparse

dense 实现、参数口径、配置生成与验证结果见 [Q38 dense 基线与 IsoFLOPs 验证](docs/experiments/q38_dense.md)。固定形状的 C1–C4 配置保持每 token 激活参数近似不变，只替换 MoE 前馈层；四档拟合后外推的 C1 另选层数和隐藏宽度。

同一 dense checkpoint 的 BF16 / INT8 / FP8 推理质量对照见 [低精度 BPB 实验](docs/experiments/q38_low_precision.md)。入口为 `python -m q38.compare_precision`，默认只量化主干 FFN，关闭 EvalTime 计时。

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

默认训练 seq=4096，使用 GZY 10K tokenizer（实际 10112 token）及共享 16B token 训练池，无需重新分词。
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
