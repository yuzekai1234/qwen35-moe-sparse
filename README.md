# Q38 sparse

从 `ef25716ff031345d156abf791010ed2d0a5c381c` 创建的 `q38` 版本，破坏式替换原 Q35 代码。
架构参考 [hzh-q38f-sparse](https://github.com/huangzhenhua111/hzh-q38f-sparse)，固定参考提交 `12862f6ded28d4cedf7de411d3a632d660ade140`。

模型包含每四层三层 Gated DeltaNet、一层 QSA；QSA 使用 64-token 块、默认选 8 块、4 头 MQA indexer 和 KL 蒸馏。
MoE 默认 256 专家选 5，加一个共享专家，专家宽 h/4；其余组件为第二层前注入的 bigram/trigram 哈希嵌入、4 路 mHC/Sinkhorn 残差、训练专用一层 MTP。
训练使用 Muon + AdamW，损失为主 CE + 0.001 router auxiliary + 0.3 MTP CE + 0.05 indexer KL。保留 DDP、断点恢复、数据采样和 TensorBoard。

Q38 是这里的本地缩小版实现，继承的 GDN/MoE 基础代码来自 Transformers 5.19 的 Qwen3.5。
QSA 的块选择使用每块首个 query 的分数，避免参考实现中块内后续 query 影响前面 token 的选择。
N-gram 不跨 EOS；序列开头不足的前文使用独立占位 ID。模型只支持无 cache 的 packed prefill；评估用主头，MTP 不参与评估。

## 使用

```bash
# 环境；已有 .venv_q35 可直接复用，脚本会自动回退
bash scripts/setup_env.sh

# 单卡；GPUS=0,1 可用 DDP
GPUS=0 bash scripts/train.sh q38_c2_m --run_dir runs/q38_c2_m

# 模型参数及预算
.venv_q35/bin/python -m q38.flops q38_c2_m --seq 2048

# 评估 HF 导出
GPUS=0 bash scripts/eval.sh runs/q38_c2_m/hf /path/to/eval.jsonl 2048 1
```

默认训练 seq=2048，沿用现有 GZY tokenizer 和 token 数据，无需重新分词。参考仓库的 16k tokenizer 未替换进来。
入口为 `python -m q38.train`、`python -m q38.evaluate`、`python -m q38.data`。
配置新增 `n_branch`、`mhc_iters`、`qsa_blocks`、`indexer_heads`、`indexer_head_dim`、`ngram_rows`、`mtp`；`ngram_rows=-1` 自动按非词表参数规模选择行数，`0` 关闭哈希表。
训练参数可设置 `--muon 0`、`--mtp_coef`、`--kl_coef` 和 `--dense_warmup`。QSA 在学习率 warmup 阶段使用 dense attention。

HF 导出包含全部自定义模型代码、N-gram 表和 MTP 权重；可用 `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` 加载，前向请传 `use_cache=False`。
旧 Q35 checkpoint 不兼容。IsoFLOPs 配置逐字保留指定提交的 Q35 内容，目录改为 `configs/q35_isoflops/`，仅作 Q35 历史配置；请在原 Q35 提交上运行或重新生成。

## 核算与验证

`q38/flops.py` 在原 Megatron GDN 约定上计入 QSA 实际选块、indexer 投影/打分、KL 目标、mHC、N-gram 门和训练 MTP；优化器与 dense warmup 开销不计。
非词表总参数包含完整 N-gram 表与 MTP；激活参数采用主头评估口径，只计每 token 访问的两条 N-gram 行，排除训练专用 MTP。
参考的 256 选 5 配置不承诺满足比赛 10% 激活比例，生成器不再按旧 Q35 比例筛选。
`audits/` 与 `scripts/audit_flops.py` 仅保留继承的 Q35 backbone 核算历史，不代表 Q38 全模型核算。

快速检查：

```bash
.venv_q35/bin/python -m unittest tests.test_model tests.test_presets tests.test_isoflops -q
```

模型 smoke 覆盖小模型前后向、Muon 更新和自定义 HF 导出加载；不进行完整训练或评测。
