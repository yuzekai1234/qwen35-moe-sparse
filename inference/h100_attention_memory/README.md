# C4 attention 显存修复

这份独立代码基于 [冻结的 C1/C4 fullcompile baseline](../h100_fullcompile/README.md)，解决 C4 的 batch 被 attention 中间张量限制的问题。根据用户最新指示，它已选为新的 C4 baseline（batch940、attention tile512）；C1 继续使用原 baseline。原版本的目录和 Git tag 作为历史对照保留，模型结构、checkpoint 权重和依赖均保持原样。当前选择见 [../BASELINES.json](../BASELINES.json)。

## 根因及修改

C4 有 44 个 query heads；旧代码对整个评测 batch 一次性做 Q/gate 投影，并同时保留原始 Q+gate、归一化/RoPE 后的 Q、复制出的 gate。batch768 三块分别占 33、16.5、16.5 GiB，共 66 GiB。加上 residual、FP8 input、权重等，峰值 allocated 为 78.500668 GiB，769 就 OOM。这是当前实现的边界，不是模型架构的上限。

新代码沿 **batch 维**分块，每块完成 Q/K/V 投影、norm/RoPE、cuDNN attention、gate/FP8 quant 和输出投影，再拼接回 hidden states。默认每块最多 512 个序列，因此这三块大张量每块最多合计 44 GiB。完整序列长度仍是 4096，序列之间没有 attention 依赖，计算公式和计分 token 不变。

只有 `compiled_attention.py`、`full_compile.py` 两个 Python 文件改变；`release-defaults.json` 的默认 batch 从 C1 的 989 改为 `auto`，避免 C4 误用 C1 的默认值。相对服务器上的原 C4 代码，JSON 默认值已经是 `auto`，因此实际运行代码只有这两个 Python 文件不同。没有将整个 attention 包成 eager opaque op；分块循环被展开进 `fullgraph=True, dynamic=False` 的完整前向图，仍禁止静默 eager fallback。

## 完整 A 集结果

单卡 H100 80GB、seq4096、120,142,976 个计分 token、445,910,984 UTF-8 字节；计时排除编译、预热和 cross entropy。

| 路径 | GPU | Batch | BPB | 累计 forward 秒 | EvalTime | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冻结 baseline | 0 | 768 | 0.9383547679682429 | 261.69925671386716 | 5.764464921254351 | 78.500668 |
| 分块修复，同 batch 对照 | 1 | 768 | 0.9383533273806052 | 250.05332214355468 | 5.507939235439393 | 64.389340 |
| 新 C4 baseline，接近显存上限 | 0 | 940 | 0.9383550647909364 | 275.1607272949219 | 6.060981525571003 | 78.427854 |

940 完成完整评测，941 在最终完整 BF16 logits 分配时 OOM；batch 容量提高 22.3958%。900 也通过容量检查。BPB 相对冻结 baseline 的差异为 +0.0000002968（940）和 -0.0000014406（768），均在此次 1e-5 数值门限内。

**容量修复不等于速度收益。** 同一张 GPU0 上，940 的 forward 比旧 baseline 的 768 慢 13.461 秒（5.1439%）。修复后的 768 虽测得更快，但在 GPU1 上运行，不能据此严格归因 4.45% 的收益。按用户指示采用 batch940 作为新 C4 baseline，同时保留这一比较结论，不将它标为 EvalTime 改进。940 更慢的组件归因尚未通过完整模型 profile 确认。

attention 单项的分块粒度测试中，512/256/128 的中位数分别为 335.302/337.657/339.635 ms，未显示更小分块更快，因此保留 512。这是独立 attention 测试，不是完整模型 EvalTime。原始数据见 [attention-tile-bench.json](attention-tile-bench.json)。

## 数值与完整性检查

- 比较原版编译 attention 与分块编译 attention；seq4096 的相对 RMSE 为 9.15e-5，另含不同 batch 和尾块检查。不是逐 bit 一致。
- 原版 compile 与 eager 也存在更大的输出差异；本次修改没有消除该既有差异，也不能将旧 C4 BF16 评测与当前 FP8 路径的 BPB 差异归因于本次分块。
- 两组完整 A 评测均 materialize 所有 BF16 logits，scored tokens、数据 SHA、参数核算口径一致，整图编译与无 eager fallback 证据保存在 metadata 中。
- 原 C1 的 41 个受保护文件、原 C4 checkpoint 完整 SHA256、49 个依赖源码校验通过。
- [evaluation-source-sha256.json](evaluation-source-sha256.json) 锁定此次实测代码；[results.json](results.json) 包含完整结果、容量探测、比较、来源与完整性记录；[evidence/](evidence/) 保留两组输出。

## 复现

依赖与 FlashQLA D64 补丁沿用 [原 baseline 的环境说明](../h100_fullcompile/README.md)。推荐通过 [run_baseline.sh](run_baseline.sh) 固定 batch940、tile512、seq4096 和单卡；dataset/model 使用绝对路径，输出目录必须是新目录：

```bash
CUDA_VISIBLE_DEVICES=0 Q38_PYTHON=/path/to/qualified/python \
  bash /path/to/qwen35-moe-sparse/inference/h100_attention_memory/run_baseline.sh \
  /path/to/evaluation-a.jsonl /path/to/C4/hf /path/to/new-output
```

等价的底层入口：

```bash
REPO=/path/to/qwen35-moe-sparse
PY=/data/q38/environments/q38-flashqla-torch280-cu128-py312/bin/python
DATA=/data/q38/datasets/evaluation-a-v1/dolma3_mix_100M_a.jsonl
MODEL=/data/q38/experiments/dense-c1-c4-20261030/c4/train/hf
mkdir /path/to/new-c4-memory-eval
cd /path/to/new-c4-memory-eval
CUDA_VISIBLE_DEVICES=0 Q38_PYTHON="$PY" EVAL_BATCH_SIZE=940 \
  Q38_ATTENTION_BATCH_LIMIT=512 \
  bash "$REPO/inference/h100_attention_memory/evaluation/eval.sh" "$DATA" "$MODEL"
```

`Q38_ATTENTION_BATCH_LIMIT` 只控制 attention 内部块大小；`EVAL_BATCH_SIZE` 是评测 batch。设 `EVAL_BATCH_SIZE=auto` 会重新搜索当前机器的容量并完整评测。设 `Q38_ATTENTION_BATCH_LIMIT=0` 可用于未分块路径对照，大 batch 会恢复原有的 attention 峰值问题。

服务器工作目录：`/data/q38/experiments/c4-attention-memory-20261031`。新的 C4 指针为 `/data/q38/experiments/c4-active-baseline`，记录为 `/data/q38/experiments/C4_ACTIVE_BASELINE.json`；C1 原 active baseline 指针保持原样。
