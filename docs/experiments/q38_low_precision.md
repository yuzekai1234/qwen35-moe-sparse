# Q38 dense 低精度 BPB 对照

复用同一个已训练的 BF16 dense checkpoint，比较主干 FFN 的 INT8、FP8 训练后量化。训练入口保持原有 BF16 混合精度；本实验不重新训练、不修改来源 checkpoint。默认在同一次运行中完成三种精度，固定文本、packing、batch 和序列长度，直接报告 BPB 以及相对本次 BF16 的差值。EvalTime 与 forward_seconds 均为 null，不作速度结论。

## 完整 A 集实测

同一 h512/L10 checkpoint、Torch 2.7.1+cu126、batch16、seq4096。三种方案分别使用一张同型号 RTX 4090（SM 8.9），均完成全部 120,142,976 个计分 token / 445,910,984 原文 UTF-8 bytes，进程退出码均为 0。

| 方案 | BPB ↓ | 相对 BF16 增量 | BPB 相对增幅 |
|---|---:|---:|---:|
| BF16 | 1.133234624 | 0 | 0 |
| INT8 W8A8 FFN | 1.133429785 | +0.000195161 | +0.01722% |
| FP8 W8A8 FFN | 1.133996058 | +0.000761433 | +0.06719% |

结果使用原生 INT8 / FP8 矩阵乘法，不是 reference 后端。当前 FFN 量化在完整 A 集上的 BPB 损失较小；结论限于所记录的模型、量化范围、动态缩放规则与 batch。没有评估 EvalTime。

由于仅比较 BPB，本次三个单精度任务在独立 GPU 上并行运行；汇总前逐项验证权重、配置、tokenizer、数据与实现 SHA256、packing、batch、seq、设备型号/架构一致，重新核算总 NLL / UTF-8 分母，并核对当前实现仍匹配记录的哈希。原始单精度结果没有本次 BF16 基线时，其差值字段为 null；汇总中的差值在上述一致性检查通过后计算。

完整精度数值、设备标识和来源文件见 [结果 JSON](q38_low_precision_results.json) 的 full 字段。原始执行产物位于 `runs/q38_precision_20261029_full/{bf16,int8,fp8}/`，日志为该根目录下的各精度 `.log`，合并结果为 `comparison.json`。

## 量化范围与计算

| 方案 | 主干 FFN 权重 | FFN 输入激活 | GEMM / 输出 |
|---|---|---|---|
| bf16 | 原始 BF16 | 原始 BF16 | 原评测前向 |
| int8 | 对称 INT8，每输出通道缩放 | 动态对称 INT8，每 token 缩放 | 原生 INT8 → INT32，FP32 反缩放，再转 BF16 |
| fp8 | E4M3FN，每张量缩放 | 动态 E4M3FN，每张量缩放 | 原生 FP8，use_fast_accum=False，输出 BF16 |

只替换 `model.layers.*.mlp.{gate_proj,up_proj,down_proj}`。attention/GDN、mHC、归一化、N-gram/词表 embedding、绑定的输出头和训练专用 MTP 均保留原路径与精度。这不是全模型 INT8/FP8。结果 JSON 记录全部替换名称、量化权重元素数和原始总权重元素数；不按量化后的 buffer 数重新定义激活参数量。

FP8 每张量动态缩放取当前整个 linear 输入的绝对最大值，batch 或序列长度会影响量化误差，因此比较时必须保持一致。两种量化都使用 absmax 缩放，不使用校准集，也不根据 A 集拟合缩放参数。

RTX 4090 为 Ada SM 8.9，支持 FP8；参见 [NVIDIA 支持矩阵](https://nvidia.github.io/TransformerEngine/support_matrix.html)。本实现直接使用现有 Torch 的 `torch._int_mm` / `torch._scaled_mm`，不安装或升级 TorchAO。原生 FP8 要求 SM 8.9 或更新架构，线性层 K/N 为 16 的倍数；矩阵短行会 padding，之后裁回原行数。内核失败会报错，不自动改成 BF16。

显式指定 `--backend reference` 可做 Q/DQ 后的 FP32 矩阵乘法，检查量化误差；结果会标记 reference，不能当作原生 INT8/FP8 GEMM 的速度或完全相同数值。该选项不改变量化格式和缩放规则。

## 运行

队友完成训练的外推 C1 为 h512/L10，权重位于：

`/home/shared/qwen35_moe_sparse/experiments/q38-4090-evaluation-20261028/models/dense/`

该权重 SHA256 为 `4e1a73cd826f2eb7a5ccbc7266d9adb0dea9d1f3bae4253cbcd19bc6427e5b78`，与 [dense 结果记录](q38_dense_results.json)一致。`scripts/train_dense.sh c1` 的默认 h448/L9 配置与此 checkpoint 不同，做推理对照应直接使用上述导出。

先用前 256 篇非空 JSONL 记录做 pilot；这不是完整 A 集成绩：

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0 "$PY" -m q38.compare_precision \
  --model-path /home/shared/qwen35_moe_sparse/experiments/q38-4090-evaluation-20261028/models/dense \
  --dataset-path /home/shared/data/dolma3_mix_100M_a.jsonl \
  --output-dir runs/q38_precision_pilot --batch-size 16 --max-documents 256
```

完整 A 集：

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m q38.compare_precision \
  --model-path /home/shared/qwen35_moe_sparse/experiments/q38-4090-evaluation-20261028/models/dense \
  --dataset-path /home/shared/data/dolma3_mix_100M_a.jsonl \
  --output-dir runs/q38_precision_full --batch-size 16
```

输出目录必须是新目录，默认依次执行 bf16、int8、fp8；可用 `--precisions bf16 fp8` 选择子集。每种精度重新加载原始 BF16 权重，避免重复量化；文本仅分词和 packing 一次。输出 `experiment.json`、各精度 JSON 和 `comparison.json`，包含权重、配置、tokenizer、数据、评测实现的 SHA256，以及设备、量化范围、原文 bytes、计分 token 数和总 NLL。

BPB 沿用单卡评测口径：完整分词后按文档插入 EOS、顺序 packing，EOS 和 padding 不计分；分母为对应原文 UTF-8 字节数。`--max-documents` 同时限制原文与分母，绝不把子集 NLL 除以全量 bytes。只有同次比较包含 BF16，才计算 BPB 增量；不直接减历史评测的 BPB。

## 检查

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "$PY" -m unittest \
  tests.test_quantization tests.test_evaluate_single_gpu tests.test_dense -v
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 "$PY" -m unittest \
  tests.test_quantization.Quantization.test_native_matches_qdq_reference_including_tail_rows -v
```

CPU 检查覆盖零值/极值缩放、误差、替换范围、真实 Q38 小模型前向、绑定权重、禁用计时后的完整计分、子集 UTF-8 分母与结果差值。GPU 检查对照原生 INT8/FP8 与 Q/DQ 参考计算，包括短行、尾行和全零输入。

## 256 篇 pilot 实测

单张 RTX 4090、Torch 2.7.1+cu126、原生低精度 GEMM、batch16、seq4096。相同的 571,351 个计分 token / 2,066,533 原文 UTF-8 bytes；这是确定性的前 256 篇子集，不是完整 A 集成绩。

| 方案 | BPB ↓ | 相对同次 BF16 增量 |
|---|---:|---:|
| BF16 | 1.149023329 | 0 |
| INT8 W8A8 FFN | 1.149231722 | +0.000208393 |
| FP8 W8A8 FFN | 1.149833388 | +0.000810059 |

两种量化都覆盖 30 个主干 FFN 投影、11,796,480 个权重元素；原模型共 108,747,936 个权重元素（含 embedding 和 MTP）。实际量化范围不能解读成全模型 8 bit。原始结果和各项身份哈希见 [结果 JSON](q38_low_precision_results.json) 的 pilot 字段；执行输出位于 `runs/q38_precision_20261029_pilot/`。

当前相关 CPU 检查 19 项通过、1 项 GPU 检查跳过；该 GPU 检查另在 RTX 4090 上独立通过。训练和 HF 导出回归也包含在这 19 项 CPU 检查内。
