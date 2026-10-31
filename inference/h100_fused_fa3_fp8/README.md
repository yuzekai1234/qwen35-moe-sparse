# C4 fusion + FA3 FP8 attention baseline

用户指定的 C4 新基线，版本标签 `baseline-c4-fused-fa3-fp8-h100-20261031`。当前选择见 [../BASELINES.json](../BASELINES.json)；C1 及所有旧 baseline 目录保持原样。权重、模型结构和计分规则沿用原 C4。

融合代码来自 teammate gzy 的 committed tree `c5fd8e8`，包含 trunk 两阶段分块256、GDN/attention 投影合并、conv/SiLU/L2 和 gated norm/FP8 quant 融合，以及 deferred scale。只在 SDPA 回调接入原生 FA3 E4M3 QK/PV；没有带入 teammate 工作树中未完成的改动，也没有叠加无明确收益的静态 scale 方案。

## 完整 A 集实测

以下均为同一张 **H100 80GB GPU1**、batch938、seq4096、120,142,976 scored tokens、445,910,984 UTF-8 bytes；不是官方 H200 成绩。

| 代码 | BPB | Forward 秒 | EvalTime |
|---|---:|---:|---:|
| 融合对照 | 0.9383548908748438 | 209.87386584472654 | 4.622903988117676 |
| 融合 + FA3 FP8 | 0.9387014573578255 | 206.3110681152344 | 4.544426032960023 |

前向减少3.562798秒（1.69759%），BPB增加0.0003465664829817。指标原始输出在 [evidence/](evidence/)，差值和源文件验证见 [evidence/root-full-a-verification.json](evidence/root-full-a-verification.json)。这是已完成实验的 canonical 分数；正式自动入口的独立发布复测另存 [release-validation.json](release-validation.json)，不把跨次运行的时钟波动算成新优化。

Fresh-process真实窗口包含全部 BF16 logits 和 CE 的容量检查中，938通过、939在最终 logits 分配时OOM。trunk按256分块编译为 `fullgraph=True, dynamic=False`，禁止 eager fallback；外层分块调度和原生 `_scaled_mm.out` 输出头仍包含在完整前向计时中。计时包含量化/布局转换、trunk、完整 logits 分配及输出头，排除编译、预热和CE。量化的动态 `amax`/cast 保留在 FX 中。默认安装的历史 cuDNN 路径元数据可能仍出现；实际执行路由以 `inference_paths.full_compile.external_ops` 和 `fp8_attention` 为准，必须显示 `q38fp8::fa3_fused_baseline_native`。

## 使用已验证二进制复现

使用 [原环境与 FlashQLA 补丁](../h100_fullcompile/README.md)；Python3.12、Torch2.8.0+cu128 等版本及49份依赖源码仍由原 guard 校验。额外的 FA3 必须是 [evaluation/fa3-dependency.json](evaluation/fa3-dependency.json) 锁定的六个文件。运行时会验证 interface、config、扩展二进制SHA256，缺失/不匹配立即失败，不回退到BF16。二进制和缓存没有提交到Git。

在当前服务器，已验证的 FA3 来源目录是：

```text
/data/q38/experiments/c4-lowprecision-20261031/fp8-attention/round2/src/flash-attention-4616f5116d33e3ba3cbb71d582caffcfcb49649c/hopper
```

将已验证产物复制到本版本的私有 `external/fa3/hopper`（只做一次），不会修改 site-packages：

```bash
PY=/data/q38/environments/q38-flashqla-torch280-cu128-py312/bin/python
RELEASE=/absolute/qwen35-moe-sparse/inference/h100_fused_fa3_fp8
"$PY" "$RELEASE/prepare_fa3.py" --from-hopper /absolute/qualified/fa3/hopper
CUDA_VISIBLE_DEVICES=1 Q38_PYTHON="$PY" bash "$RELEASE/run_baseline.sh" \
  /absolute/evaluation-a.jsonl /absolute/C4/hf /absolute/new-output-directory
```

固定入口明确设置 batch938、seq4096、trunk256、全部已验证融合开关和单卡，输出目录必须不存在。模型和数据路径必须为绝对路径。也可通过 `Q38_FA3_HOPPER_DIR` 指向同哈希的私有 FA3 目录。服务器 frozen release 自带该私有目录，直接可用。

底层两参数入口 `bash evaluation/eval.sh <data.jsonl> <C4-model>` 同样自动启用FA3；设 `EVAL_BATCH_SIZE=auto` 会重新搜索当前机器容量，而不是采用固定938。该实现显式检查 Q44/KV2/D64；其他档位尚未通过这套FA3精度的完整资格评测，C1不切换。

## 从固定源码构建

FA3 commit `4616f5116d33e3ba3cbb71d582caffcfcb49649c`，CUTLASS commit `7127592069c2fe01b041e174ba4345ef9b279671`。原始成功构建命令/环境在 [evidence/build/](evidence/build/)，包括私有 CUDA12.8.93、Python开发头和CUDA12依赖头；系统CUDA13不能替代。构建不绕过PyTorch的CUDA版本检查。

```bash
"$PY" "$RELEASE/build_fa3.py" --destination /absolute/new-fa3-build \
  --cuda-home /absolute/cuda-12.8 --python-include /absolute/python3.12/include \
  --extra-include /absolute/nvidia/cusparse/include
```

脚本下载并校验固定源码archive，在独立目录编译D64 SM90 BF16/E4M3及PackGQA；不会安装或升级共享包。需要已有兼容的 `setuptools`、`ninja`、C++编译器和相应CUDA12开发头。此可移植构建脚本由原成功命令整理，本次发布复用已验证二进制，未重新编译。不同机器重编译可能产生不同二进制SHA；需在独立副本完成数值/速度资格评测并更新依赖manifest后才能作为另一个版本使用，不能将旧实测分数直接归给新二进制。

## 版本与证据

[evaluation-source-sha256.json](evaluation-source-sha256.json) 固定发布源码；固定入口先检查该manifest。FA3的 `native`、`quant`、`attention` 三个函数AST与已完成完整A实验的版本一致。发布复测及声明见 [baseline.json](baseline.json) 和 [release-validation.json](release-validation.json)。共享的全部尝试记录仍以服务器 `/data/q38/optimization-log/LEDGER.md` 为准。
