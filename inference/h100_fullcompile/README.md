# H100 full compile 推理快照：C1 / C4

2026-10-31 冻结的已验证版本。`evaluation/` 逐字保留当前 C1 baseline 的推理代码；本次 C4 使用相同 Python 源码和 tokenizer，通过各自 checkpoint 的配置选择结构。结构和权重未改变。此目录独立于仓库历史训练/评测入口。

| Baseline | Batch | BPB ↓ | 累计 forward 秒 ↓ | EvalTime ↓ |
| --- | ---: | ---: | ---: | ---: |
| C1 | 989 | 1.1234243392688898 | 43.03095544433592 | 10.604063761004959 |
| C4 | 768 | 0.9383547679682429 | 261.69925671386716 | 5.764464921254351 |

机器为 `zkrh-54`，单卡 NVIDIA H100 80GB HBM3；seq4096，完整 A 集 120,142,976 个计分 token、445,910,984 UTF-8 字节。编译及每种 shape 的 5 次预热不计入 forward；cross entropy 也不计入 forward。CUDA events 计时，物化全部 token/vocabulary 的 BF16 logits。

机器可读的身份、公式、配置、权重 SHA256、结果和完整运行 metadata 见 [baselines/2026-10-31/](baselines/2026-10-31/)；入口索引为 [baselines.json](baselines/2026-10-31/baselines.json)。代码文件校验见 [evaluation-source-sha256.json](evaluation-source-sha256.json)。权重、数据和编译缓存保留在服务器，不存入 Git。

## 实现与环境

- `torch.compile(fullgraph=True, dynamic=False)`，禁止静默 eager fallback，禁用 CUDA Graphs。
- 原 native CUTLASS FP8 gate/up GEMM + 单独 SwiGLU/FP8 quantization。已停止的 CUTLASS SwiGLU epilogue 实验未采用。
- 动态逐行 E4M3 W8A8；BF16 residual/logits，FP32 norm/state。
- Q/K norm + RoPE + gate packing 融合；cuDNN causal GQA；FlashQLA SM90 D64；GDN 内部分块上限 256。
- FLA convolution、FlashQLA 和 cuDNN 是图中的显式 library custom ops，其余主干由 Inductor 编译。

精确版本与 49 个依赖源码校验见 [dependency-manifest.json](evaluation/dependency-manifest.json)。已验证运行时为 CPython 3.12、Torch 2.8.0+cu128、Triton 3.4.0、FLA 0.5.1、FlashQLA 0.1.3、Tilelang 0.1.12、Transformers 5.19.0。服务器现有解释器：

```text
/data/q38/environments/q38-flashqla-torch280-cu128-py312/bin/python
```

FlashQLA 使用两个锁定 SHA 的 SM90 D64 补丁。[patches/flashqla/](patches/flashqla/) 保留补丁、应用脚本和安装核验说明。不能用未修改的同版本 wheel 代替已验证源码；运行入口会核验版本与源码。新机器还需要开发版 CUDA Toolkit、nvcc 和 C++ 编译器；本次机器使用 CUDA Toolkit 13.0。安装/补丁应在专用环境或独立 target 目录进行。

## 复现当前 baseline

以下命令使用 checkout 中的代码，不修改现有 release。`REPO`、输出目录和 checkpoint 路径需按机器实际位置设置；输出目录应是新目录。checkpoint 是完整 HF 导出；共享模型源码/tokenizer 的快照在 [model/](model/)，各档配置在对应 baseline 子目录。模型权重身份以 `baselines.json` 中的 SHA256 为准。

```bash
REPO=/path/to/qwen35-moe-sparse
PY=/data/q38/environments/q38-flashqla-torch280-cu128-py312/bin/python
DATA=/data/q38/datasets/evaluation-a-v1/dolma3_mix_100M_a.jsonl
C1=/data/q38/experiments/dense-c1-inference-h100-20261030/c1-infra-fullcompile-20261031/release/submission/models/FLOPs_3e17
C4=/data/q38/experiments/dense-c1-c4-20261030/c4/train/hf

# C1：新建输出目录后进入该目录
mkdir /path/to/new-c1-eval
cd /path/to/new-c1-eval
CUDA_VISIBLE_DEVICES=0 Q38_PYTHON="$PY" EVAL_BATCH_SIZE=989 \
  bash "$REPO/inference/h100_fullcompile/evaluation/eval.sh" "$DATA" "$C1"

# C4：同样使用新的输出目录，显式覆盖 C1 的默认 batch
mkdir /path/to/new-c4-eval
cd /path/to/new-c4-eval
CUDA_VISIBLE_DEVICES=0 Q38_PYTHON="$PY" EVAL_BATCH_SIZE=768 \
  bash "$REPO/inference/h100_fullcompile/evaluation/eval.sh" "$DATA" "$C4"
```

每次输出 `result.json` 与 `run-metadata.json`。评测机器需空闲，GPU 型号、依赖、精度、数据与 batch 保持一致。显式设置 `EVAL_BATCH_SIZE=auto` 可在新环境中重新搜索容量并跑完整评测；它不改变本目录冻结的历史记录。

## 容量边界与 C4 已知问题

C1 的 989、C4 的 768 均完成全量评测；相邻 990、769 分别 OOM。这是当前实现与已验证环境的实测边界，不能解释为模型架构的显存极限。

C4 的 44 个 query heads 在当前 packing 路径中，同时存活原始 Q+gate、处理后的 Q、gate 副本。batch768 三块分别为 33、16.5、16.5 GiB，合计 66 GiB；加上 residual、FP8 input、权重等，实测峰值 allocated 为 78.500668 GiB。batch900 时同三块合计 77.34375 GiB，尚未包含其他开销。897 的容量探测已在该处 OOM。

因此，约 900 的目标需要先减少 Q/gate 重复物化或分块处理 attention，再验证容量和速度。该优化尚未实施在此快照中，不能将 batch900 写为已通过。

旧 C4 batch16 的 BPB / forward / EvalTime 为 0.9369962943702319 / 442.09027031157166 / 9.73794838869281。本次前向提速 1.6893×，但 BPB 上升 0.001358473598011。两次 batch 和推理路径不同，精度差异尚未单独归因；这组历史对比不证明纯 compile 的独立收益。

原始已验证服务器目录：

```text
C1: /data/q38/experiments/dense-c1-inference-h100-20261030/c1-infra-fullcompile-20261031/release
C4: /data/q38/experiments/c4-infra-fullcompile-20261031
```
