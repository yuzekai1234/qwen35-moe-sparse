# Dense C1：Hopper 单卡推理与完整 A 榜复现

此目录是独立、可选的评测入口，来自通过两张 H100 完整 A 榜验证的源码快照。
不改变 `q38/train.py`、模型定义、原来的 `q38.evaluate` 或 `scripts/eval.sh`。
基于 PR #2 的 `feat/inference-low-precision-bpb`，模型源码 SHA256 为
`3f59c7322a50d28b9474426c166f70869e2d3254b6dbd4be2a02aa1bc8a301f5`。

## 已验证范围

单 H100 80GB HBM3，Dense C1 h512/L10/FFN768，无 mHC；8 层 GDN、2 层 causal full attention。
同一 checkpoint、seq4096、完整 A 榜；各方案使用独立搜索的最大稳定 batch。

| 方案 | 最大 batch | EvalTime | bpb |
|---|---:|---:|---:|
| 原 BF16 | 926 | 24.72494 | 1.12129183 |
| 原通用 FP8 | 942 | 21.49954 | 1.12341596 |
| 当前组合，GPU6 | 942 | 11.67973 | 1.12342475107462 |
| 最终入口独立复核，GPU7 | 942 | 11.63639 | 1.123424751074623 |

batch943 在相同最终配置下 OOM。相对 BF16，bpb 增加约 0.002133，低于 0.03 门限；
EvalTime 相对原通用 FP8 降低约46%。11.64 是归一化指标，不是整个作业墙钟秒数；
独立复核的纯 forward 为47.22010秒。**没有 H200 或 Dense C4 实测结果。**

默认采用原生 E4M3 Linear/SwiGLU、cuDNN BF16 causal GQA、输出/预量化融合、
BF16 残差存储、FlashQLA D64 和输入生命周期修复。完整 logits 保留 BF16；
FP8 logits、FP8 残差、CUDA Graph 均未启用。不在推理时删除 mHC 或改模型架构。

## 环境安装

使用隔离的 **CPython3.12 / Linux x86_64** 环境，避免覆盖训练环境。
已验证 Torch2.8.0+cu128、Triton3.4.0、FLA0.5.1、cuDNN9.10.2.21，精确版本和源码
由 `evaluation/dependency-manifest.json` 与启动时 guard 检查；不会静默 fallback。

```bash
cd inference/hopper
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-deps --target ./q38_backend_deps \
  -r backend-installation/requirements-flashqla-backend.txt
export PYTHONPATH="$PWD/q38_backend_deps${PYTHONPATH:+:$PYTHONPATH}"
python backend-installation/apply_flashqla_d64.py \
  --package ./q38_backend_deps/flash_qla --record ./flashqla-patch.json
python backend-installation/verify_flashqla_dependencies.py \
  --manifest backend-installation/flashqla-source-manifest.json \
  --verify-records --output backend-integrity.json
python -c 'import sys; sys.path.insert(0,"evaluation"); from dependency_guard import verify; print(verify())'
```

JIT 还需要开发版 CUDA Toolkit、C++20 编译器和可写 cache。实测 JIT 工具是
**NVCC13.0.88 / g++13**，Torch CUDA runtime 为12.8；不是“已验证纯 CUDA12.8 toolkit”。
FlashQLA 上游最低要求不等于本项目已验证工具链。新机器/工具链需重跑数值、memcheck 和完整模型门限。
不要覆盖 cuDNN 的动态库搜索路径。完整的两处 D64 补丁和安装审计见 `backend-installation/`。

## 评测

使用可信、离线 HF 导出的 checkpoint（含 `auto_map`、对应模型 Python 文件、tokenizer）。
入口通过 `trust_remote_code=True` 读取 checkpoint 自带定义，**不会自动换成当前仓库模型**。
已验证权重 SHA：`127e70821263996202b5631be5919388eca703230fe4055abe535751f834201d`。
数据 SHA：`4689036c6d9b413b03b68f0042509e0f3ea7b4d98cb210da4ef5b157a18c9f5c`。
权重、tokenizer 与 A 数据不随本 PR 上传。

```bash
export Q38_PYTHON=/path/to/qualified/venv/bin/python
export CUDA_HOME=/path/to/qualified/cuda-toolkit
export PATH="$CUDA_HOME/bin:$PATH"
# 同时保留上面指向 q38_backend_deps 的 PYTHONPATH。
CUDA_VISIBLE_DEVICES=0 "$Q38_PYTHON" evaluation/eval.py \
  --dataset-path /path/A.jsonl --model-path /path/checkpoint \
  --output /path/results/result.json
```

省略 batch 时，CPU 外层按实际显存建立上界，fresh 子进程探测容量，再用原始完整数据验证；
真实 OOM 才回退，其他错误直接失败。不保留10%空闲，不固定942，不用质量 pilot 的序列数限制容量。
容量探测重复真实窗口只用于测内存，不用于计分、校准或更改权重。
自动最大 batch 仅支持单 GPU。请独占所选 GPU，避免其他作业干扰容量结果。

显式 batch 优先级为 **CLI `--batch-size` > `EVAL_BATCH_SIZE` > 自动**：

```bash
CUDA_VISIBLE_DEVICES=0 "$Q38_PYTHON" evaluation/eval.py \
  --dataset-path /path/A.jsonl --model-path /path/checkpoint \
  --batch-size 942 --output /path/results/result.json
# 比赛包装入口参数顺序是 dataset、model：
CUDA_VISIBLE_DEVICES=0 EVAL_BATCH_SIZE=942 bash evaluation/eval.sh /path/A.jsonl /path/checkpoint
```

显式 batch 不会被偷偷缩小；其他显存容量/模型要重新测，不应照抄942。
正式输出 `result.json` 仍只有 model_path/bpb/evaltime；旁边写 `run-metadata.json`，
自动模式另写 `auto-batch-metadata.json`。`release-defaults.json` 尊重已有显式环境变量；
更改默认配置即不再是表中已验证组合。BF16 残差要求 prequant NORM/INPUT/FFN 全开。

## 指标与证据

完整计分120142976 tokens、445910984原文 UTF-8 bytes，Na33776160，排除 Embedding/词表头/未调用 MTP。
`EvalTime = forward_seconds / ((Na/1e9)*(scored_tokens/1e6))`。
完整 token/词表 logits 计算在计时内；packing、H2D、预热与 CE 按原协议不计时。
EOS/padding mask、右移位置、计分窗口不变。CE FP32、NLL FP64；残差加法/Norm归约 FP32，
残差持久存储 BF16。FP8 GEMM 使用 fast accumulation，不能宣称等同严格 IEEE FP32 归约。

- `evidence/`：两张卡完整 A 输出和容量/profile摘要；仅替换机器绝对路径占位符，数值未改。
- `validated-source-sha256.json`：`evaluation/` 与最终 GPU7 复核源码逐文件对应；不包含新模型成绩。
- `OPTIMIZATION_RECORD.md`：采用/未采用优化及限制。CUDA Graph/compile、DeepGEMM、FP8 attention
  尚无合格的整模型提速证据；FP8 logits 大 batch 反而变慢，按要求停用。
- `backend-installation/flashqla_*unit.py`：固定维度及非零初态/尾批数值检查；memcheck摘要在 evidence。

`evaluation/` 有旧实验分支模块，原样保留以维持已验证快照；它们不是默认选项，也不代表本 PR
新增 MoE 专家加速。legacy `residual_norm.original_FP32` 是早期安装 metadata，最终
`prequant_norm.residual_storage=bf16` 覆盖；mHC模块数为0，安装层数含未调用 MTP。

本地无需 GPU 的包装/计分契约检查：

```bash
python tests/test_contract.py
```

kernel 性能/数值结论来自附带 H100 完整评测和 sanitizer 记录，CPU 检查不能替代实机资格。
