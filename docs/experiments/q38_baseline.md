# 无 mHC、全注意力 dense C1

## 配方

入口为 [`configs/baseline.json`](../../configs/baseline.json)，生成器为 [`make_baseline_config.py`](../../scripts/make_baseline_config.py)。模型为 h512/L10/dense FFN768，8 层 Gated DeltaNet、2 层因果全注意力。删除全部 mHC 和 indexer，保留 N-gram、GQA、Q/K norm、attention 输出门及训练专用 MTP；不计算 indexer KL。

| 设置 | 数值 |
|---|---:|
| 目标训练 FLOPs | 3e17，预留 0.5% |
| 每 token 训练 FLOPs | 366,904,320 |
| 实际逻辑训练 FLOPs | 298,452,020,090,634,240 |
| 训练 token / 更新 | 813,432,832 / 6206 |
| 序列长度 / 全局每步 token | 4096 / 131,072 |
| 八卡每卡 micro-batch / 梯度累积 | 4 / 1 |
| 学习率 / warmup 更新 | 0.002038 / 62 |
| seed / data_seed | 0 / 0 |
| 优化器 / MTP 系数 | Muon + AdamW / 0.3 |
| 总参数 / 评测激活参数 $N_a$ | 105,763,460 / 33,776,160 |

FP32 主权重和优化器，BF16 混合精度；FFN 与投影均显式设置为 BF16，不使用 INT8/FP8。HF 导出为 BF16。训练采用无放回的 shuffled-block 采样；最终验证为 offset=2,002,536、16,001,875 targets 的纯 CE，包含 EOS，单位 nats/target。

训练池为 `/home/shared/llm_pt_ablation/cache/ling-no-mtp-c1-global-pruned10k-scale-isoflops-v1/{train.bin,valid.bin}`，训练文件包含 5,617,222,311 token。分词器为 `/home/shared/llm_pt/cache/tokenizers/ling-no-mtp-c1-top10k-v1`，实际词表 10112。训练数据与原 round2 消融池逐文件 SHA256 一致；与旧 17 点 dense IsoFLOPs 使用的 16B 池不同。

```bash
# 不读取训练数据的配置一致性检查
CUDA_VISIBLE_DEVICES='' "$PY" scripts/make_baseline_config.py --check

# 八卡完整 C1；输出目录必须是新的
PY="$PY" GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/train_dense.sh baseline --resume 0 --run_dir /path/to/new/c1
```

当前配置已固化 micro4、auto_micro_bs=0 和 BF16 设置，保留默认 `resume=1`；上面的从头训练命令显式覆盖为 0。原完整训练使用源码 `9f395f5377c5862c7b92d3eef9e2ba7a42df9343`，原 `configs/baseline.json` SHA256 为 `73604411e94ef5cc86229a0fcdc636e999bbdbf5595369868557d6cf1dd5b25d`，并在命令行显式覆盖 micro4、auto_micro_bs=0、resume=0 和 BF16 参数。当前配置不是该原始输入文件的逐字节副本；模型、数据、学习率及预算不变。

## 完整预算结果

八卡 A100 40GB 从头完成 6206 次更新，正式训练命令耗时 18 分 47 秒；最终验证纯 CE 为 **2.8175596495497355**。下表为同一物理 RTX 4090 48GB、完整 A 集、seq4096、batch16、每份固定权重三次评测的中位数。

| 模型 | BPB ↓ | 累计完整 forward 秒 ↓ | EvalTime ↓ |
|---|---:|---:|---:|
| round2 dense：mHC + learned QSA | 1.131918140 | 380.524227 | 86.159052 |
| 无 mHC + 因果全注意力 | 1.121291246 | 216.442143 | 53.337563 |

BPB 降低 0.010626894；forward 耗时下降 43.12%（1.758 倍加速），EvalTime 下降 38.09%。两者目标预算均为 3e17；原对照实际训练 FLOPs 为 298,480,036,430,741,504，整步取整后的预算差占目标约 0.00934%。

评测入口为非量化的 `q38.evaluate_single_gpu`。完整 A 计分文本 token 数为 120,142,976，原文 UTF-8 字节数为 445,910,984；插入的 EOS 和 padding 不计分。每种 batch 形状预热 5 次；CUDA 同步后累计完整 forward，包含全词表 logits，排除读数、packing、H2D、预热、CE 和日志。$\mathrm{BPB}=\mathrm{NLL}/(\ln 2\cdot\mathrm{bytes})$，$\mathrm{EvalTime}=T/[(N_a/10^9)(N_{\rm scored}/10^6)]$；$N_a$ 排除全部 Embedding、输出头和未调用的 MTP。

固定 batch16 用于与单项消融表比较，未搜索最大稳定 batch，也不是官方 H200 硬件成绩。只有一个训练种子，三次评测不是三次训练；两项结构同时改变，不能分离交互效应。评测先完成对照再完成组合，仍有时间顺序影响；GPU 遥测并未覆盖对照的全部三遍。未重新核算文档级训练/A 集交集。

## 产物与身份

共享根目录：`/shared/a100-209c/gzy/q38-no-mhc-full-attention-c1-20261030-1214/`。

- `train/hf/`：模型与分词器；权重 SHA256 为 `127e70821263996202b5631be5919388eca703230fe4055abe535751f834201d`。
- `train/run_config.json`、`train/final_eval.json`、`train/completion-validation.json`：实际配置、终点及 checkpoint/HF 逐张量核验。
- `control/formal-train.command`、`control/formal-train.log`：原八卡命令与训练日志。
- `evaluation_4090/evaluation/`：六份完整 A 原始结果与日志；`comparison.json` 保存三次全精度数值与对照权重身份。
- `evaluation_4090/PUBLICATION_VALIDATION.json`：归档内容和共享组读取权限校验。

`comparison.json` SHA256 为 `9f63189c24481a56f4aea24add9a5dcc9ea929e1d9ba0c11ca8680ec9d8dc5db`。大权重、训练日志和评测缓存保留在共享盘，不纳入 Git。
