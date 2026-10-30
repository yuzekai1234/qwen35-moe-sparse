# 当前 dense IsoFLOPs 方法

## 问题与模型家族

已验证 C1 使用 dense SwiGLU、无 mHC、因果全注意力、FP32 主权重/优化器及 BF16 混合精度。删除 mHC 和 indexer 改变了参数量与计算开销，旧 mHC/QSA 家族的 N/D 幂律不能直接当作该模型的最优尺度。目标是在新结构上重新估计每 token 激活参数量 $N_a$ 与训练 token 数 $D$ 随训练 FLOPs $C$ 的关系，不直接预测 CE 或 BPB。

模型与配方由 [`make_baseline_config.py:make_training_config`](../../scripts/make_baseline_config.py) 统一构造：原 `shape_for` 先解析 MoE 的 N-gram 行数，再转 dense、设置 `mhc_enabled=0` 和 `qsa_mode="full"`。保留 GDN、N-gram、GQA、Q/K norm、输出门和训练专用 MTP；MTP 系数 0.3，KL 为 0，无 dense warmup，不使用 FP8/INT8。C1=h512/L10/F768、61,440 行 N-gram 的模型、预算和科学训练设置与已完成 C1 一致。

## 固定实验设计

| 目标 FLOPs | 隐藏宽度 | 点数 |
|---:|---|---:|
| 3e15 | 64、128、192、256 | 4 |
| 1e16 | 128、192、256、320 | 4 |
| 3e16 | 192、256、320、384 | 4 |
| 1e17 | 256、320、384、448、512 | 5 |

四档网格在训练前固定，不因新计算图重新过滤。所有点使用相同 5,617,222,311-token 训练池、10112-token 分词器、seq4096、seed/data_seed0 和无放回 block 采样；全局 batch 与学习率采用原预算缩放规则。保留 0.5% FLOPs 预留，训练 token 向下取整到完整全局 batch；每点独立完成完整预算与学习率调度。

纯 CE 验证分区为 offset=2,002,536、16,001,875 targets，包含 EOS，单位 nats/target。$N_a$ 排除全部 Embedding、输出头及评测不调用的 MTP；训练 FLOPs 包括 MTP。各预算对 CE 关于 $\log_{10}N_a$ 拟合凸二次曲线，在顶点插值 $D$，再以实际 $FD$ 的预算内均值拟合 N/D 幂律。任何缺点、非有限值、实测最低点在边界、非凸或顶点越界都拒绝输出，不自动补点或引入 C1。

C1 固定为已验证 h512，不重复训练、不参与拟合，也不称为新拟合前冻结的独立验证。另报告新幂律自由选择的 C1 形状是否仍为 h512。C2–C4 按原联合对数相对误差选择宽度，但只允许 $D$ 不超过数据池可用的唯一 4096-token block 容量；同时保留不加该约束的建议。单点 seed0 训练及留一预算敏感性不构成置信区间或最优性证明。

## 数据与输出边界

迁移后的数据目录必须显式指定，不依赖 `/home/shared` 别名。启动前对 `train.bin`、`valid.bin` 和 `tokenizer/tokenizer.json` 完成独立全文 SHA256 核验。身份快照的小 JSON 格式为：

```text
train:      {path: 绝对规范路径, bytes: 文件字节数, mtime_ns: 纳秒修改时间, sha256: 全文哈希}
validation: {path: 绝对规范路径, bytes: 文件字节数, mtime_ns: 纳秒修改时间, sha256: 全文哈希}
tokenizer_sha256: tokenizer.json 的全文哈希
```

期望的内容身份、字节数、网格与家族由纯标准库模块 [`q38_isoflops_contract.py`](../../scripts/q38_isoflops_contract.py) 固定。生成器只读取该小快照，不读取训练文件；控制器启动执行及每个 worker 启动前再次核对已鉴定文件的大小和修改时间，拟合器核对配置、源代码、终点及运行时身份。该 stat 检查不能替代独立全文哈希，也不能识别刻意保持大小和修改时间的内容替换。运行时缺失字段不得从声明中补默认值。配置、日志、checkpoint 与结果写入独立实验目录；权重、语料、缓存和临时检查输出不纳入 Git。

## 接口

`PY` 指向已经完成 GPU 资格检查的 Python；`RUN` 必须是新的实验目录。数据先完成内容核验，才可生成正式训练计划：

```bash
DATA=/data/q38/datasets/c1-pruned10k-v1
RUN=/data/q38/experiments/new-combined-isoflops

CUDA_VISIBLE_DEVICES='' "$PY" scripts/make_q38_isoflops_configs.py \
  --data-identity "$RUN/data-identity.json" \
  --data-root "$DATA" --tokenizer "$DATA/tokenizer" \
  --output "$RUN/configs" --run-root "$RUN/runs"

# 默认仅显示计划，不启动训练；0–3 为总共四个单卡 worker
"$PY" scripts/run_q38_isoflops.py --manifest "$RUN/configs/manifest.json" \
  --gpus 0,1,2,3 --python "$PY" --output "$RUN/control"
```

数据、资源和源码身份全部核验后，显式增加 `--execute` 才启动队列。每任务上限 4 小时、队列上限 6 小时；只在指定 GPU 上运行，保留控制器日志。全部 17 点成功结束后：

```bash
"$PY" scripts/fit_q38_isoflops.py --manifest "$RUN/configs/manifest.json" \
  --output "$RUN/prediction" --c1-run-dir "$RUN/c1-reference"

CUDA_VISIBLE_DEVICES='' "$PY" scripts/make_dense_scaling_configs.py \
  --fit-source "$RUN/prediction/fit.json" \
  --data-root "$DATA" --tokenizer "$DATA/tokenizer" \
  --output "$RUN/presets" --run-root "$RUN/extrapolated"
```

拟合器写出顶层 raw `fit.json` 和 C1 配方，不创建、恢复或重训 C1 模型。`--fit-source` 必须显式指定当前家族的完整结果，不自动退回旧拟合；同时复核原始点、曲线接受门及记录系数是否一致。拟合配方默认使用 raw 结果已鉴定的数据位置，显式提供的路径也必须匹配该身份；不同迁移位置须先重新鉴定，不能借同名目录替换语料。新证据审核通过后才固化结果来源并更新默认配置；在此之前，已发布 C2–C4 仍是明确标注的暂用规模。
