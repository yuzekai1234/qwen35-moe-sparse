# 当前 dense IsoFLOPs 方法与结果

## 问题与模型家族

已验证 C1 使用 dense SwiGLU、无 mHC、因果全注意力、FP32 主权重/优化器及 BF16 混合精度。删除 mHC 和 indexer 改变了参数量与计算开销，旧 mHC/QSA 家族的 N/D 幂律不能直接当作该模型的最优尺度。目标是在新结构上重新估计每 token 激活参数量 $N_a$ 与训练 token 数 $D$ 随训练 FLOPs $C$ 的关系，不直接预测 CE 或 BPB。

模型与配方由 [`make_baseline_config.py:make_training_config`](../../scripts/make_baseline_config.py) 统一构造：原 `shape_for` 先解析 MoE 的 N-gram 行数，再转 dense、设置 `mhc_enabled=0` 和 `qsa_mode="full"`。仅将原 QSA 层改为因果全注意力，GDN 层保留；同时保留 N-gram、GQA、Q/K norm、输出门和训练专用 MTP。MTP 系数 0.3，KL 为 0，无 dense warmup，不使用 FP8/INT8。C1=h512/L10/F768、61,440 行 N-gram 的模型、预算和科学训练设置与已完成 C1 一致。

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

## 2026-10-30 实测结果

四个独立单卡任务在 H100 的 GPU 0–3 上完成全部 17 点，训练、拟合和配置导出的退出码均为 0；整条任务耗时 65 分 4 秒。训练源码固定为 `5ee96657df7d923c63b6dcddb38a53dd99a69a17`。数据位于 `/data/q38/datasets/c1-pruned10k-v1/`：从下载归档还原的 5,617,222,311-token 训练前缀、验证集和分词器的独立全文 SHA256 均与原 C1 一致，本轮没有改用 16B 训练池。

[原始拟合 JSON](q38_combined_isoflops_results.json) 按原字节保存，SHA256 为 `238e4669fbf4d7b9d7b8e4929af3fa9b67d051f867b400b960f31da1f5a05621`；其中记录全部 17 点、数据身份、训练源提交、manifest、拟合代码和终点文件哈希。原始配置、日志、checkpoint、HF 权重及预测位于 `/data/q38/experiments/no-mhc-full-attention-isoflops-20261030/`；目录可遍历、文件共享可读，17 个权重和关键终点文件均通过非所有者实际读头检查。独立复核逐条核对原始终点，并用独立实现重算曲线、幂律、留一结果和离散选型，均一致。

| 目标 FLOPs | 二次曲线顶点 $N_a$，M | 插值 $D$，M token | CE 拟合 RMSE | 实测最低 CE | 实测对应 h / L |
|---:|---:|---:|---:|---:|---:|
| 3e15 | 1.831 | 59.636 | 0.011573 | 3.728040 | 192 / 4 |
| 1e16 | 2.661 | 155.490 | 0.029776 | 3.460762 | 256 / 5 |
| 3e16 | 8.350 | 235.607 | 0.013263 | 3.250696 | 256 / 5 |
| 1e17 | 20.989 | 393.511 | 0.001550 | 3.026588 | 448 / 9 |

顶点及插值 $D$ 不是实训模型的测量；最低 CE 列才来自实际训练点。四条曲线的实测最低点、凸性和顶点位置均通过预定接受门。以预算内实际训练 FLOPs 均值为 $C$，得到：

$$
\log_{10}N_a=0.726483\log_{10}C-5.062557,\qquad
\log_{10}D=0.523786\log_{10}C-0.270745.
$$

$C$ 的单位为 FLOPs，$D$ 为 token；展示系数经过取整，配置使用原始 JSON 的完整精度。$N_a$ 排除的 Embedding、输出头和 MTP 仍可能贡献训练计算，故不以 $C=6N_aD$ 约束两指数之和。逐预算留一时，C1 连续预测为 $N_a=32.596$–$59.266$M、$D=609.040$–$1022.426$M；这只是敏感性范围，不是置信区间。

### 当前配方与结论边界

默认 [`configs/dense_isoflops/`](../../configs/dense_isoflops/) 已绑定当前原始拟合，按实际计算图重新核算预算：

| 配置 | 目标 FLOPs | h / L / FFN | $N_a$ | 训练 token $D$ | 状态 |
|---|---:|---:|---:|---:|---|
| C1 | 3e17 | 512 / 10 / 768 | 33,776,160 | 813,432,832 | 保留既有已验证模型 |
| C2 | 1e18 | 704 / 14 / 1056 | 92,437,586 | 1,183,318,016 | 未训练外推 |
| C3 | 3e18 | 896 / 18 / 1344 | 188,347,804 | 1,895,563,264 | 未训练外推 |
| C4 | 1e19 | 1152 / 22 / 1728 | 377,872,340 | 3,375,366,144 | 未训练外推 |

C1 连续预测为 $N_a=43.080$M、$D=764.752$M；自由网格会选 h576/L11，而不是已验证 h512。按预定规则保留 h512，不重复训练，也不将既有 C1 重新包装成事前独立验证。该 C1 的模型、预算和非路径训练字段与 `configs/baseline.json` 相同；数据、验证和分词器路径指向内容相同的已鉴定 H100 副本，输出目录独立。

C2–C4 在 h64:64:1600 上最小化联合对数相对误差；本轮三个无约束最优解已全部满足唯一数据容量 5,617,221,632 token，因此容量限制未改变选型。连续 N/D 预测不等于同 FLOPs 下的离散配方：例如 C4 连续预测约 550.325M / 4,799.355M token，实际配方仅为其 68.66% / 70.33%，不能混报。

这些结果支持固定配方下的有限预算测量与可复现外推，不证明规模最优或训练已充分收敛。各点仅 seed0，终点虽完成 cosine 调度且无非有限值，最后一次开发集 CE 仍低于上次；尤其 3e16 的 RMSE=0.013263，大于 h256 与 h320 的 CE 差 0.005751，顶点位置不宜过度解读。C2–C4 分别外推到最高实测预算的 10/30/100 倍，尚未做完整训练、目标硬件显存或质量验证；N/D 不预测 CE、BPB 或 EvalTime。历史 mHC/QSA 拟合还使用另一份 16B 训练池，新旧指数变化不能单独归因于结构变化。

## 接口

`PY` 指向已经完成 GPU 资格检查的 Python；`RUN` 必须是新的实验目录。数据先完成内容核验，才可生成正式训练计划：

```bash
DATA=/data/q38/datasets/c1-pruned10k-v1
RUN=/data/q38/experiments/new-combined-isoflops

CUDA_VISIBLE_DEVICES='' "$PY" scripts/make_q38_isoflops_configs.py \
  --data-identity "$DATA/data-identity.json" \
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

拟合器写出顶层 raw `fit.json` 和 C1 配方，不创建、恢复或重训 C1 模型。配置生成器默认使用已审核并保存在仓库内的原始结果，也可用 `--fit-source` 显式选择另一份当前家族的完整结果；None、缺失、损坏或旧家族均拒绝，不自动退回历史尺度。生成时复核原始点、曲线接受门及记录系数是否一致，但不改写原始拟合。配方使用 raw 结果已鉴定的数据位置，显式提供的路径也必须匹配该身份；不同迁移位置须先重新鉴定，不能借同名目录替换语料。

```bash
# 读取小结果并复核五份配置，不读取大语料、不启动训练、不写文件
CUDA_VISIBLE_DEVICES='' "$PY" scripts/make_dense_scaling_configs.py --check
```
