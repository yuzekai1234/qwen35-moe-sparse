# Q38 dense

这是技术说明。dense 基于冻结 Q38 提交 [`6daa43d6`](https://github.com/yuzekai1234/qwen35-moe-sparse/commit/6daa43d6fdb5cb929036028fd7b03df488df1cc3)，只替换 MoE 前馈结构；每 token 激活参数近似不变，总参数减少。GDN、QSA、mHC、N-gram、层数、隐藏宽度及 MTP 保留，MTP 的前馈层同样替换。

## 结构与预算

dense 使用 SwiGLU，中间宽度为 `topk * moe_ffn + shared_ffn`。删除专家路由器和共享专家门，保留 SwiGLU 的 `gate_proj`。配置通过 `num_experts=0` 选择 dense，`dense_ffn` 对应 HF 的 `intermediate_size`。N-gram 行数先按原 MoE 解析，再显式固定，避免自动缩表；这不是原 MoE 权重的无损转换。

下表为原 C1–C4 预设的对应配置，不是低预算拟合网格。\(N_a\) 排除全部 Embedding、输出头和训练专用 MTP；完整参数量与训练 FLOPs 包含 MTP。

| 配置 | 层数 | 隐藏宽度 | dense 中间宽度 | MoE \(N_a\)，M | dense \(N_a\)，M |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 9 | 448 | 672 | 27.12 | 26.86 |
| C2 | 11 | 576 | 864 | 52.25 | 51.84 |
| C3 | 15 | 768 | 1152 | 129.4 | 128.7 |
| C4 | 20 | 1024 | 1536 | 300.2 | 298.9 |

计数来源为 [`configs/dense/`](../../configs/dense/)、[`presets.py`](../../q38/presets.py) 和 [`flops.py`](../../q38/flops.py)。C1 删除的主路径门控参数为 262,080 个，不补偿；全部权重为 78,649,350 个参数，不等于每 token 激活量。历史 `active_nonvocab_params`、`total_nonvocab_params` 仅排词表，保留原口径；表中 \(N_a\) 使用 `active_params_excluding_embeddings`。

同目标训练 FLOPs \(C\)，分别按每 token FLOPs \(F\) 和全局每步 token 数 \(B\) 重算训练 token 数 \(D\)，保留原 0.5% 预留：

\[
D=B\left\lfloor\frac{0.995C}{FB}\right\rfloor.
\]

C1 dense 的 \(F=289{,}974{,}016\)、\(B=131{,}072\)、\(D=1{,}029{,}308{,}416\)，均见对应配置。核算包括训练 MTP 和 indexer KL 的矩阵计算，不包括优化器、验证、编译及 QSA dense warmup 额外开销；不代表实测硬件操作计数。MoE/dense 不固定相同 token 数或最终 loss。

另有两种结构共用的 QSA 兼容修复：Torch 2.7 对少于 128-token 的尾窗口显式选择 prefill 内核，保留同一稀疏选块、因果掩码和尾目标，避免自动 decoding 的 GQA 块大小断言。4096-token 分支不变，不回退到全注意力。

## 使用

`PY` 指向已有 Torch 2.7.1、Transformers 5.19.0 环境。默认输出位于 `runs/q38_dense/`；可通过 `--run_dir` 指定新的目录。

```bash
CUDA_VISIBLE_DEVICES='' "$PY" scripts/make_dense_configs.py --check
PY="$PY" GPUS=0 bash scripts/train_dense.sh c1 --dry-run
PY="$PY" GPUS=0 bash scripts/train_dense.sh c1 --run_dir /path/to/new/output
```

入口也接受 `c2`、`c3`、`c4` 或配置 JSON。改变模型或预算后应重新生成配置；显式 `tokens` 超过 `compute` 时在读数据前拒绝。`--tokens 0 --warmup 0` 按完整预算重算，不保留上述预留。禁止跨配置恢复 checkpoint。

## 验证

以下 CPU 测试共 21 项通过，覆盖结构、参数与 FLOPs、N-gram 固定、MTP 前反向、优化器分组、HF 保存加载及预算保护：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "$PY" -m unittest \
  tests.test_dense tests.test_model tests.test_presets \
  tests.test_flops_audit tests.test_isoflops -v
```

GPU 短测入口为 [`smoke_dense.sh`](../../scripts/smoke_dense.sh)，必须指定新的输出目录：

```bash
PY="$PY" GPUS=0 timeout 900s bash scripts/smoke_dense.sh /path/to/new/smoke-output
```

A100 40GB、Torch 2.7.1+cu126 实测通过：C1 完整形状、seq4096、全局每步 131072 token，先两次更新保存，再独立进程恢复至三次，共 393216 token；其中两次为非 warmup 稀疏训练。检查覆盖优化器步数、有限损失和梯度、全部 259 个去重导出参数（含 18 个 MTP 参数）、本地模型与 `AutoModelForCausalLM(..., trust_remote_code=True)` 加载模型的同输入 GPU 稀疏 logits 逐位一致。64-token GQA 与 1024-token 非全选块注意力回归也通过。

短测使用 warmup1，不是完整预算或质量评测；正式实验不恢复该 checkpoint。独立结果复核退出码为 0，原件位于 `/shared/a100-209c/gzy/q38-dense-isoflops-20261027/qualification/c1-qsa-fixed/`；运行模型 SHA256 为 `2b89565fa1c966a7725ab825e18aa1150c460d8520f400087f8efc091012313a`。默认配置与短测配置的输出目录不同，不影响模型和预算。

## 未验证与待确认

- dense 四档 IsoFLOPs（3e15、1e16、3e16、1e17）的完整训练、拟合及 C1=3e17 独立验证尚未完成；C1 不参与拟合。
- 完整预算下的 CE、BPB 和同口径速度尚无本轮结果。短测和 A100 本地运行不代表 H200 官方成绩。
