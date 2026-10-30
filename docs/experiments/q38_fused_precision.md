# 无 mHC / 全注意力 C1：融合 BF16 与 FP8

目标是在完整前向口径下取得明显的 EvalTime 改善，并报告相对同次 BF16 对照的 BPB 代价。按用户最新要求，速度优先，0.001 不作为 BPB 增量的硬门槛；最终依据速度与质量的取舍选择主线。

## 实现

- `q38/fused_precision.py` 提供 inference-only 的 BF16 / FP8 FFN 整块后端，保留原始参数名称和 checkpoint；推理缓存不写入 state dict，禁止通过这些缓存反向传播。
- BF16 FFN 合并 gate/up GEMM，并用 Triton 融合 SiLU 和门控乘法。
- FP8 FFN 使用真实 E4M3 W8A8 GEMM、高精度累加、BF16 GEMM 输出。gate/up 共用输入；SiLU、门控乘法和 down 输入量化融合。
- 可选的融合 trunk 将 FP32 残差相加、RMSNorm、BF16 舍入与下一次输入写出合并。进入 FP8 FFN 时直接写 FP8，不再落地完整 BF16 归一化输出。保留 FP32 残差；GDN 核心、全注意力核心、embedding、词表头及 MTP 的计算精度保持原规则。
- FP8 scale 由独立训练流上的 absmax 校准得到，默认留 1.2 倍余量。推理不在 A 集计算、拟合或更新 scale；权重和布局在安装后缓存。
- `install_fp8_training` 使用真实 FP8 前向和 BF16 反向 GEMM，保留 FP32 主权重/优化器。反向为直通近似，不是对离散 FP8 cast 求精确导数。训练前向和部署前向共享固定 scale 与 E4M3 规则。
- 速度优先的扩大范围候选同时量化 GDN 的 qkv/z/out 与两层全注意力的 Q/K/V/O。GDN 的 a/b 控制投影、卷积、递推核心以及注意力 QK/softmax/PV 保留原精度。多个输入投影在部署时共享一次激活量化；正式 FP8 适应分支也训练这些投影。
- 训练使用原 trunk 和可微点运算；高性能推理缓存仅在加载 eval 模型后安装。FP8 训练导出目录保留 BF16 主权重及校准元数据，需要 `q38.evaluate_fused --backend fp8_fused` 启用部署路径；直接加载 HF 权重执行的是 BF16 参考路径。
- `--ffn_precision` 和校准文件 SHA256 纳入恢复检查。仅显式 `--precision_transition 1` 允许 BF16 checkpoint 转为 FP8，其他数据、预算和学习率恢复约束保持有效。

## 首轮速度筛选

来源：`runs/q38_baseline_qualification_20261030_v3/run/hf`，仅训练 3 个全局更新。它用于布局、执行路径和速度验证，不代表完整训练后的 BPB。相同 RTX 4090 UUID、真实 packed 首批数据、batch16、seq4096；预热 5 次、11 次 CUDA event 计时取中位数，包含完整词表 logits。

| 路径 | 完整前向中位数 ms |
|---|---:|
| 原始 baseline | 114.655 |
| 仅融合 BF16 FFN | 112.254 |
| 仅融合 FP8 FFN | 108.772 |
| 融合 trunk + BF16 FFN | 99.007 |
| 融合 trunk + FP8 FFN，归一化直接写 FP8 | 94.441 |

最后一项相对原始路径耗时降低约 17.6%，相对同样融合的 BF16 降低约 4.6%。这组数据是先后运行的短测，尚不能替代完整 A 集配对计时。

单独的 CUDA-event 归因测量显示，原始路径 GDN 约 50.0 ms、FFN 14.9 ms、全注意力 19.1 ms、主干归一化 5.7 ms、N-gram 3.0 ms、词表头 4.0 ms。GDN 内投影约 15.1 ms，属于 GDN 的嵌套耗时，不能重复相加；这些项也没有覆盖全部残差/布局/框架开销。

GDN 合并投影后，FP8 + 融合 trunk 为 97.637 ms；进一步改 chunk32 为 100.015 ms，均慢于保留原 GDN 的候选。这两项不进入正式实验默认路径。

## C1 对照协议

入口 `scripts/run_fused_c1.py` 在运行开始时冻结源码、配置和 SHA256。`status.json`、各阶段日志及进程信息独立落盘；控制器应以独立 session、空 stdin 和文件日志启动。

1. 新 baseline 按原 C1 配方训练到 step 5500，预算与学习率曲线始终按完整 6206 步计算。
2. 在该 checkpoint 上仅用训练数据校准，保存主权重、优化器和数据进度，分成 BF16 / FP8 两个分支。
3. 两个分支均续训至 6206 步。每个最终模型都是 813,432,832 token、逻辑训练 FLOPs 298,452,020,090,634,240；FP8 适应占最后 706 步。共享前缀只训练一次。精度转换和校准开销另于逻辑 GEMM FLOPs 口径，不宣称实际计算开销相同。
4. 前缀使用四卡，两个续训分支各两卡；全局 batch、样本顺序、学习率和剩余步数一致。GPU 数变化导致 per-rank RNG 不能原样重放，不能宣称与全程单卡 bitwise 相同。
5. 同一评测卡依次评估原 BF16、融合 BF16、仅 FFN 的 FP8 PTQ、扩大投影范围的 FP8 PTQ、扩大范围的 FP8 适应后部署，共享完整 A 集缓存、batch16 和 seq4096。FP8 PTQ 使用最终 BF16 checkpoint 的训练流校准；适应分支使用前缀校准，不重新拟合。

单次完整 A 集结果同时记录 BPB、完整前向累计秒和 EvalTime。短测另记录原始样本与中位数。正式流程最后增加三轮交错顺序短测，记录顺序/温度敏感性；这不等于三次完整 A 集评测。不能将单次不同 GPU 的耗时直接比较。`--qualification` 只执行 2 步共享训练和各 1 步续训，再跑五个后端短测，明确不生成完整预算 BPB。

## 使用

```bash
# 创建不可变源码快照，先跑完整训练/恢复/部署资格验证。
.venv_q35/bin/python scripts/run_fused_c1.py \
  --qualification --output runs/fused-qualification \
  --train-gpus GPU_UUID_1,GPU_UUID_2,GPU_UUID_3,GPU_UUID_4 \
  --eval-gpu GPU_UUID_0

# 正式 C1：去掉 --qualification，指定新的输出目录。
# 长任务使用 nohup/setsid，或由 subprocess.Popen(start_new_session=True) 启动。

# 完整 A 集评测；FP8 适应后的导出自带训练校准元数据。
GPUS=GPU_UUID_0 bash -c 'source scripts/env.sh; "$PY" -m q38.evaluate_fused \
  --model-path /path/to/fp8_adapted/hf --backend fp8_fused --projection-scope all \
  --dataset-path /home/shared/data/dolma3_mix_100M_a.jsonl \
  --packed-cache /home/shared/qwen35_moe_sparse/experiments/q38-4090-evaluation-20261028/packed \
  --output /new/evaluation.json'
```

CPU 回归覆盖原 baseline、预算、恢复默认值及现有量化入口。GPU 检查覆盖融合残差/归一化、SiLU 舍入、短尾、因果性、真实 FP8 训练前向与推理前向的一致性、有限梯度和优化器更新；完整形状与 DDP 恢复由资格验证流程检查。
