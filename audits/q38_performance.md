# Q38 训练与推理性能记录

独占 NVIDIA A100-SXM4-40GB（GPU 0），Torch 2.7.1+cu126，BF16 autocast、词表 10112、seq=4096。预热两次，三次测量取中位数；profile 时间不并入吞吐。没有完整预训练或 C2–C4 GPU 长跑。

| 模型/路径 | 训练步 ms | 训练 tok/s | 推理 ms | 推理 tok/s | 训练峰值 allocated |
|---|---:|---:|---:|---:|---:|
| h128/L4 baseline | 220.53 | 18.57K | 20.61 | 198.69K | 1.04 GiB |
| h128/L4 optimized | 158.56 | 25.83K | 19.19 | 213.46K | 1.12 GiB |
| C1 h448/L9 micro1 | 232.32 | 17.63K | 42.02 | 97.48K | 6.88 GiB |
| C1 h448/L9 micro4 | 439.05 | 37.32K | 49.60 | 330.32K | 21.38 GiB |

基准每个 micro batch 做一次 optimizer step，包含主 CE、router aux、MTP、indexer KL、前反向、Muon+AdamW；不含数据加载与 grad clipping。正式 C1 global batch=131072 token，micro4 累积 8 次才更新，optimizer 开销进一步摊薄。推理为完整主头 logits 的 packed prefill，使用与比赛 eval 相同的 Triton experts、FLA、compiled norms；不包含自回归 decode。

C1 micro4 的训练吞吐是 micro1 的约 2.12 倍，推理约 3.39 倍。micro1 有明显小 batch/launch 开销，应优先采用已验证可运行的 micro4。micro4 forward/backward/optimizer 中位数为 206.92/205.38/26.80 ms，optimizer 仅占约 6%。最后两次推理均约 49.60 ms，第一测量55.12 ms，存在短测调频/缓存噪声。

## 修复与证据

- 原始 GPU 训练在 RoPE 后 q/k=FP32、v=BF16，flex attention 直接报错。现按 q dtype 转换 cos/sin。未修复前无有效 GPU 速度。
- Torch 2.7 在读取 kernel_options 前，以默认 128 tile 验证并拒绝64-token blocks。兼容调整仅对2.7.x设 forward=(64,64,4,3)、backward=(32,64,4,3)，先初始化 Inductor lowering，避免 CPU 导入循环。更高版本不套用私有补丁。
- A100+Torch2.7 grouped_mm退回逐专家 GEMM。训练用无丢token的精确容量 padded batched GEMM；每层读取一次最大专家负载。原生 grouped_mm 可用的 Hopper/新Torch仍走原生实现。H100未实测。推理保持 forward-only Triton experts。
- Muon 按矩阵尺寸合批，以 foreach 更新，AdamW使用fused实现。NS仍逐专家独立。64Mi元素分块限制cat临时空间，支持专家张量内分片。
- C1 全编译NS的大归约产生寄存器spill，profile四次归约累计约41ms。保留原生CUDA norm，只编译多项式迭代；C1 micro1 optimizer由71.01降至39.83ms，整步262.82降至232.32ms。旧证据见 q38_c1_norm_spill_before.*。
- mHC归一化/系数/聚合合成单compiled图。推理indexer仍对所有token投影及RoPE，仅对每块首query打分，将评分query维缩小64倍；不改变首query因果选块语义，不改变每token投影参数激活口径。
- LossModel hooks仅注册一次，只记录带梯度训练，并在计算aux后清空；profile显式检查无router图残留。MTP用sum/clamp(valid_count,1)，去掉每micro的torch.any主机同步；全忽略目标仍是0且梯度有限。训练循环指标统一步末传回CPU。

h128/L4基线先应用GPU正确性修复、persistent hooks和MTP sum，然后比较性能：220.53→158.56ms（1.39倍），aten::mm从3354降到1038次。小模型optimized记录早于C1大norm分离修复，是优化过程对照，不能当作C1同模型前后对照。C1 micro4剩余GPU时间主要为BMM/GEMM与激活/路由拷贝，没有发现先前的异常归约或逐专家退化路径。

## 正确性与端到端

GPU小case：FP32 experts输出max误差3.49e-10，输入/权重梯度约1e-11；BF16输出8.93e-6、权重梯度4.86e-6。Muon单步max差3.1e-5；强制1024元素chunk覆盖专家内/跨参数切片后max差3.98e-5，momentum buffer逐位一致。NS为BF16近似正交化，Inductor融合/舍入造成非逐位一致，对照rtol=1e-4、atol=5e-5。

全query与首query indexer score误差0、选块完全一致、QSA输出逐位一致。Liger全ignore MTP sum=0、输入/权重梯度有限。详见 q38_kernel_parity.json。

真实C1入口已跑一次optimizer step：micro4×accum8×4096=131072 token，训练语料来自原10K池，loss9.2872、MTP9.309、KL0.0011、aux5.012、gradnorm1.06、val CE9.2883，checkpoint和BF16 HF导出成功，输出 runs/q38_e2e_smoke/。此单步包含冷启动编译，日志2922tok/s不可当稳定吞吐。概要见 q38_e2e_smoke.json。

首次C1新shape编译/自动调优曾约111s，micro4首训练warmup71.55s、后续约0.43s；首推理编译约12s。冷启动与稳定吞吐分开报告，复用编译缓存后不逐步承担启动开销。

## C4显存范围

只用meta tensors统计，未实例化/测量GPU C4。1,804,657,560参数的FP32权重+梯度14.44GB，Adam两个moment4.13GB，Muon momentum5.15GB，持久状态共23.72GB（22.09GiB）。另需NS临时空间建议预留1–2GiB，以及激活、workspace、可能的DDP buckets；2GiB不是实测严格上界。每NS chunk的FP32 cat和Nesterov update各最多256MiB，避免整bucket数GB。

不能由持久状态断言40GB可训练C4；按C1激活体积粗略外推，C4连micro1都可能超过40GB。应先在80GB设备从micro1实测。普通DDP不分摊每卡参数/optimizer状态。当前trunk没有执行activation checkpointing，继承的supports_gradient_checkpointing标记不能节省显存；若仍不足需另行实现并验证checkpointing/FSDP。本次没有扩大范围实现这些机制。

## C4 activation checkpointing 实测（临时措施）

单卡 A100-40GB、seq 4096 实测：不开 checkpointing 时 C4 micro1 前向即 OOM（已分配 38.35 GiB）。按 2/4 层截断测得每层前向保存激活约 1.11 GiB/序列，20 层约 22.6 GiB，加持久状态 22.1 GiB 超出 40GB。

`--grad_ckpt 1` 对每个 decoder 层做非 reentrant checkpointing，层间只保存 mHC 残差流，router aux 只在前向记录、不重复计入重算。C1 形状、真实 FLA/Triton kernel 下开关前后 4 项 loss 相同、梯度相对差 0，C1 micro2 峰值 10.71→4.07 GiB。C4 结果（每 micro 做一次 optimizer step，峰值含优化器状态，不含 DDP buckets）：

| C4 grad_ckpt | 峰值 allocated | 整步 ms | 训练 tok/s |
|---|---:|---:|---:|
| micro1 | 23.73 GiB | 1192 | 3.44K |
| micro2 | 25.20 GiB | 1857 | 4.41K |
| micro4 | 29.28 GiB | 3088 | 5.31K |
| micro8 | 38.03 GiB | 5736 | 5.71K |

原始数据见 q38_c4_ckpt_micro{1,2,4,8}.json。重算约多一次前向，只影响墙钟时间，**不计入 FLOPs 核算**。这是为在 40GB 卡上放下 C4 训练的临时措施，正式提交代码前需改回（C4 预设 `grad_ckpt` 置 0 或移除）。普通 DDP 每卡另有约 6.7 GiB 梯度 bucket，多卡 micro_bs 以 `--auto_micro_bs` 探测为准。

## 复现与范围

```bash
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0
"$PY" scripts/profile_q38.py --preset q38_c1 --seq 4096 --micro-bs 4 --output audits/q38_c1_micro4.json --profile
"$PY" scripts/check_q38_kernels.py
```

只覆盖本机A100的C1 packed prefill/单卡训练，未测H100、C2–C4、多卡通信和完整bpb。已修复有证据的退化路径和异常归约，不能保证任意硬件/模型都没有瓶颈。
