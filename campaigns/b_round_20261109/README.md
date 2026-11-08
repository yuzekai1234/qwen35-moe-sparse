# B 榜工作台：2026-11-09

目标：取得可复现、可上传、可运行的最低综合得分。以四档完整 A 集 BPB 拟合 Performance，结合 C4 EvalTime 判断候选；最终由 B 榜决定排名。没有沿用旧 agent 的临时试验时限或试验次数限制。

## 已有结果和当前缺口

- 最佳已完成 C1：Engram + 无辅助损失均衡 + DenseFormer；原始 NN BPB **1.0042105367**，加混合预测后 **0.9909068648**。
- 完整训练、HF 权重、断点、训练数据和兼容环境已迁到本机根目录；原包保持不动。
- 新组合 C2/C3/C4 尚无已验证结果。K4096 的另一轮训练完成但评测缺少 `lmc_backoff/gpu-qualification.json`，不能当成新成绩。
- 网站 11-06 15:00 榜：562 第7，Performance 0.9814，EvalTime 0.35，S 0.9722。最新正式提交是 11-03 18:02；11-11 23:59 截止。网站之后可能更新。
- 11-07 群聊要求先 C4；11-08 晚要求在新机器准备并跑齐 C1–C4。其他任务占卡时先完成 CPU 准备。

## 起始配置

| 档 | 预算 FLOPs | hidden / layers | 专家 / top-k | token 数约 | 原生 GPU 数 |
|---|---:|---:|---:|---:|---:|
| C1 | 3e17 | 512 / 10 | 1024 / 4 | 10.55亿 | 2 |
| C2 | 1e18 | 640 / 12 | 1024 / 4 | 20.24亿 | 2 |
| C3 | 3e18 | 768 / 14 | 1024 / 4 | 41.47亿 | 4 |
| C4 | 1e19 | 832 / 16 | 1024 / 4 | 101.61亿 | 4 |

共同规则：Engram、loss-free routing、DenseFormer；两处 LC 位于 floor(L/3)、floor(2L/3)，通道等于 hidden，K=2048；训练4096；初始评测10240、FP8 trunk/BF16完整 logits、单卡、自动 batch。专家宽与 Engram 维度随 hidden 缩放。

这些是以 C1 结果和显存负担为依据的**起始网格**，没有声称已找到 N–D 最优值。C4 的参数/梯度/分片优化器常驻内存下界约76.9 GiB/卡，另有 FFT、激活、通信和临时内存，仍需实际显存验证。显存不足时另开小一档 run-id，或在相同全局 batch 下调整 micro batch；不能把四卡断点当成两卡断点恢复。

已准备的语料约56亿 token。C4 的预算需要跨两轮读取；新增 sampler 保留第一轮原顺序，之后 seed+epoch 重新打乱，每轮内部不重复，所有 rank 共享同一全局序列。与随机抽样不同，没有引入新数据或过滤数据。

## 入口与 CPU 检查

```bash
source /data/quark-7db0d9a22ffb/activate-c1.sh
cd /data/quark-7db0d9a22ffb/team-repo/campaigns/b_round_20261109
export PYTHONPATH="$PWD/runtime:$C1_MODEL_ROOT/compat-source"
python prepare.py --run-id combined-v1
CUDA_VISIBLE_DEVICES='' python preflight.py --run-id combined-v1 --level all
python test_campaign.py
bash launch.sh status
```

`prepare.py`按实际代码重新核算四档 FLOPs/token、整步 token 预算、跨轮次数、参数量及激活比例；保留约0.5%用于转换/校准等额外工作，额外计算仍需实际记录。它把本次代码冻结到 `runs/<run-id>/source/`，`launch.sh`从该快照启动，其他agent更新仓库不会改变已经准备的任务。`preflight.py`实际使用原生解析器、分词器、数据元信息，并以 meta 模型核对参数量，不创建 CUDA 上下文。

## 腾卡后先 C4

先确认分配给自己的 GPU 空闲，再启动。下面GPU编号是示例，需要按实际分配替换。

```bash
tmux new-session -d -s br-yzk-c4 \
  'cd /data/quark-7db0d9a22ffb/team-repo/campaigns/b_round_20261109 && bash launch.sh train --run-id combined-v1 --level c4 --owner yzk --gpus 0,1,2,3'
```

日志：`/data/quark-7db0d9a22ffb/b-round-20261109/runs/combined-v1/c4/training.log`。训练结束检查 `training-complete.json`，包含预算完成核对和 FP32 控制参数导出。`--wait`只在明确想等待指定GPU时使用，不会停止其他进程。失败后先读日志；自己的同版本断点可加 `--resume`，不得覆盖或拼接不同训练历史。

完整训练后，单独认领完整 A 集评测，释放其余卡：

```bash
tmux new-session -d -s br-yzk-c4-eval \
  'cd /data/quark-7db0d9a22ffb/team-repo/campaigns/b_round_20261109 && bash launch.sh eval --run-id combined-v1 --level c4 --owner yzk --gpus 0'
```

评测结果位于 `.../c4/evaluations/raw-s10240-auto/`；包括 `result.json`、`run-metadata.json`、`campaign-result.json`和日志。只有完整120142976计分token结果才登记完成。重复评测使用新的 `--eval-tag`，保留原始输出。

随后分配 C2、C3。C1 可先复测已迁移的现有权重，避免无必要重训：

```bash
bash launch.sh eval --run-id combined-v1 --level c1 --owner csy --gpus 4 \
  --model "$C1_HF_MODEL" --eval-tag raw-baseline-s10240-auto
```

所有四档应使用相同源版本和评测 profile，不能把C1混合预测与其他档原始NN混在一个正式比较里。支持 `eval --hybrid-spec /path/spec.json`；必须属于该候选自己的模型、记忆和校准。新档混合预测的记忆重建、校准和因果验证仍是待做工作。

## 分工建议与优化顺序

1. 一名 agent 负责 C4：GPU显存/内核可运行性、完整训练和完整A结果，先取得可用四档主线的最大档证据。
2. 一至两名 agent 分别负责 C2、C3；C1复测和导出核对可用另一张已分配空卡。通过同一状态库认领，避免重复训练。
3. 推理 agent 对已冻结权重测 batch/上下文/算子组合；报告 BPB 与 forward秒的共同变化，保留 raw 和 hybrid 两个端点。C1历史计时受其他进程干扰，需重新取得同机独占计时。
4. 质量 agent 搜索专家容量、宽深、学习率、训练长度等；新增候选用新run-id，例如 `python prepare.py --run-id e768-k2048 --experts 768`。K4096也可生成但需实测，先复用已完成断点排除旧评测缺文件问题。延迟启用/冻结LC仍属提议，不能当成已验证改进。
5. 四档齐全后计算总分；保留 Pareto 候选。优先评估能改善四档质量的方案，再判断推理速度改进是否足以补偿质量变化。
6. 提交 agent 组装四个 HF 模型、统一 `evaluation/eval.sh`、MODEL_CARD、训练代码和日志、依赖/Docker及复现命令。用全新输出目录、断网容器实际跑通，保留提交回执和旧可用包。

## 总分计算

11-06规则：`S = Performance * EvalTime^lambda`，
`lambda = log(P_Q3/P_Q1)/(5*log(T_Q3/T_Q1))`。权重依赖整榜分布。

创建一个JSON文件，将 `c1,c2,c3,c4`映射到各自的 `campaign-result.json`，然后：

```bash
python score.py --suite /path/to/suite.json --output /path/to/score.json
```

脚本在log BPB空间等权拟合 `E+k*(C/3e17)^(-p)`，计算固定预算区间的曲线面积与R²，并替换榜单562队后重算全部队伍的四分位数、权重和排名。默认榜单为已保存的11-06展示值；正式比较前可用 `--board`更新，不能把旧榜情景当成B榜预测。

## 交接

先读本文件、`AGENTS.md`、共享 `BOARD.md`和实时status；每个结果保留source/config哈希、完整命令、模型路径、GPU、原始日志和状态。仓库只同步代码/配置/摘要，不放密钥、权重、数据或环境。本目录导入代码的原始哈希见 `provenance/imported-source.json`，可移植性改动与训练流改动都在Git中可审查。
