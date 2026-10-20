# 4090-8 部署验证

日期：2026-10-20（Asia/Shanghai）。目录：`/home/yzk/qwen35-moe-sparse`。本次仅部署代码并进行短程验证，未启动 C1/C2 完整预算预训练，也未重新复现文档中的 bpb。

## 环境

Python 3.11.17、torch 2.7.1+cu126、transformers 5.19.0、flash-linear-attention 0.5.2、liger-kernel 0.8.4。完整环境冻结见 `environment-4090-8.txt`。安装入口默认清华源，独立 `.venv_q35`，未修改其他项目环境。

运行输出确认 GDN 使用 `fla.ops.gated_delta_rule.chunk`。与上游依赖一致，未安装可选 `causal_conv1d`；卷积使用正确但较慢的 PyTorch 路径，因此本次性能不应与文档 EvalTime 直接比较。

## 验证内容

| 检查 | 结果 |
|---|---|
| 五组配置的参数/FLOPs 与历史记录 | 全部一致 |
| 单双卡启动命令路径与参数覆盖 | 通过 |
| 小语料抽样、uint32 长度、缓存复用/不匹配拒绝 | 通过 |
| 单卡 C2-m，真实 EXAONE 词表，seq=256 | 两步、16384 token，HF 导出和 checkpoint 成功 |
| 单卡 C2-m，seq=8192 | 一步、8192 token；训练 loss=11.9870，验证 loss=12.0167；显存峰值约 6.4 GiB |
| DDP 断点状态恢复 | rank 0 / rank 1 的模型、优化器、NumPy/Torch/CUDA 随机状态与 checkpoint 逐项完全一致 |
| DDP 断点恢复 vs 连续训练 | token 计数、训练步数、最终各 rank 采样状态一致，参数均为有限值 |
| HF 模型重新加载与官方评测适配脚本 | 在两条独立示例、30 个文本 token 上完成 bpb/EvalTime 计算；仅验证流程 |
| 实际训练 FLOPs | 8192-seq 短训为 8192 × 321447456 = 2633297559552 |

`python -m unittest discover -s tests -v` 共三项测试全部通过。双卡测试使用 GPU 2、6，完整 C2-m 架构（142965680 参数），seq=256、全局 batch=512 token，在第 1 步保存/退出并恢复到第 2 步。`Q35_VERIFY_RESUME=1` 在读取 checkpoint 后、继续训练前执行精确状态核对。

## 数值复现边界

当前 BF16/FLA/MoE 运行环境观察到非逐位复现：两个独立、相同种子、相同数据和卡数的全新一步运行，最终权重最大绝对差约 0.006995，采样随机状态相同。具体产生差异的 kernel 尚未单独定位。恢复测试因此以**加载时模型/优化器/随机状态与 checkpoint 的精确一致性**、采样推进与 token 预算为验收条件，同时记录最终权重差异，不宣称连续训练与恢复训练权重逐位相同。

## 产物与复跑

- 机器可读结果：`validation-results.json`。
- 服务器日志：`smoke.log`、`smoke-8192.log`、`ddp-resume.log`、`unit-tests.log`、`eval-smoke.log`。
- 模型、checkpoint 与训练指标：`runs/smoke/`、`runs/smoke_8192/`、`runs/ddp_resume/`、`runs/ddp_reference/`。
- 语料仅准备了 `data/smoke/train.bin`（264368 token）；完整官方训练分片已在 `/home/shared/data`，完整约 1/12 抽样训练集的生成命令见 README。
- 后续正式训练须先准备完整训练语料并选择空闲卡；本次未启动长时间后台训练。
- GitHub 为私有仓库；代码、词表与历史配置提交云端，训练数据/环境/权重不提交。
