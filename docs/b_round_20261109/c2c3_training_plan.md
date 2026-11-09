# 配套 C2 C3 训练安排

本次负责补齐同学正在训练的 E512、LC3、batch乘1.5 架构的 C2/C3。原 combined-v6 C4继续保持暂停；新C4由同学在gpu1运行，不由本任务重启或替换。

## 配置

| 项目 | C2 | C3 |
|---|---:|---:|
| 预算FLOPs | 1e18 | 3e18 |
| hidden / layers | 640 / 12 | 768 / 14 |
| 专家 / top-k | 512 / 4 | 512 / 4 |
| LC位置 零起始索引 | 4,8,10 | 4,9,12 |
| LC taps / channels | 2048 / 640 | 2048 / 768 |
| NN总参数 | 4,768,909,810 | 7,246,372,287 |
| 全局batch token | 98,304 | 196,608 |
| 训练token | 2,050,523,136 | 4,194,631,680 |
| 优化器步数 | 20,859 | 21,335 |
| 计划world / micro batch | 8 / 3 | 8 / 2 |
| 梯度累积 | 1 | 3 |

保留Engram、loss-free balancing和DWA；LC优化器状态留GPU，FFN重计算和优化器分片启用。数据、tokenizer、4096训练序列、种子与尾段调度沿用当前家族；根据各自实际FLOPs/token重算训练量，预留0.5%预算。C2/C3所用runtime与正在运行的C4逐Python文件比较一致。新节点必须再次检查GPU型号、拓扑、显存和实际占用。

## 顺序与工期

如果新机八卡均可用，先C3再C2，使用全部八卡串行，避免四加四分配在较小任务结束后闲置一半资源。C4的实测吞吐约13.7–14.6万token/秒；以当前计账FLOPs折算同等有效计算效率，C3约7.1–7.5小时、C2约2.4–2.5小时。这是跨配置估计；迁移、首次编译、检查点、最终验证和导出另计，新机启动后用稳态日志更新ETA。

群里给出的新服务器使用安排为先到11月10日晚，未给出精确回收时刻。目标是在11月10日白天完成训练并留出评测/导出余量。不能在未看到实际GPU和吞吐前承诺上述时间。

## 新节点准备

目标数据根目录为 `/models/q38-b-round-20261109`。需要迁移约22.47GB train.bin、313.53MB valid.bin、455.68MB完整A JSONL、tokenizer及其SHA清单；环境使用已验证的Python3.12.16、Torch2.8.0+cu128和相同依赖。环境包包含可移植Python解释器及必要兼容源码，解包后修正venv解释器链接和激活路径。复制成功之后比较SHA256，再运行CPU预检和GPU试运行。

VPN和SSH凭据只放专用私有配置目录，不进入仓库、共享说明或日志。接入状态与服务器资源清单同步；VPN通并不等于SSH或GPU可用。

## 可复现入口

```bash
python prepare_c2c3_matched.py --workspace /models/q38-b-round-20261109 \
  --run-id matched-lc3-e512-b15-c23-20261109 --world 8

# 必须使用目标环境并隐藏GPU执行CPU预检；分别检查两个档位。
CUDA_VISIBLE_DEVICES='' python preflight.py --workspace /models/q38-b-round-20261109 \
  --run-id matched-lc3-e512-b15-c23-20261109 --level c3
CUDA_VISIBLE_DEVICES='' python preflight.py --workspace /models/q38-b-round-20261109 \
  --run-id matched-lc3-e512-b15-c23-20261109 --level c2

# 在目标机实时查卡、认领和tmux中运行；先C3，成功后再C2。
Q38_WORKSPACE=/models/q38-b-round-20261109 bash launch.sh train \
  --run-id matched-lc3-e512-b15-c23-20261109 --level c3 \
  --owner codex-gzy --gpus 0,1,2,3,4,5,6,7
```

每个档位分别保存配置哈希、source哈希、GPU UUID、日志和原生断点。完整训练完成后，才导出HF并认领统一设置的完整A评测。新C1目前的结果是raw BPB 1.01100120 / EvalTime 0.56475571；不能用旧E1024 C1的hybrid 0.99090686替代同族raw端点。

创新说明与文献见 `architecture_and_evidence.md`。逐样本分析工具为 `analyze_longconv_pairs.py`，它需要配对NLL输入；当前尚无可据此导出的真实提升样本集。
