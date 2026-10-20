# 相对共享源码的修改

- 训练结构、五组原始配置、词表与官方评测计分方式保留。
- 增加五组配置统一入口 `code/launch.py` 与 `launch/train.sh`；支持单卡、torchrun DDP、命令行覆盖、可见 GPU 数自动检测和失败退出。
- 安装默认使用清华源；安装失败立即返回；libcuda 软链位于项目目录。
- 明确用 `PreTrainedTokenizerFast` 加载 EXAONE，校验词表 153600；训练关闭 KV cache。
- 训练二进制数据改为只读 mmap，避免每个 DDP rank 各读入一份完整语料。
- 增加 `Q35_VERIFY_RESUME=1`，在恢复时逐项核对模型、优化器和各 rank 随机状态与 checkpoint 完全一致，用于集成测试。
- checkpoint 保存各 rank NumPy / Torch / CUDA 随机状态，正常结束也保存 FP32 主权重；原子替换 checkpoint。
- 续训校验关键模型、数据和优化参数，防止不兼容地覆盖原运行；允许改变卡数与 micro batch。
- `max_steps` 保持原 token 预算的学习率曲线，summary 标注预算是否完成。
- 语料处理增加缓存配置校验、输入/分片检查及子进程失败检测。
- FLOPs 脚本读取实际序列长度和实际训练 token 数。
- 增加真实训练语料 smoke 脚本、基线参数/FLOPs 测试、中文部署说明和 Git 忽略规则。

官方评测适配版 `eval_q35.py` 未改评分逻辑。原始作者与上游来源见 README。

## 模型实现纳入版本管理

- 增加 `models/qwen3_5_moe/`，从环境中固定的 Transformers 5.19.0 提取模型与配置源码，保留上游版权、Apache-2.0 LICENSE 和原始 SHA256。
- 训练和评测显式使用本地类；不修改 site-packages。运行配置保存模型源码哈希。
- 使用独立文本 model_type；HF 导出自动保存本地模型/配置代码与 auto_map，外部 AutoModel 加载需显式启用这些代码。
- 模型计算结构保持原样；新增本地/上游权重兼容、GPU 前向对齐、导出/加载和防止静默回退的测试。
- 本地模型显式保留原版初始化遍历，避免 Transformers 对外部代码的初始化兼容分支跳过 MoE 父模块初始化；测试要求同种子初始权重逐项相等。
