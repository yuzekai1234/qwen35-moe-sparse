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

## gzy 10K 词表分支

- 将 guozeyuan 的 10112 词表快照及裁剪记录纳入 `tokenizer/gzy_10k/`，默认训练、预处理、smoke 与启动器使用该词表；模型形状及历史配置归档保留。
- 新运行目录带 `_gzy10k`，完整语料为 `data/corpus_gzy10k/`，避免与 153600 词表缓存及 checkpoint 混用。
- 模型词表大小及 BOS/EOS/PAD 从 tokenizer 读取；语料始终按 little-endian uint32 读取，修复小词表被误判为 uint16 的问题。
- 启动器按实际词表、序列长度和 batch 重新核算 C1/C2 的完整步数、token 数及 warmup；显式 `--tokens` 仍可覆盖预算用于短跑。
- 训练记录 tokenizer 内容 SHA256，检查语料元数据和续训词表身份。HF 导出携带 10K tokenizer，现有评测直接使用导出词表。
- 增加小词表语料、缓存不匹配、预算与命令行覆盖、CPU 前后向及导出重新加载验证。

## 模型实现纳入版本管理

- 增加 `models/qwen3_5_moe/`，从环境中固定的 Transformers 5.19.0 提取模型与配置源码，保留上游版权、Apache-2.0 LICENSE 和原始 SHA256。
- 训练和评测显式使用本地类；不修改 site-packages。运行配置保存模型源码哈希。
- 使用独立文本 model_type；HF 导出自动保存本地模型/配置代码与 auto_map，外部 AutoModel 加载需显式启用这些代码。
- 模型计算结构保持原样；新增本地/上游权重兼容、GPU 前向对齐、导出/加载和防止静默回退的测试。
- 本地模型显式保留原版初始化遍历，避免 Transformers 对外部代码的初始化兼容分支跳过 MoE 父模块初始化；测试要求同种子初始权重逐项相等。
