# guozeyuan 50K 词表快照

- 来源：`/home/shared/llm_pt/cache/tokenizers/ling-no-mtp-c1-top50k-v1`（`/home/shared/qwen35_moe_sparse/tokenizers/50048` 为同一份拷贝）。
- 作者／实验目录：guozeyuan；本仓库原样复制 tokenizer 文件，不使用其模型权重。
- 实际词表大小：50048；BOS=1、EOS=53、PAD=0、UNK=3。
- 由 EXAONE 153600 词表按官方训练语料 token 频率裁剪，保留必要特殊 token、字节 token 与 BPE 合并依赖；完整选择记录及来源指纹见 `pruning.json`。
- 已分好的语料（默认训练 / 验证数据）：
  - 训练 `/home/shared/llm_pt_ablation/cache/ling-no-mtp-c1-global-pruned50k-scale-isoflops-v1/train.bin`：3.56B token，全局随机抽取文档（seed 1101，无放回，无清洗）。
  - 验证 `/home/shared/llm_pt_ablation/cache/ling-no-mtp-c1-holdout-pruned50k-scale-isoflops-v1/valid.bin`：78M token，按文本哈希独立留出的 1% 文档，与训练集不相交，约 4.63 字节/token。
  - 两份 `manifest.json` 记录的 `tokenizer.json` SHA256 与本目录一致，训练启动时校验。

加载使用 `PreTrainedTokenizerFast.from_pretrained("tokenizer/gzy_50k", local_files_only=True)`。
