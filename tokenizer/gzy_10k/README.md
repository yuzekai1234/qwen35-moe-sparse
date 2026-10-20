# guozeyuan 10K 词表快照

- 来源：`/home/shared/llm_pt/cache/tokenizers/ling-no-mtp-c1-top10k-v1`。
- 作者／实验目录：guozeyuan；本仓库原样复制 tokenizer 文件，不使用其模型权重。
- 实际词表大小：10112；BOS=1、EOS=53、PAD=0、UNK=3。
- 由 EXAONE 153600 词表按官方训练语料 token 频率裁剪，保留必要特殊 token、字节 token 与 BPE 合并依赖；完整选择记录及来源指纹见 `pruning.json`。
- 文件：`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`pruning.json`。

加载使用 `PreTrainedTokenizerFast.from_pretrained("tokenizer/gzy_10k", local_files_only=True)`。语料需要重新分词，保存为 little-endian uint32；不可复用原 EXAONE token IDs。
