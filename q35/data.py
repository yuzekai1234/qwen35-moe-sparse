"""Token corpora: little-endian uint32 token files with EOS between documents (any vocabulary size).

默认训练 / 验证语料是共享目录里用 gzy 30K 词表分好的语料 (DEFAULT_DATA / DEFAULT_VAL_DATA), 无需本地准备;
下面的 corpus 命令用于为其他词表或抽样规则自行准备语料。
python -m q35.data corpus --out data/corpus_gzy30k [--data_dir /home/shared/data --every 12 --sub 6 --workers 8]
    从 8 个官方训练分片均匀抽取 1/every 文档 -> 分词 -> 每个子任务各写一个 .bin -> 按随机顺序拼成 train.bin，
    第一块切出 val.bin。分片可以是 dolma3_mix_50B-*.jsonl.zst 或 .jsonl。--every 12 ≈ 1/12 文档；--every 1 为全量。
    子任务结果落盘而不是经进程间管道传回（大对象经管道会把 Pool 弄死），已完成的分片可续跑。
python -m q35.data sample --out data/smoke_gzy30k/train.bin [--tokens 262144]
    从一个训练分片开头取少量 token 供 smoke 测试；不读取官方评测数据。
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOKEN_DTYPE = np.dtype("<u4")
DEFAULT_TOKENIZER = ROOT / "tokenizer/gzy_30k"
SHARED_CACHE = Path("/home/shared/llm_pt/cache")
DEFAULT_DATA = SHARED_CACHE / "ling-no-mtp-c1-global-pruned30k-isoflops-v1/train.bin"
DEFAULT_VAL_DATA = SHARED_CACHE / "ling-no-mtp-c1-holdout-pruned30k-isoflops-v1/valid.bin"
CHUNK_CHARS = 100_000   # 超长文档按 10 万字符切块再分词(内容不变; HF 分词器对单篇数 MB 文本极慢)


def load_tokenizer(path):
    # Load tokenizer.json verbatim: AutoTokenizer would swap in the Qwen pre-tokenizer for this model type.
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)


def tokenizer_sha256(path):
    return hashlib.sha256((Path(path) / "tokenizer.json").read_bytes()).hexdigest()


def recorded_tokenizer_sha(meta):
    """tokenizer.json SHA256 recorded by q35 metadata, or by llm_pt manifests under (spec.)tokenizer.files."""
    if "tokenizer_sha256" in meta:
        return meta["tokenizer_sha256"]
    tok = meta.get("tokenizer") if isinstance(meta.get("tokenizer"), dict) else meta.get("spec", {}).get("tokenizer", {})
    return tok.get("files", {}).get("tokenizer.json")


def open_tokens(path, tokenizer_sha=None):
    """Memory-map a token file; refuse corpora whose metadata records a different tokenizer."""
    path = Path(path)
    for name in ("meta.json", "parts_manifest.json", "manifest.json"):
        meta = path.parent / name
        if tokenizer_sha and meta.exists():
            if recorded_tokenizer_sha(json.loads(meta.read_text())) not in (None, tokenizer_sha):
                raise ValueError("Corpus tokenizer mismatch: prepare fresh data for this tokenizer")
    if path.stat().st_size % TOKEN_DTYPE.itemsize:
        raise ValueError("Misaligned binary token file")
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def _chunks(text):
    return [text[j:j + CHUNK_CHARS] for j in range(0, max(1, len(text)), CHUNK_CHARS)]


TOK = None


def _init_worker(tokenizer):
    global TOK
    TOK = load_tokenizer(tokenizer)


def _tokenize_part(task):
    shard, every, offset, outfile = task
    if os.path.exists(outfile + ".ok") and os.path.exists(outfile):
        return outfile, 0, 0
    p = subprocess.Popen(["zstd", "-dc", shard] if shard.endswith(".zst") else ["cat", shard], stdout=subprocess.PIPE)
    eos, buf, n_docs, n_tok = TOK.eos_token_id, [], 0, 0
    with open(outfile, "wb") as w:
        def flush():
            nonlocal buf, n_docs, n_tok
            for ids in TOK(buf, add_special_tokens=False)["input_ids"]:
                arr = np.asarray(ids + [eos], dtype=TOKEN_DTYPE)
                w.write(arr.tobytes()); n_docs += 1; n_tok += len(arr)
            buf = []
        for i, line in enumerate(p.stdout):
            if (i + offset) % every:
                continue
            for chunk in _chunks(json.loads(line)["text"]):
                buf.append(chunk)
                if len(buf) >= 512:
                    flush()
        if buf:
            flush()
    if rc := p.wait():
        raise RuntimeError(f"Failed reading {shard}: exit {rc}")
    open(outfile + ".ok", "w").write(f"{n_docs} {n_tok}")
    return outfile, n_docs, n_tok


def prepare_corpus(tokenizer, out, data_dir, every=12, sub=6, workers=8, val_tokens=30_000_000, seed=0):
    t0 = time.time()
    out = str(out)
    os.makedirs(f"{out}/parts", exist_ok=True)
    # Refuse to reuse part caches produced with a different tokenizer or sampling rule.
    manifest = {"every": every, "sub": sub, "data_dir": os.path.realpath(data_dir), "tokenizer_sha256": tokenizer_sha256(tokenizer)}
    manifest_path = os.path.join(out, "parts_manifest.json")
    if os.path.exists(manifest_path):
        if json.load(open(manifest_path)) != manifest:
            raise ValueError("Part cache mismatch: choose a fresh --out directory")
    else:
        if os.listdir(os.path.join(out, "parts")):
            raise ValueError("Unverified existing parts: choose a fresh --out directory")
        json.dump(manifest, open(manifest_path, "w"), indent=2)
    shards = sorted(f for f in os.listdir(data_dir) if f.startswith("dolma3_mix_50B-") and f.endswith((".jsonl.zst", ".jsonl")))
    if not shards:
        raise ValueError("No dolma3_mix_50B JSONL shards found")
    tasks = [(os.path.join(data_dir, s), every * sub, k, f"{out}/parts/{s.split('.jsonl')[0]}_{k}.bin") for s in shards for k in range(sub)]
    with Pool(min(workers, len(tasks)), initializer=_init_worker, initargs=(str(tokenizer),)) as pool:
        for i, (f, nd, nt) in enumerate(pool.imap_unordered(_tokenize_part, tasks)):
            print(f"任务 {i+1}/{len(tasks)} 完成 {os.path.basename(f)} {nd} 篇 {nt/1e6:.0f}M token  {time.time()-t0:.0f}s", flush=True)
    parts = [t[3] for t in tasks]
    random.Random(seed).shuffle(parts)
    stats = {os.path.basename(p): tuple(map(int, open(p + ".ok").read().split())) for p in parts}
    n_tok = sum(v[1] for v in stats.values()); n_docs = sum(v[0] for v in stats.values())
    if n_tok <= val_tokens:
        raise ValueError("Corpus is smaller than val_tokens")
    if os.path.getsize(parts[0]) // TOKEN_DTYPE.itemsize < val_tokens:
        raise ValueError("First shuffled part must contain val_tokens; reduce --val_tokens")
    with open(f"{out}/val.bin", "wb") as wv, open(f"{out}/train.bin", "wb") as wt:
        first = np.fromfile(parts[0], dtype=TOKEN_DTYPE)
        wv.write(first[:val_tokens].tobytes()); wt.write(first[val_tokens:].tobytes())
        for p in parts[1:]:
            wt.write(open(p, "rb").read())
    json.dump({"shards": shards, "every": every, "sub": sub, "docs": n_docs, "tokens": n_tok, "val_tokens": val_tokens,
               "tokenizer": str(tokenizer), "tokenizer_sha256": manifest["tokenizer_sha256"], "dtype": TOKEN_DTYPE.str,
               "part_order": [os.path.basename(p) for p in parts]}, open(f"{out}/meta.json", "w"), indent=1)
    print(f"共 {n_docs} 篇 {n_tok/1e9:.2f}B token; train.bin {os.path.getsize(out+'/train.bin')/1e9:.2f} GB, "
          f"val.bin {os.path.getsize(out+'/val.bin')/1e6:.0f} MB  {time.time()-t0:.0f}s", flush=True)


def prepare_sample(tokenizer, out, source, tokens=262144):
    tok = load_tokenizer(tokenizer)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(source) as src, out.open("wb") as w:
        for line in src:
            for chunk in _chunks(json.loads(line)["text"]):
                ids = tok.encode(chunk, add_special_tokens=False) + [tok.eos_token_id]
                np.asarray(ids, dtype=TOKEN_DTYPE).tofile(w)
                count += len(ids)
                if count >= tokens:
                    break
            if count >= tokens:
                break
    if count < tokens:
        raise ValueError(f"Source exhausted: {count} < {tokens}")
    meta = {"source": str(source), "tokens": count, "path": str(out), "dtype": TOKEN_DTYPE.str,
            "tokenizer": str(tokenizer), "tokenizer_sha256": tokenizer_sha256(tokenizer)}
    (out.parent / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus", help="sampled training corpus -> train.bin / val.bin")
    c.add_argument("--out", required=True)
    c.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    c.add_argument("--data_dir", default="/home/shared/data")
    c.add_argument("--every", type=int, default=12)
    c.add_argument("--sub", type=int, default=6)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--val_tokens", type=int, default=30_000_000)
    c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sample", help="small token sample for smoke tests")
    s.add_argument("--out", required=True)
    s.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    s.add_argument("--source", default="/home/shared/data/dolma3_mix_50B-00000.jsonl")
    s.add_argument("--tokens", type=int, default=262144)
    a = ap.parse_args()
    if a.cmd == "corpus":
        if min(a.every, a.sub, a.workers, a.val_tokens) < 1:
            ap.error("sampling, workers and val_tokens must be positive")
        prepare_corpus(a.tokenizer, a.out, a.data_dir, a.every, a.sub, a.workers, a.val_tokens, a.seed)
    else:
        prepare_sample(a.tokenizer, a.out, a.source, a.tokens)


if __name__ == "__main__":
    main()
