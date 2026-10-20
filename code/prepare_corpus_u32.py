"""从 8 个官方分片均匀抽样文档 -> 分词 -> 每个子任务各写一个 .bin -> 按随机顺序拼成 train.bin，另切 val.bin（uint32，文档间插 EOS；154k 词表必须 uint32）。
用法(我们 Qwen3.5 sparse 用的语料): prepare_corpus_u32.py --tokenizer /home/shared/tokenizer --data_dir /home/shared/data --out ../data/corpus_exaone154k --every 12 --sub 6 --workers 48 --val_tokens 30000000
分片可以是 dolma3_mix_50B-*.jsonl.zst 或未压缩的 .jsonl（/home/shared/data 下是 .jsonl）。
--every 12 ≈ 1/12 文档 ≈ 45 亿 token；--every 1 为全量。子任务结果落盘而不是经进程间管道传回（大对象经管道会把 Pool 弄死）。"""
import argparse, os, json, subprocess, time, random, hashlib, numpy as np
from multiprocessing import Pool
ap = argparse.ArgumentParser(); ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True); ap.add_argument("--every", type=int, default=12)
ap.add_argument("--workers", type=int, default=48); ap.add_argument("--val_tokens", type=int, default=30_000_000); ap.add_argument("--data_dir", default="../data"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sub", type=int, default=6)
a = ap.parse_args()
if min(a.every, a.sub, a.workers, a.val_tokens) < 1: ap.error("sampling, workers and val_tokens must be positive")
os.makedirs(f"{a.out}/parts", exist_ok=True)
# Refuse to reuse part caches produced with a different tokenizer or sampling rule.
manifest = {"every": a.every, "sub": a.sub, "data_dir": os.path.realpath(a.data_dir),
            "tokenizer_sha256": hashlib.sha256(open(os.path.join(a.tokenizer, "tokenizer.json"), "rb").read()).hexdigest()}
manifest_path = os.path.join(a.out, "parts_manifest.json")
if os.path.exists(manifest_path):
    if json.load(open(manifest_path)) != manifest: raise ValueError("Part cache mismatch: choose a fresh --out directory")
else:
    if os.listdir(os.path.join(a.out, "parts")): raise ValueError("Unverified existing parts: choose a fresh --out directory")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
TOK = None
def init():
    global TOK
    from transformers import AutoTokenizer; TOK = AutoTokenizer.from_pretrained(a.tokenizer)
def work(args):
    shard, every, offset, outfile = args
    if os.path.exists(outfile + ".ok") and os.path.exists(outfile): return outfile, 0, 0
    p = subprocess.Popen(["zstd", "-dc", shard] if shard.endswith(".zst") else ["cat", shard], stdout=subprocess.PIPE); eos = TOK.eos_token_id; buf = []; n_docs = n_tok = 0
    with open(outfile, "wb") as w:
        def flush():
            nonlocal buf, n_docs, n_tok
            for ids in TOK(buf, add_special_tokens=False)["input_ids"]:
                arr = np.asarray(ids + [eos], dtype=np.uint32); w.write(arr.tobytes()); n_docs += 1; n_tok += len(arr)
            buf = []
        for i, line in enumerate(p.stdout):
            if (i + offset) % every: continue
            t = json.loads(line)["text"]
            for j in range(0, max(1, len(t)), 100_000):   # 超长文档按 10 万字符切块再分词(内容不变; HF 分词器对单篇数 MB 文本极慢)
                buf.append(t[j:j+100_000])
                if len(buf) >= 512: flush()
        if buf: flush()
    rc = p.wait()
    if rc: raise RuntimeError(f"Failed reading {shard}: exit {rc}")
    open(outfile + ".ok", "w").write(f"{n_docs} {n_tok}"); return outfile, n_docs, n_tok
if __name__ == "__main__":
    t0 = time.time(); shards = sorted(f for f in os.listdir(a.data_dir) if f.startswith("dolma3_mix_50B-") and f.endswith((".jsonl.zst", ".jsonl")))
    if not shards: raise ValueError("No dolma3_mix_50B JSONL shards found")
    tasks = [(os.path.join(a.data_dir, s), a.every * a.sub, k, f"{a.out}/parts/{s.split('.jsonl')[0]}_{k}.bin") for s in shards for k in range(a.sub)]
    with Pool(min(a.workers, len(tasks)), initializer=init) as pool:
        for i, (f, nd, nt) in enumerate(pool.imap_unordered(work, tasks)):
            print(f"任务 {i+1}/{len(tasks)} 完成 {os.path.basename(f)} {nd} 篇 {nt/1e6:.0f}M token  {time.time()-t0:.0f}s", flush=True)
    parts = [t[3] for t in tasks]; random.Random(a.seed).shuffle(parts)
    stats = {os.path.basename(p): tuple(map(int, open(p + ".ok").read().split())) for p in parts}
    n_tok = sum(v[1] for v in stats.values()); n_docs = sum(v[0] for v in stats.values())
    if n_tok <= a.val_tokens: raise ValueError("Corpus is smaller than val_tokens")
    if os.path.getsize(parts[0]) // 4 < a.val_tokens: raise ValueError("First shuffled part must contain val_tokens; reduce --val_tokens")
    with open(f"{a.out}/val.bin", "wb") as wv, open(f"{a.out}/train.bin", "wb") as wt:
        first = np.fromfile(parts[0], dtype=np.uint32); wv.write(first[:a.val_tokens].tobytes()); wt.write(first[a.val_tokens:].tobytes())
        for p in parts[1:]: wt.write(open(p, "rb").read())
    json.dump({"shards": shards, "every": a.every, "sub": a.sub, "docs": n_docs, "tokens": n_tok, "val_tokens": a.val_tokens, "tokenizer": a.tokenizer, "part_order": [os.path.basename(p) for p in parts]}, open(f"{a.out}/meta.json", "w"), indent=1)
    print(f"共 {n_docs} 篇 {n_tok/1e9:.2f}B token; train.bin {os.path.getsize(a.out+'/train.bin')/1e9:.2f} GB, val.bin {os.path.getsize(a.out+'/val.bin')/1e6:.0f} MB  {time.time()-t0:.0f}s"); print("DONE", flush=True)
