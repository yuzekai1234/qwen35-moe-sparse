"""Prepare a small real training sample; deliberately never read official eval data."""
import argparse
import json
from pathlib import Path
import numpy as np
from transformers import PreTrainedTokenizerFast

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default="/home/shared/data/dolma3_mix_50B-00000.jsonl")
parser.add_argument("--out", type=Path, default=root / "data/smoke/train.bin")
parser.add_argument("--tokens", type=int, default=262144)
args = parser.parse_args()
tok = PreTrainedTokenizerFast.from_pretrained(root / "tokenizer/exaone_154k")
args.out.parent.mkdir(parents=True, exist_ok=True)
count = 0
with open(args.source) as source, args.out.open("wb") as out:
    for line in source:
        text = json.loads(line)["text"]
        for start in range(0, max(1, len(text)), 100000):
            ids = tok.encode(text[start:start + 100000], add_special_tokens=False) + [tok.eos_token_id]
            np.asarray(ids, dtype="<u4").tofile(out)
            count += len(ids)
            if count >= args.tokens:
                break
        if count >= args.tokens:
            break
if count < args.tokens:
    raise ValueError(f"Source exhausted: {count} < {args.tokens}")
print(json.dumps({"source": args.source, "tokens": count, "path": str(args.out)}))
