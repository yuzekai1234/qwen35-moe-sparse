"""Corpus preparation on a synthetic shard; does not use shared training files."""
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from q35.data import ROOT, load_tokenizer, open_tokens, tokenizer_sha256


class Corpus(unittest.TestCase):
    def test_sample_counts_cache_reuse_and_cache_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "source"
            data.mkdir()
            source = data / "dolma3_mix_50B-00000.jsonl"
            source.write_text("".join(json.dumps({"text": f"Example document {i}: test English and 中文训练数据。"}) + "\n" for i in range(100)))
            output = Path(tmp) / "tokens"
            cmd = [sys.executable, "-m", "q35.data", "corpus",
                   "--tokenizer", str(ROOT / "tokenizer/gzy_10k"),
                   "--data_dir", str(data), "--out", str(output),
                   "--every", "2", "--sub", "2", "--workers", "2", "--val_tokens", "64"]
            subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
            meta = json.loads((output / "meta.json").read_text())
            self.assertEqual(meta["docs"], 50)
            self.assertEqual((output / "val.bin").stat().st_size, 64 * 4)
            self.assertEqual((output / "train.bin").stat().st_size // 4 + 64, meta["tokens"])
            # Small vocabularies still use uint32, with real EOS IDs and valid text.
            tok = load_tokenizer(ROOT / 'tokenizer/gzy_10k')
            train_ids = np.fromfile(output / 'train.bin', dtype='<u4')
            val_ids = np.fromfile(output / 'val.bin', dtype='<u4')
            self.assertLess(max(train_ids.max(), val_ids.max()), len(tok))
            self.assertEqual(np.count_nonzero(train_ids == tok.eos_token_id)
                             + np.count_nonzero(val_ids == tok.eos_token_id), 50)
            self.assertEqual(meta['dtype'], '<u4')
            original = (output / "train.bin").read_bytes()
            subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
            self.assertEqual((output / "train.bin").read_bytes(), original)
            cmd[cmd.index("--every") + 1] = "3"
            failed = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Part cache mismatch", failed.stderr)
            cmd[cmd.index("--every") + 1] = "2"
            # Same vocabulary, different tokenizer.json contents.
            other = Path(tmp) / "other_tok"
            shutil.copytree(ROOT / "tokenizer/gzy_10k", other)
            with (other / "tokenizer.json").open("a") as f:
                f.write("\n")
            cmd[cmd.index("--tokenizer") + 1] = str(other)
            failed = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Part cache mismatch", failed.stderr)
            # Training refuses a corpus prepared with another tokenizer.
            self.assertEqual(len(open_tokens(output / "train.bin", tokenizer_sha256(ROOT / "tokenizer/gzy_10k"))),
                             meta["tokens"] - 64)
            with self.assertRaisesRegex(ValueError, "tokenizer mismatch"):
                open_tokens(output / "train.bin", tokenizer_sha256(other))


if __name__ == "__main__":
    unittest.main()
