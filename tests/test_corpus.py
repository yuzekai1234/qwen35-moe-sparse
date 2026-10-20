"""Requires the installed model environment; does not use shared training files."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Corpus(unittest.TestCase):
    def test_sample_counts_cache_reuse_and_cache_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "source"
            data.mkdir()
            source = data / "dolma3_mix_50B-00000.jsonl"
            source.write_text("".join(json.dumps({"text": f"Example document {i}: test English and 中文训练数据。"}) + "\n" for i in range(100)))
            output = Path(tmp) / "tokens"
            cmd = [sys.executable, str(ROOT / "code/prepare_corpus_u32.py"),
                   "--tokenizer", str(ROOT / "tokenizer/exaone_154k"),
                   "--data_dir", str(data), "--out", str(output),
                   "--every", "2", "--sub", "2", "--workers", "2", "--val_tokens", "64"]
            subprocess.run(cmd, check=True, capture_output=True)
            meta = json.loads((output / "meta.json").read_text())
            self.assertEqual(meta["docs"], 50)
            self.assertEqual((output / "val.bin").stat().st_size, 64 * 4)
            self.assertEqual((output / "train.bin").stat().st_size // 4 + 64, meta["tokens"])
            original = (output / "train.bin").read_bytes()
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual((output / "train.bin").read_bytes(), original)
            cmd[cmd.index("--every") + 1] = "3"
            failed = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Part cache mismatch", failed.stderr)


if __name__ == "__main__":
    unittest.main()
