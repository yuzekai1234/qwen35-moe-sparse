"""Presets reproduce the historical runs, and budgets are recalculated for the real vocabulary."""
import unittest

from tokenizers import Tokenizer

from q35.data import DEFAULT_TOKENIZER
from q35.flops import count
from q35.presets import ROOT, default_layer_types, load_preset, model_config, preset_budget, preset_names
from q35.train import parse_args
from scripts.audit_flops import legacy_train_flops_per_token

NAMES = ("q35_c1_s1", "q35_c1_s2", "q35_c1_s3", "q35_c2_b", "q35_c2_m")


class Presets(unittest.TestCase):
    def test_reference_parameter_and_flop_counts(self):
        self.assertEqual(tuple(preset_names()), NAMES)
        for name in NAMES:
            with self.subTest(name=name):
                p = load_preset(name); ref = p["reference"]
                cfg = model_config(p["model"], ref["vocab_size"], 8192).to_dict()
                actual = count(cfg, 8192)
                # Archived experiments retain the old convention; never rewrite their history.
                self.assertEqual(legacy_train_flops_per_token(cfg,8192), ref["flops_per_token"])
                self.assertLess(actual["train_flops_per_token"], ref["flops_per_token"])
                self.assertEqual(actual["total_nonvocab_params"] + actual["embed_params"], ref["n_params"])
                self.assertEqual(actual["active_nonvocab_params"], ref["n_nonvocab_active"])
                self.assertEqual(cfg["layer_types"], default_layer_types(p["model"]["layers"]))
                self.assertTrue(cfg["tie_word_embeddings"])
                # Historical runs rounded the step count; budgets are now floored to stay within compute.
                tokens, _ = preset_budget(name, ref["vocab_size"])
                self.assertEqual(round(p["compute"] / ref["flops_per_token"] / 524288) * 524288, ref["tokens"])
                flops=actual["train_flops_per_token"]
                self.assertLessEqual(tokens*flops,p["compute"])
                self.assertGreater((tokens+524288)*flops,p["compute"])

    def test_budget_for_tokenizer_and_seq(self):
        vocab = Tokenizer.from_file(str(DEFAULT_TOKENIZER / "tokenizer.json")).get_vocab_size()
        self.assertEqual(vocab, 50048)
        for name in NAMES:
            for seq in (8192, 4096):
                with self.subTest(name=name, seq=seq):
                    tokens, fc = preset_budget(name, vocab, seq)
                    flops, budget = fc["train_flops_per_token"], load_preset(name)["compute"]
                    self.assertEqual(tokens % 524288, 0)
                    self.assertLessEqual(tokens * flops, budget)
                    self.assertGreater((tokens + 524288) * flops, budget)

    def test_cli_overrides_preset(self):
        a = parse_args(["--preset", "q35_c1_s1", "--lr", "1e-3", "--tokens", "3e6"])
        self.assertEqual((a.layers, a.hidden, a.micro_bs, a.experts_impl), (5, 96, 16, "grouped_mm"))
        self.assertEqual((a.lr, a.tokens, a.compute), (1e-3, 3_000_000, 3e17))
        self.assertEqual(a.run_dir, str(ROOT / "runs/q35_c1_s1"))
        with self.assertRaises(SystemExit):
            parse_args(["--layers", "4"])   # no preset: run_dir and budget required


if __name__ == "__main__":
    unittest.main()
