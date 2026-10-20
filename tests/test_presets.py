"""Presets reproduce the historical runs, and budgets are recalculated for the real vocabulary."""
import unittest

from tokenizers import Tokenizer

from q35.data import DEFAULT_TOKENIZER
from q35.evaluate import active_parameter_count
from q35.flops import count
from q35.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
from q35.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP
from q35.presets import ROOT, default_layer_types, load_preset, model_config, preset_budget, preset_names
from q35.train import parse_args

NAMES = ("q35_c1_dense", "q35_c1_s1", "q35_c1_s2", "q35_c1_s3", "q35_c2_b", "q35_c2_m")
HISTORICAL = tuple(n for n in NAMES if "reference" in load_preset(n))


class Presets(unittest.TestCase):
    def test_reference_parameter_and_flop_counts(self):
        self.assertEqual(tuple(preset_names()), NAMES)
        for name in HISTORICAL:
            with self.subTest(name=name):
                p = load_preset(name); ref = p["reference"]
                cfg = model_config(p["model"], ref["vocab_size"], 8192).to_dict()
                actual = count(cfg, 8192)
                self.assertEqual(actual["train_flops_per_token"], ref["flops_per_token"])
                self.assertEqual(actual["total_nonvocab_params"] + actual["embed_params"], ref["n_params"])
                self.assertEqual(actual["active_nonvocab_params"], ref["n_nonvocab_active"])
                self.assertEqual(cfg["layer_types"], default_layer_types(p["model"]["layers"]))
                self.assertTrue(cfg["tie_word_embeddings"])
                # Historical runs rounded the step count; budgets are now floored to stay within compute.
                tokens, _ = preset_budget(name, ref["vocab_size"])
                self.assertEqual(round(p["compute"] / ref["flops_per_token"] / 524288) * 524288, ref["tokens"])
                self.assertIn(ref["tokens"] - tokens, (0, 524288))

    def test_budget_for_tokenizer_and_seq(self):
        vocab = Tokenizer.from_file(str(DEFAULT_TOKENIZER / "tokenizer.json")).get_vocab_size()
        self.assertEqual(vocab, 30080)
        for name in NAMES:
            for seq in (8192, 4096):
                with self.subTest(name=name, seq=seq):
                    tokens, fc = preset_budget(name, vocab, seq)
                    flops, budget = fc["train_flops_per_token"], load_preset(name)["compute"]
                    self.assertEqual(tokens % 524288, 0)
                    self.assertLessEqual(tokens * flops, budget)
                    self.assertGreater((tokens + 524288) * flops, budget)

    def test_dense_ablation(self):
        p = load_preset("q35_c1_dense"); base = load_preset(p["ablation"]["base"])
        cfg = model_config(p["model"], 30080, 8192)
        fc, fb = count(cfg.to_dict(), 8192), count(model_config(base["model"], 30080, 8192).to_dict(), 8192)
        self.assertEqual(p["compute"], base["compute"])
        self.assertLess(abs(fc["train_flops_per_token"] / fb["train_flops_per_token"] - 1), 0.01)
        model = Qwen3_5MoeForCausalLM(cfg)
        self.assertTrue(all(type(layer.mlp) is Qwen3_5MoeMLP for layer in model.model.layers))
        n_nv = sum(p.numel() for p in model.parameters()) - model.get_input_embeddings().weight.numel()
        self.assertEqual(fc["active_nonvocab_params"], fc["total_nonvocab_params"])
        self.assertEqual(active_parameter_count(model), n_nv)
        self.assertEqual(fc["total_nonvocab_params"], n_nv)

    def test_cli_overrides_preset(self):
        a = parse_args(["--preset", "q35_c1_s1", "--lr", "1e-3", "--tokens", "3e6"])
        self.assertEqual((a.layers, a.hidden, a.micro_bs, a.experts_impl), (5, 96, 16, "grouped_mm"))
        self.assertEqual((a.lr, a.tokens, a.compute), (1e-3, 3_000_000, 3e17))
        self.assertEqual(a.run_dir, str(ROOT / "runs/q35_c1_s1"))
        with self.assertRaises(SystemExit):
            parse_args(["--layers", "4"])   # no preset: run_dir and budget required


if __name__ == "__main__":
    unittest.main()
