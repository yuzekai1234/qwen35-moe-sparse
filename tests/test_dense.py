"""FFN-only dense structure, parameter accounting, gradients and portable export."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from q38.flops import count, validate_resume_accounting
from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM
from q38.models.qwen3_8_moe import modeling_qwen3_8_moe as M
from q38.muon import HybridOptimizer
from q38.presets import ROOT, dense_shape, model_config
from q38.train import LossModel, parse_args, prepare_inputs
from scripts.make_dense_configs import make_config


class CE(torch.nn.Module):
    def forward(self, weight, hidden, target):
        return F.cross_entropy(F.linear(hidden, weight), target)


def tiny_shape():
    return dict(layers=4, hidden=32, heads=2, kv_heads=1, lin_k_heads=1,
                lin_v_heads=2, experts=8, topk=2, moe_ffn=8, shared_ffn=8,
                ngram_rows=64, qsa_blocks=1)


class Dense(unittest.TestCase):
    def test_only_ffn_dimensions_change_and_ngram_auto_is_materialized(self):
        shape = dict(tiny_shape(), ngram_rows=-1)
        reference = model_config(shape, 128, 128).to_dict()
        dense = model_config(dense_shape(shape, 128, 128), 128, 128).to_dict()
        allowed = {"num_experts", "num_experts_per_tok", "moe_intermediate_size",
                   "shared_expert_intermediate_size", "intermediate_size", "router_aux_loss_coef"}
        for key in set(reference) | set(dense):
            if key not in allowed and not key.startswith("_experts"):
                self.assertEqual(reference.get(key), dense.get(key), key)
        self.assertEqual(dense["ngram_rows"], reference["ngram_rows"])
        self.assertEqual(dense["intermediate_size"], 2*8+8)

    def test_rejects_partial_or_ambiguous_dense_configs(self):
        with self.assertRaises(ValueError):
            model_config(dict(tiny_shape(), experts=0), 128, 128)
        dense = dense_shape(tiny_shape(), 128, 128)
        for field, value in [("ngram_rows", -1), ("topk", 1), ("dense_ffn", 0), ("aux_coef", .001)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                model_config(dict(dense, **{field: value}), 128, 128)

    def test_actual_parameter_counts_and_router_differences(self):
        for mtp in (False, True):
            shape = dict(tiny_shape(), mtp=int(mtp))
            before_cfg = model_config(shape, 128, 128)
            cfg = model_config(dense_shape(shape, 128, 128), 128, 128)
            model = Qwen3_8MoeForCausalLM(cfg)
            old, new = count(before_cfg.to_dict(), 128), count(cfg.to_dict(), 128)
            self.assertEqual(sum(p.numel() for p in model.parameters()), new["total_params_all"])
            embeddings = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Embedding))
            self.assertEqual(new["total_params_excluding_embeddings"], new["total_params_all"]-embeddings)
            per_block = 32*(8+1)
            self.assertEqual(old["active_params_excluding_embeddings"]-new["active_params_excluding_embeddings"], 4*per_block)
            self.assertEqual(old["train_flops_per_token"]-new["train_flops_per_token"], 6*per_block*(4+int(mtp)))
            self.assertEqual(new["breakdown"]["router"], 0)
            self.assertEqual(new["breakdown"]["shared_gate"], 0)
            self.assertFalse(any(isinstance(m, (M.Qwen3_8MoeSparseMoeBlock, M.Qwen3_8MoeExperts,
                                               M.Qwen3_8MoeTopKRouter)) for m in model.modules()))
            if mtp:
                self.assertIsInstance(model.mtp["block"].mlp, M.Qwen3_8MoeMLP)
            with self.assertRaises(ValueError):
                validate_resume_accounting(dict(flops_per_token=old["train_flops_per_token"],
                    flops_accounting=old["accounting_method"]), new)

    def test_dense_mtp_backward_optimizer_and_hf_roundtrip(self):
        torch.manual_seed(0)
        cfg = model_config(dense_shape(tiny_shape(), 128, 128), 128, 128, eos_token_id=0)
        model = Qwen3_8MoeForCausalLM(cfg)
        wrapper = LossModel(model, 0, 0, ce=CE())
        self.assertEqual(wrapper._router_hooks, [])
        x = torch.randint(1, 128, (1, 128))
        ce, aux, mtp, kl = wrapper(x, x.roll(-1, 1))
        self.assertEqual(aux.item(), 0)
        loss = ce+.3*mtp+.05*kl
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for layer in [*model.model.layers, model.mtp["block"]]:
            for proj in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                self.assertIsNotNone(proj.weight.grad)
                self.assertTrue(torch.isfinite(proj.weight.grad).all())
        optimizer = HybridOptimizer(model, lr=.001, weight_decay=.1)
        muon_ids = {id(p) for group in optimizer.muon.param_groups for p in group["params"]}
        for layer in [*model.model.layers, model.mtp["block"]]:
            for proj in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                self.assertIn(id(proj.weight), muon_ids)
        optimizer.step(); optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            expected = model(input_ids=x, use_cache=False, output_router_logits=True).logits
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded = AutoModelForCausalLM.from_pretrained(tmp, trust_remote_code=True, local_files_only=True).eval()
            with torch.no_grad():
                actual = loaded(input_ids=x, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(loaded.config.num_experts, 0)
            self.assertEqual(loaded.config.intermediate_size, 24)

    def test_four_frozen_presets_and_whole_batch_budgets(self):
        expected_widths = (672, 864, 1152, 1536)
        expected_active = (26855906, 51837772, 128657274, 298910218)
        for tier, width, active in zip(range(1,5), expected_widths, expected_active):
            reference = json.loads((ROOT/f"configs/q38_c{tier}.json").read_text())
            p = make_config(reference, Path(f"/unused/dense/c{tier}"))
            self.assertEqual(p["model"]["dense_ffn"], width)
            self.assertEqual(p["accounting"]["active_params_excluding_embeddings"], active)
            self.assertEqual(p["model"]["ngram_rows"], reference["model"]["ngram_rows"])
            batch, budget = p["train"]["batch_tokens"], p["compute"]*(1-p["accounting"]["compute_reserve_fraction"])
            tokens, fpt = p["train"]["tokens"], p["accounting"]["train_flops_per_token"]
            self.assertLessEqual(tokens*fpt, budget)
            self.assertGreater((tokens+batch)*fpt, budget)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"config.json"; path.write_text(json.dumps(p))
                args = parse_args(["--config", str(path), "--dry-run"])
                self.assertEqual(args.dense_ffn, width)

    def test_cli_cannot_keep_large_token_budget_after_lowering_compute(self):
        reference = json.loads((ROOT/"configs/q38_c1.json").read_text())
        p = make_config(reference, Path("/unused/dense/c1"))
        class Tokenizer:
            eos_token_id, bos_token_id, pad_token_id = 0, 1, 0
            def __len__(self): return 10112
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"config.json"; path.write_text(json.dumps(p))
            args = parse_args(["--config", str(path), "--compute", "3e15", "--dry-run"])
            with patch("q38.train.load_tokenizer", return_value=Tokenizer()), \
                    patch("q38.train.tokenizer_sha256", return_value="not-read"):
                with self.assertRaisesRegex(ValueError, "token budget exceeds compute"):
                    prepare_inputs(args)


if __name__ == "__main__":
    unittest.main()
