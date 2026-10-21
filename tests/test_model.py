"""The vendored text-only model matches stock transformers and exports its own source."""
import inspect
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import Qwen3_5MoeTextConfig as StockConfig
from transformers import Qwen3_5MoeForCausalLM as StockModel

from q35.data import DEFAULT_TOKENIZER, ROOT, load_tokenizer
from q35.inference import compile_inference_norms, optimize_for_inference, triton_experts_forward
from q35.models.qwen3_5_moe import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM


def config(cls):
    cfg = cls(vocab_size=256, hidden_size=96, num_hidden_layers=4,
              num_attention_heads=2, num_key_value_heads=1, head_dim=64,
              linear_num_key_heads=1, linear_num_value_heads=2,
              linear_key_head_dim=64, linear_value_head_dim=64,
              num_experts=4, num_experts_per_tok=2, moe_intermediate_size=96,
              shared_expert_intermediate_size=48, tie_word_embeddings=True,
              max_position_embeddings=256, use_cache=False,
              rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                               "partial_rotary_factor": 0.25})
    cfg._experts_implementation = "eager"
    cfg._attn_implementation = "sdpa"
    return cfg


class LocalModel(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "GPU required for compiled normalization")
    def test_compiled_norms_match_eager(self):
        from q35.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm, Qwen3_5MoeRMSNormGated

        torch.manual_seed(0)
        for cls, dim in ((Qwen3_5MoeRMSNorm, 192), (Qwen3_5MoeRMSNormGated, 64)):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(module=cls.__name__, dtype=dtype):
                    eager = cls(dim).cuda().to(dtype).eval()
                    with torch.no_grad():
                        eager.weight.normal_(mean=1 if cls is Qwen3_5MoeRMSNormGated else 0, std=0.1)
                    compiled = copy.deepcopy(eager)
                    self.assertEqual(compile_inference_norms(compiled), 1)
                    # Noncontiguous rows also occur in attention normalization.
                    x = torch.randn(37, 2, dim, device="cuda", dtype=dtype)[:, 0]
                    args = (x, torch.randn_like(x)) if cls is Qwen3_5MoeRMSNormGated else (x,)
                    with torch.inference_mode():
                        expected, actual = eager(*args), compiled(*args)
                    tolerance = 1e-2 if dtype is torch.bfloat16 else 1e-5
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    self.assertEqual(actual.dtype, dtype)
                    for name, value in eager.state_dict().items():
                        self.assertTrue(torch.equal(value, compiled.state_dict()[name]))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required for Triton experts")
    def test_triton_experts_match_grouped_mm(self):
        from transformers.integrations.moe import grouped_mm_experts_forward
        from q35.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

        # Preset shape, then a non-power-of-two expert count with intermediate != hidden.
        for experts, top_k, hidden, inter in ((64, 4, 192, 192), (6, 2, 96, 160)):
            cfg = config(Qwen3_5MoeTextConfig)
            cfg.num_experts, cfg.num_experts_per_tok = experts, top_k
            cfg.hidden_size, cfg.moe_intermediate_size = hidden, inter
            torch.manual_seed(0)
            block = Qwen3_5MoeSparseMoeBlock(cfg).cuda()
            with torch.no_grad():
                for param in block.parameters():
                    param.normal_(std=0.05)
            for dtype in (torch.bfloat16, torch.float32):
                block.to(dtype)
                for tokens in (1, 37, 5000):
                    with self.subTest(experts=experts, dtype=dtype, tokens=tokens), torch.inference_mode():
                        x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
                        _, weights, index = block.gate(x)
                        expected = grouped_mm_experts_forward(block.experts, x, index, weights)
                        actual = triton_experts_forward(block.experts, x, index, weights)
                        tolerance = 1e-2 if dtype is torch.bfloat16 else 1e-5
                        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                        self.assertEqual(actual.dtype, dtype)
        x = torch.randn(4, hidden, device="cuda", dtype=dtype, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "no backward"):
            triton_experts_forward(block.experts, x, index, weights)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required for inference fast paths")
    def test_inference_fast_paths_match_reference(self):
        torch.manual_seed(0)
        exact = Qwen3_5MoeForCausalLM(config(Qwen3_5MoeTextConfig)).cuda().eval()
        reference = copy.deepcopy(exact).to(torch.bfloat16)
        fast = copy.deepcopy(reference)
        installed = optimize_for_inference(fast)
        self.assertEqual(installed, dict(experts="q35_triton", gated_deltanet=3, norms=14))

        def unexpected(*args, **kwargs):
            raise AssertionError("uncached GatedDeltaNet fell back to the reference forward")
        for layer in fast.model.layers:
            if layer.block_type == "linear_attention":
                layer.linear_attn._reference_forward = unexpected
        ids = torch.randint(0, 256, (2, 200), device="cuda")
        with torch.inference_mode():
            truth = exact(ids, use_cache=False).logits
            errors = [(model(ids, use_cache=False).logits.float() - truth).abs() for model in (reference, fast)]
        # Against FP32, the fast paths must stay within the BF16 reference path's own rounding error.
        self.assertLess(errors[1].mean().item(), 1.5 * errors[0].mean().item())
        self.assertLess(errors[1].max().item(), 1.5 * errors[0].max().item())

    def test_gzy_vocab_forward_backward_and_export(self):
        tok = load_tokenizer(DEFAULT_TOKENIZER)
        cfg = config(Qwen3_5MoeTextConfig)
        cfg.vocab_size = len(tok)
        cfg.num_hidden_layers = 1
        cfg.layer_types = ['full_attention']
        cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id = (
            tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
        model = Qwen3_5MoeForCausalLM(cfg)
        text = 'Small vocabulary: 中文训练数据。'
        ids = tok(text, add_special_tokens=False, return_tensors='pt').input_ids
        loss = model(ids, labels=ids, use_cache=False).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())
        self.assertEqual(model.lm_head.out_features, 50048)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tok.save_pretrained(tmp)
            loaded = AutoModelForCausalLM.from_pretrained(tmp, trust_remote_code=True,
                                                        local_files_only=True)
            restored = load_tokenizer(tmp)
            self.assertEqual(restored.encode(text), tok.encode(text))
            self.assertEqual(restored.decode(restored.encode(text)), text)
            self.assertEqual((loaded.config.bos_token_id, loaded.config.eos_token_id,
                              loaded.config.pad_token_id), (1, 53, 0))
            self.assertEqual(loaded.lm_head.weight.data_ptr(), loaded.model.embed_tokens.weight.data_ptr())

    def test_baseline_state_and_exported_source(self):
        torch.manual_seed(0)
        stock = StockModel(config(StockConfig))
        torch.manual_seed(0)
        local = Qwen3_5MoeForCausalLM(config(Qwen3_5MoeTextConfig))
        for key, value in stock.state_dict().items():
            self.assertTrue(torch.equal(value, local.state_dict()[key]), f"seeded init: {key}")
        local.load_state_dict(stock.state_dict(), strict=True)
        self.assertTrue(Path(inspect.getfile(type(local))).is_relative_to(ROOT / "q35/models"))
        self.assertEqual(list(stock.state_dict()), list(local.state_dict()))
        self.assertEqual(local.lm_head.weight.data_ptr(), local.model.embed_tokens.weight.data_ptr())
        with tempfile.TemporaryDirectory() as tmp:
            local.save_pretrained(tmp)
            path = Path(tmp)
            self.assertEqual((path / "modeling_qwen3_5_moe.py").read_bytes(),
                             (ROOT / "q35/models/qwen3_5_moe/modeling_qwen3_5_moe.py").read_bytes())
            self.assertTrue((path / "configuration_qwen3_5_moe.py").exists())
            saved = json.loads((path / "config.json").read_text())
            self.assertEqual(saved["model_type"], "qwen35_moe_sparse")
            self.assertIn("AutoModelForCausalLM", saved["auto_map"])
            with self.assertRaises(ValueError):
                AutoConfig.from_pretrained(tmp, trust_remote_code=False)
            restored = AutoModelForCausalLM.from_pretrained(tmp, trust_remote_code=True,
                                                         local_files_only=True)
            self.assertIn("transformers_modules.", type(restored).__module__)
            for key, value in local.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required for FLA")
    def test_stock_and_local_forward(self):
        torch.manual_seed(0)
        stock = StockModel(config(StockConfig)).cuda().eval()
        local = Qwen3_5MoeForCausalLM(config(Qwen3_5MoeTextConfig)).cuda().eval()
        local.load_state_dict(stock.state_dict(), strict=True)
        ids = torch.arange(64, device="cuda").view(1, -1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = stock(ids, use_cache=False).logits
            actual = local(ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
