"""Check that vendoring preserves the baseline and exports the editable implementation."""
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import PreTrainedTokenizerFast
from transformers import Qwen3_5MoeTextConfig as StockConfig
from transformers import Qwen3_5MoeForCausalLM as StockModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.qwen3_5_moe import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM


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
    def test_gzy_vocab_forward_backward_and_export(self):
        tok = PreTrainedTokenizerFast.from_pretrained(ROOT / 'tokenizer/gzy_10k')
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
        self.assertEqual(model.lm_head.out_features, 10112)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tok.save_pretrained(tmp)
            loaded = AutoModelForCausalLM.from_pretrained(tmp, trust_remote_code=True,
                                                        local_files_only=True)
            restored = PreTrainedTokenizerFast.from_pretrained(tmp, local_files_only=True)
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
        self.assertTrue(Path(inspect.getfile(type(local))).is_relative_to(ROOT / "models"))
        self.assertEqual(list(stock.state_dict()), list(local.state_dict()))
        self.assertEqual(local.lm_head.weight.data_ptr(), local.model.embed_tokens.weight.data_ptr())
        with tempfile.TemporaryDirectory() as tmp:
            local.save_pretrained(tmp)
            path = Path(tmp)
            self.assertEqual((path / "modeling_qwen3_5_moe.py").read_bytes(),
                             (ROOT / "models/qwen3_5_moe/modeling_qwen3_5_moe.py").read_bytes())
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
