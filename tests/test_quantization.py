"""Quantization numerics, scope isolation, native CUDA and BPB accounting."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from q38.compare_precision import compare_results, parse_args
from q38.evaluate import pack_jsonl
from q38.evaluate_single_gpu import evaluate_model
from q38.quantization import LowPrecisionLinear, apply_precision, quantize_tensor


class Quantization(unittest.TestCase):
    def test_zero_and_extreme_values_are_finite(self):
        for precision in ("int8", "fp8"):
            for values in (torch.zeros(2, 32), torch.tensor([[1e-20, -1e-20], [1e20, -1e20]])):
                quantized, scale = quantize_tensor(values, precision, dim=1)
                restored = quantized.float()*scale
                self.assertTrue(torch.isfinite(restored).all())
                torch.testing.assert_close(restored, values, rtol=.01, atol=1e-25)

    def test_reference_linear_preserves_shape_bias_and_bounded_error(self):
        torch.manual_seed(11)
        source = nn.Linear(64, 48, bias=True).eval()
        inputs = torch.randn(2, 17, 64)
        expected = source(inputs)
        for precision, tolerance in (("int8", .02), ("fp8", .08)):
            layer = LowPrecisionLinear(source, precision, "reference")
            actual = layer(inputs)
            self.assertEqual(actual.shape, expected.shape)
            self.assertLess((actual-expected).norm()/expected.norm(), tolerance)
            self.assertFalse(actual.requires_grad)
            torch.testing.assert_close(layer(torch.zeros_like(inputs)), source.bias.expand_as(expected))
            self.assertEqual(layer(inputs[:, :0]).shape, (2, 0, 48))
            with self.assertRaises(ValueError):
                layer(torch.zeros(2, 128))
            layer.train()
            with self.assertRaisesRegex(RuntimeError, "inference only"):
                layer(inputs)

    def test_scope_excludes_attention_mhc_head_embeddings_and_mtp(self):
        from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM
        from q38.presets import dense_shape, model_config
        shape = dict(layers=4, hidden=32, heads=2, kv_heads=1, lin_k_heads=1,
                     lin_v_heads=2, experts=8, topk=2, moe_ffn=16, shared_ffn=16,
                     ngram_rows=64, qsa_blocks=1)
        config = model_config(dense_shape(shape, 128, 128), 128, 128, eos_token_id=0)
        torch.manual_seed(3)
        source = Qwen3_8MoeForCausalLM(config).eval()
        original = {name: value.clone() for name, value in source.state_dict().items()}
        ids = torch.randint(1, 128, (1, 64))
        with torch.inference_mode():
            baseline = source(ids, use_cache=False).logits
        bf16_copy = copy.deepcopy(source)
        report = apply_precision(bf16_copy, "bf16")
        self.assertEqual(report["quantized_modules"], [])
        with torch.inference_mode():
            torch.testing.assert_close(bf16_copy(ids, use_cache=False).logits, baseline, rtol=0, atol=0)
        for precision in ("int8", "fp8"):
            model = copy.deepcopy(source)
            report = apply_precision(model, precision, "reference")
            self.assertEqual(len(report["quantized_modules"]), 12)
            self.assertEqual(report["quantized_weight_elements"], 4*3*32*48)
            self.assertEqual(model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr())
            for name, value in model.state_dict().items():
                if name in original:
                    torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            with torch.inference_mode():
                logits = model(ids, use_cache=False).logits
            self.assertTrue(torch.isfinite(logits).all())
            self.assertFalse(torch.equal(logits, baseline))
            with self.assertRaisesRegex(ValueError, "Reload"):
                apply_precision(model, precision, "reference")
        source.config.num_experts = 8
        with self.assertRaisesRegex(ValueError, "dense checkpoint"):
            apply_precision(source, "int8", "reference")

    def test_bpb_only_does_not_time_or_warm_and_scores_same_targets(self):
        class Toy:
            calls = 0
            def __call__(self, input_ids, **kwargs):
                self.calls += 1
                return SimpleNamespace(logits=F.one_hot(input_ids % 6, 6).float())
        tokens = np.array([0, 1, 2, 0, 3, 4, 5], dtype=np.int32)
        mask = tokens != 0
        meta = dict(total_tokens=5, total_bytes=13, pad_token_id=0)
        model = Toy()
        with patch("torch.cuda.synchronize") as sync, patch("q38.evaluate_single_gpu.time.perf_counter") as clock:
            result = evaluate_model(model, tokens, mask, meta, 1, 4, "cpu", 100,
                                    measure_time=False, warmup_steps=0)
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["scored_tokens"], 5)
        self.assertIsNone(result["evaltime"])
        self.assertIsNone(result["forward_seconds"])
        self.assertFalse(result["timing_enabled"])
        sync.assert_not_called()
        clock.assert_not_called()

    def test_prefix_packing_counts_only_selected_utf8_bytes(self):
        class Tokenizer:
            eos_token_id, pad_token_id = 0, 0
            def __call__(self, texts, **kwargs):
                return dict(input_ids=[[1]*len(text) for text in texts])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/"data.jsonl"
            data.write_text('\n'+json.dumps(dict(text="中文"))+'\n\n'+json.dumps(dict(text="skip"))+'\n')
            metadata = pack_jsonl(data, Tokenizer(), root, max_documents=1)
            self.assertEqual(metadata["total_bytes"], 6)
            self.assertEqual(metadata["total_tokens"], 2)
            self.assertEqual(np.fromfile(root/"tokens.bin", dtype=np.int32).tolist(), [0, 1, 1])
            with self.assertRaises(ValueError):
                pack_jsonl(data, Tokenizer(), root, max_documents=0)

    def test_comparison_requires_local_baseline_and_cli_rejects_bad_runs(self):
        rows = [dict(bpb=1.1, quantization=dict(precision="int8"))]
        self.assertIsNone(compare_results(rows)[0]["delta_bpb_vs_bf16"])
        rows.append(dict(bpb=1.0, quantization=dict(precision="bf16")))
        self.assertAlmostEqual(compare_results(rows)[0]["delta_bpb_vs_bf16"], .1)
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--model-path", "/unused", "--dataset-path", "/unused", "--output-dir", tmp+"/new"]
            self.assertEqual(parse_args(args).precisions, ["bf16", "int8", "fp8"])
            with patch("sys.stderr"):
                for extra in (["--max-documents", "0"], ["--batch-size", "0"],
                              ["--precisions", "fp8", "fp8"], ["--output-dir", tmp]):
                    with self.assertRaises(SystemExit):
                        parse_args(args+extra)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for real low-precision GEMM")
    def test_native_matches_qdq_reference_including_tail_rows(self):
        torch.manual_seed(9)
        source = nn.Linear(64, 48, bias=False, device="cuda", dtype=torch.bfloat16).eval()
        for precision in ("int8", "fp8"):
            if precision == "fp8" and torch.cuda.get_device_capability() < (8, 9):
                continue
            native = LowPrecisionLinear(source, precision, "native")
            reference = LowPrecisionLinear(source, precision, "reference")
            for rows in (1, 17, 32):
                inputs = torch.randn(2, rows, 64, device="cuda", dtype=torch.bfloat16)
                torch.testing.assert_close(native(inputs), reference(inputs), rtol=.02, atol=.01)
            zeros = torch.zeros(1, 1, 64, device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(native(zeros), torch.zeros(1, 1, 48, device="cuda", dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
