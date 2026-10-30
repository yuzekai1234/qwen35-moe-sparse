"""Numerical boundaries, actual FP8 forward and optimizer-compatible backward."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM
from q38.presets import model_config
from q38.train import NEW_RESUME_DEFAULTS, RESUME_KEYS, parse_args


class PrecisionArguments(unittest.TestCase):
    def test_full_cache_validation_rejects_corruption_and_wrong_identity(self):
        import numpy as np
        from q38.evaluate_fused import validate_packed_cache
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            np.array([0, 1, 2], dtype=np.int32).tofile(cache/'tokens.bin')
            np.array([False, True, True], dtype=np.bool_).tofile(cache/'is_text.bin')
            metadata = dict(dataset_sha256='data', tokenizer_sha256='tokenizer', pad_token_id=0,
                            total_tokens=2, total_bytes=3)
            for name, key in [('tokens.bin', 'tokens_sha256'), ('is_text.bin', 'is_text_sha256')]:
                metadata[key] = hashlib.sha256((cache/name).read_bytes()).hexdigest()
            (cache/'metadata.json').write_text(json.dumps(metadata))
            self.assertEqual(validate_packed_cache(cache, 'data', 'tokenizer', 0)['total_tokens'], 2)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                validate_packed_cache(cache, 'different', 'tokenizer', 0)
            (cache/'tokens.bin').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                validate_packed_cache(cache, 'data', 'tokenizer', 0)

    def test_resume_tracks_precision_and_calibration_contents(self):
        self.assertIn('ffn_precision', RESUME_KEYS)
        self.assertIn('ffn_calibration_sha256', RESUME_KEYS)
        self.assertEqual(NEW_RESUME_DEFAULTS['ffn_precision'], 'bf16')
        args = parse_args(['--preset', 'baseline', '--dry-run'])
        self.assertEqual(args.ffn_precision, 'bf16')


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9),
                     'SM89+ CUDA GPU required')
class FusedPrecisionGPU(unittest.TestCase):
    def test_fused_trunk_causality_and_short_ngram_tail(self):
        from q38.fused_precision import install_fused_ffn, install_fused_trunk
        from q38.inference import optimize_for_inference
        shape = dict(hidden=64, layers=4, heads=2, kv_heads=1, head_dim=64,
            lin_k_heads=2, lin_v_heads=4, lin_head_dim=64, experts=0, topk=0,
            moe_ffn=0, shared_ffn=0, aux_coef=0., dense_ffn=128,
            ngram_rows=128, mhc_enabled=0, qsa_mode='full', mtp=0)
        torch.manual_seed(9)
        model = Qwen3_8MoeForCausalLM(model_config(shape, 128, 128, eos_token_id=0)).cuda().bfloat16().eval()
        optimize_for_inference(model)
        fused = copy.deepcopy(model)
        install_fused_ffn(fused, 'bf16_fused')
        install_fused_trunk(fused)
        with torch.inference_mode():
            for length in (1, 65, 128):
                ids = torch.randint(1, 128, (1, length), device='cuda')
                before, after = model(input_ids=ids).logits, fused(input_ids=ids).logits
                torch.testing.assert_close(after, before, atol=.008, rtol=.03)
                if length > 1:
                    ids[:, length//2:] = (ids[:, length//2:]+13) % 127+1
                    changed = fused(input_ids=ids).logits
                    torch.testing.assert_close(after[:, :length//2], changed[:, :length//2], atol=0, rtol=0)

    def test_fused_add_norm_and_swiglu_preserve_rounding_boundaries(self):
        from q38.fused_precision import add_norm, quantize_static, swiglu
        from q38.models.qwen3_8_moe.modeling_qwen3_8_moe import Qwen3_8MoeRMSNorm
        torch.manual_seed(7)
        norm = Qwen3_8MoeRMSNorm(512).cuda().to(torch.bfloat16)
        for rows in (1, 65, 1024):
            x = torch.randn(rows, 512, device='cuda', dtype=torch.float32)
            y = torch.randn_like(x, dtype=torch.bfloat16)
            with torch.inference_mode():
                r, actual = add_norm(x, y, norm)
                torch.testing.assert_close(r, x+y.float(), atol=0, rtol=0)
                expected = norm(r).bfloat16()
                torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.008)
                gu = torch.randn(rows, 256, device='cuda', dtype=torch.bfloat16)
                gate, up = gu.chunk(2, -1)
                reference = F.silu(gate)*up
                torch.testing.assert_close(swiglu(gu), reference, atol=0.015625, rtol=0.008)
                scale = torch.tensor(.025, device='cuda')
                q = quantize_static(reference, scale)
                torch.testing.assert_close(swiglu(gu, scale).float(), q.float(), atol=0.5, rtol=0.13)

    def test_real_fp8_forward_bf16_backward_and_immutable_inference_cache(self):
        from q38.fused_precision import install_fused_ffn, install_fp8_training
        shape = dict(hidden=64, layers=1, heads=2, kv_heads=1, head_dim=64,
            lin_k_heads=2, lin_v_heads=4, lin_head_dim=64, experts=0, topk=0,
            moe_ffn=0, shared_ffn=0, aux_coef=0.,
            dense_ffn=128, ngram_rows=0, mhc_enabled=0, qsa_mode='full', mtp=0)
        cfg = model_config(shape, 128, 128)
        model = Qwen3_8MoeForCausalLM(cfg).cuda().float()
        calibration = dict(layers={'0': {'x': .025, 'h': .005}})
        original = copy.deepcopy(model).bfloat16().eval()
        reference = copy.deepcopy(original)
        install_fp8_training(model, calibration)
        install_fused_ffn(original, 'fp8_fused', calibration)
        for rows in (1, 65, 128):
            x = torch.randn(rows, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            actual = model.model.layers[0].mlp(x)
            with torch.inference_mode():
                expected = original.model.layers[0].mlp(x)
            torch.testing.assert_close(actual, expected, rtol=.03, atol=.002)
            actual.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        mlp = model.model.layers[0].mlp
        before = mlp.gate_proj.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, mlp.gate_proj.weight))
        for name, p in mlp.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertEqual(set(reference.state_dict()), set(model.state_dict()))
        with self.assertRaisesRegex(RuntimeError, 'inference_mode'):
            original.model.layers[0].mlp(x)

    def test_projection_training_matches_native_deployment_and_has_gradients(self):
        from q38.fused_precision import (calibrate_ffn, install_fp8_training,
            install_fused_ffn, install_fused_trunk, install_fp8_projections)
        from q38.inference import optimize_for_inference
        shape = dict(hidden=64, layers=4, heads=2, kv_heads=1, head_dim=64,
            lin_k_heads=2, lin_v_heads=4, lin_head_dim=64, experts=0, topk=0,
            moe_ffn=0, shared_ffn=0, aux_coef=0., dense_ffn=128,
            ngram_rows=0, mhc_enabled=0, qsa_mode='full', mtp=0)
        torch.manual_seed(12)
        model = Qwen3_8MoeForCausalLM(model_config(shape, 128, 128)).cuda().bfloat16().eval()
        optimize_for_inference(model)
        ids = torch.randint(1, 128, (1, 65), device='cuda')
        calibration = calibrate_ffn(model, [ids])
        deployed = copy.deepcopy(model)
        install_fp8_training(model, calibration, projections=True)
        install_fused_ffn(deployed, 'fp8_fused', calibration)
        install_fp8_projections(deployed, calibration)
        install_fused_trunk(deployed)
        before = model(input_ids=ids).logits
        before.float().square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        with torch.inference_mode():
            after = deployed(input_ids=ids).logits
        torch.testing.assert_close(after, before, atol=.008, rtol=.03)
        with torch.inference_mode():
            short_reference = model(input_ids=ids[:, :1]).logits
            short_deployed = deployed(input_ids=ids[:, :1]).logits
        torch.testing.assert_close(short_deployed, short_reference, atol=.008, rtol=.03)


if __name__ == '__main__':
    unittest.main()
