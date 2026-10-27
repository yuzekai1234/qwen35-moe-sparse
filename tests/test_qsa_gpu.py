"""CUDA regression for short GQA tails and nontrivial selected-block attention."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM
from q38.models.qwen3_8_moe import modeling_qwen3_8_moe as M
from q38.presets import dense_shape, model_config


@unittest.skipUnless(torch.cuda.is_available(), "QSA compiled-kernel regression requires CUDA")
class QSAGPU(unittest.TestCase):
    def test_short_gqa_tail_and_selected_blocks_match_explicit_mask(self):
        torch.manual_seed(0)
        torch._dynamo.config.suppress_errors = False
        # The C1 query/KV head ratio, but a tiny hidden width to isolate attention.
        shape = dict(layers=4, hidden=32, heads=16, kv_heads=2, lin_k_heads=1,
                     lin_v_heads=2, experts=8, topk=2, moe_ffn=8, shared_ffn=8,
                     ngram_rows=0, qsa_blocks=2)
        cfg = model_config(dense_shape(shape, 128, 1024), 128, 1024)
        model = Qwen3_8MoeForCausalLM(cfg).cuda().eval()
        attn, indexer = model.model.layers[3].self_attn, model.model.indexers["3"]
        reference_masks = []

        def explicit_attention(q, k, v, *, scale, block_mask, enable_gqa, kernel_options):
            mask = block_mask.to_dense().bool()
            mask = mask.repeat_interleave(M.BLK, -2).repeat_interleave(M.BLK, -1)
            mask = mask[..., :q.shape[-2], :k.shape[-2]]
            mask = mask & torch.ones(q.shape[-2], k.shape[-2], device=q.device,
                                    dtype=torch.bool).tril()
            reference_masks.append(mask)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                  scale=scale, enable_gqa=enable_gqa)

        for seq in (64, 1024):
            with self.subTest(sequence_length=seq), torch.inference_mode(), \
                    torch.autocast("cuda", dtype=torch.bfloat16):
                x = torch.randn(1, seq, 32, device="cuda")
                pos = torch.arange(seq, device="cuda").view(1, 1, -1).expand(3, 1, -1)
                cos, sin = model.model.rotary_emb(x, pos)
                actual, _ = M.qsa_attention(attn, x, cos, sin, indexer, 2, False, False)
                with patch.object(M, "_flex", side_effect=explicit_attention):
                    expected, _ = M.qsa_attention(attn, x, cos, sin, indexer, 2, False, False)
                # BF16 kernels can differ in reduction order. This is not an
                # assertion of bitwise parity between distinct attention kernels.
                torch.testing.assert_close(actual, expected, rtol=.02, atol=2e-4)
                self.assertTrue(torch.isfinite(actual).all())
                if seq > 64:
                    causal = torch.ones(seq, seq, device="cuda", dtype=torch.bool).tril()
                    self.assertTrue((causal & ~reference_masks[-1][0, 0]).any())


if __name__ == "__main__":
    unittest.main()
