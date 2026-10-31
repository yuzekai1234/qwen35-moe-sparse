"""Chunked trunk prefill with the output head writing into one full-batch logits tensor.

On H100 the compiled trunk costs about 10% more per token at batch 940 than at batch
256 to 768, while the contest evaluates at the largest batch that fits. The trunk
therefore runs on fixed chunks of Q38_TRUNK_CHUNK sequences in two phases:

1. every chunk runs the trunk, final RMSNorm and FP8 quantization, keeping only the
   FP8 rows and their scales (1156 bytes per token);
2. the BF16 logits for the whole batch are allocated and each chunk's FP8 output-head
   GEMM writes its rows in place.

Activations of phase 1 never coexist with the logits, so the peak memory stays close
to logits plus the FP8 head input, and no logits are copied.
"""
import os

import torch


def chunk_size():
    value = int(os.environ.get('Q38_TRUNK_CHUNK', '0'))
    if value < 0:
        raise ValueError('Q38_TRUNK_CHUNK must be nonnegative')
    return value


def build(model, compile_function, size):
    head = model.lm_head
    assert head._q8_bias is None and head._q8_weight.shape[0] == head.out_features
    weight = head._q8_weight.t()
    weight_scale = head._q8_weight_scale[None, :]
    vocab = head.out_features

    def trunk(ids):
        from prequant_norm import quant_rows
        hidden, _ = model.model.trunk(ids)
        normed = model.model.norm(hidden)
        return quant_rows(normed.reshape(-1, normed.shape[-1]))

    compiled = compile_function(trunk)

    def forward(ids):
        batch, length = ids.shape
        heads = [compiled(ids[start:start + size]) for start in range(0, batch, size)]
        logits = torch.empty((batch * length, vocab), device=ids.device, dtype=torch.bfloat16)
        row = 0
        for q, scale in heads:
            torch.ops.aten._scaled_mm.out(q, weight, scale[:, None], weight_scale, bias=None,
                                          out_dtype=torch.bfloat16, use_fast_accum=True,
                                          out=logits[row:row + q.shape[0]])
            row += q.shape[0]
        del heads
        return logits.view(batch, length, vocab)

    return forward
