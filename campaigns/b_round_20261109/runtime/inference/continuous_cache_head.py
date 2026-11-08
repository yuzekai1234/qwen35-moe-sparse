"""Input-only continuous cache, with vocabulary work limited to query chunks.

Paper motivation: https://arxiv.org/abs/1612.04426. Cosine-normalized keys and
document resets are project adaptations. Coefficients must be frozen before A.
"""
import math

import torch
from torch.nn import functional as F


class CausalNeuralCacheHead:
    def __init__(self, theta, mixture, window=1024, query_chunk=128, eos_id=53):
        if theta < 0 or not 0 <= mixture < 1 or window < 1 or query_chunk < 1:
            raise ValueError('Invalid cache coefficient or window')
        self.theta, self.mixture = float(theta), float(mixture)
        self.window, self.query_chunk, self.eos_id = int(window), int(query_chunk), int(eos_id)

    @torch.inference_mode()
    def __call__(self, ids, hidden, base_logits, output_dtype=torch.bfloat16):
        if ids.ndim != 2 or hidden.shape[:2] != ids.shape or base_logits.shape[:2] != ids.shape:
            raise ValueError('Mismatched batch/sequence shapes')
        # Exact bypass allows a zero-coefficient baseline with identical logits.
        if self.mixture == 0:
            return base_logits if base_logits.dtype == output_dtype else base_logits.to(output_dtype)
        b, length = ids.shape
        keys = F.normalize(hidden.float(), dim=-1)
        positions = torch.arange(length, device=ids.device)
        start = torch.cummax(torch.where(ids == self.eos_id, positions[None], -1), 1).values.clamp_min(0)
        result = torch.empty_like(base_logits, dtype=output_dtype)
        for lo in range(0, length, self.query_chunk):
            hi = min(length, lo+self.query_chunk)
            logp = base_logits[:, lo:hi].float().log_softmax(-1)
            klo, khi = max(0, lo-self.window), hi-1
            if khi <= klo:
                result[:, lo:hi] = logp.to(output_dtype)
                continue
            query_positions = positions[lo:hi, None]
            key_positions = positions[None, klo:khi]
            allowed = ((key_positions < query_positions) &
                       (key_positions >= query_positions-self.window))[None]
            allowed = allowed & (key_positions[None] >= start[:, lo:hi, None])
            active = allowed.any(-1)
            scores = torch.bmm(keys[:, lo:hi], keys[:, klo:khi].transpose(1, 2)) * self.theta
            scores.masked_fill_(~allowed, -torch.inf)
            scores = torch.where(active[:, :, None], scores, torch.zeros_like(scores))
            attention = scores.softmax(-1) * allowed
            values = ids[:, klo+1:khi+1, None].transpose(1, 2).expand(b, hi-lo, khi-klo)
            cache = torch.zeros_like(logp)
            cache.scatter_add_(2, values, attention)
            combined = torch.logaddexp(logp + math.log1p(-self.mixture),
                                       cache.log() + math.log(self.mixture))
            result[:, lo:hi] = torch.where(active[:, :, None], combined, logp).to(output_dtype)
        return result

    def temporary_storage_bounds(self, batch, sequence, hidden, vocab):
        """Analytical storage sizes, not a measured CUDA peak or timing claim."""
        q = min(self.query_chunk, sequence)
        k = min(sequence, self.window+q-1)
        return dict(output_bf16_bytes=2*batch*sequence*vocab,
                    normalized_keys_fp32_bytes=4*batch*sequence*hidden,
                    each_vocabulary_tile_fp32_bytes=4*batch*q*vocab,
                    each_score_tile_fp32_bytes=4*batch*q*k,
                    full_cache_probability_tensor_allocated=False,
                    cuda_peak_unmeasured=True)
