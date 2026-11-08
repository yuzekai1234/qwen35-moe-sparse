"""Private paper-method prototypes; these do not patch any existing checkpoint.

Engram: https://arxiv.org/abs/2601.07372 (adaptation without tokenizer changes).
Loss-free routing: https://arxiv.org/abs/2408.15664 (sigmoid gate variant).
Neural cache: https://arxiv.org/abs/1612.04426 (cosine-key adaptation).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PaperEngram(nn.Module):
    """Orders 2/3, independent hash heads, context gate, causal dilated conv.

    Raw token IDs and EOS boundaries are preserved. Unlike the paper, there is
    no tokenizer canonicalization and no multi-branch residual backbone.
    """

    def __init__(self, hidden, vocab, rows, eos_id, heads=4, head_dim=128):
        super().__init__()
        self.hidden, self.vocab, self.eos_id = hidden, vocab, eos_id
        self.heads, self.head_dim = heads, head_dim
        # Distinct prime table sizes reduce correlated modulo collisions.
        def previous_prime(n):
            while n > 2:
                if all(n % d for d in range(2, math.isqrt(n) + 1)):
                    return n
                n -= 1
            raise ValueError('Table too small')
        sizes = []
        n = rows
        for _ in range(2 * heads):
            n = previous_prime(n)
            sizes.append(n)
            n -= 2
        self.sizes = tuple(sizes)
        self.tables = nn.ModuleList([nn.Embedding(n, head_dim) for n in sizes])
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.02)
        memory_dim = 2 * heads * head_dim
        self.key = nn.Linear(memory_dim, hidden, bias=False)
        self.value = nn.Linear(memory_dim, hidden, bias=False)
        self.query_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.key_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.conv_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.conv = nn.Conv1d(hidden, hidden, 4, groups=hidden, dilation=3, bias=False)
        # Start at the original N-gram module's small residual scale.
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.value.weight, std=0.002)
        nn.init.zeros_(self.conv.weight)

    def indices(self, ids):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).expand(b, t)
        start = torch.cummax(torch.where(ids == self.eos_id, pos, -1), 1).values.clamp_min(0)
        prior = []
        for shift in (2, 1, 0):
            count = min(shift, t)
            x = F.pad(ids[:, :t-count], (count, 0), value=self.vocab) if shift else ids
            prior.append(torch.where(pos - shift >= start, x, self.vocab))
        result = []
        for order in (2, 3):
            for head in range(self.heads):
                # Signed-int64 arithmetic intentionally wraps modulo 2**64,
                # as in the existing XOR/multiply N-gram hash.
                z = torch.zeros_like(ids)
                for j, x in enumerate(prior[3-order:]):
                    z = z ^ (x * (1000003 + 104729 * (head + 1) + 13007 * (j + 1)))
                z = (z ^ (z >> 17)) * (2654435761 + 2 * head)
                result.append(z % self.sizes[len(result)])
        return result

    def delta(self, hidden, ids):
        memory = torch.cat([tab(ix) for tab, ix in zip(self.tables, self.indices(ids))], -1)
        key = self.key_norm(self.key(memory))
        query = self.query_norm(hidden.to(key.dtype))
        gate = torch.sigmoid((query.float() * key.float()).sum(-1, keepdim=True) / math.sqrt(self.hidden))
        value = (gate * self.value(memory).float()).to(hidden.dtype)
        # Explicit left padding and per-tap EOS masking prevent future access
        # and prevent the short convolution from crossing document boundaries.
        normed = self.conv_norm(value.to(self.conv_norm.weight.dtype)).to(value.dtype)
        b, t, h = normed.shape
        pos = torch.arange(t, device=ids.device).expand(b, t)
        start = torch.cummax(torch.where(ids == self.eos_id, pos, -1), 1).values.clamp_min(0)
        convolved = torch.zeros_like(normed)
        for tap in range(4):
            lag = (3 - tap) * 3
            n = min(lag, t)
            shifted = F.pad(normed[:, :t-n], (0, 0, n, 0)) if lag else normed
            mask = (pos - lag >= start).unsqueeze(-1)
            convolved = convolved + shifted * mask * self.conv.weight[:, 0, tap].to(value.dtype)
        return value + F.silu(convolved)


class PaperLossFreeRouter(nn.Module):
    """Bias changes selection only; the trainer calls update_bias after a step."""

    def __init__(self, hidden, experts, topk, update_rate=0.001):
        super().__init__()
        self.hidden_dim, self.num_experts, self.top_k = hidden, experts, topk
        self.update_rate = update_rate
        self.weight = nn.Parameter(torch.zeros(experts, hidden))
        # A persistent non-gradient scalar per expert is still counted in N.
        self.routing_bias = nn.Parameter(torch.zeros(experts), requires_grad=False)
        # Plain temporary tensor: DDP must not broadcast rank0's accumulated
        # load over other ranks at the start of every microbatch forward.
        self.step_load = torch.zeros(experts, dtype=torch.int64)

    def forward(self, hidden):
        hidden = hidden.reshape(-1, self.hidden_dim)
        logits = F.linear(hidden, self.weight)
        scores = logits.float().sigmoid()
        selected = (scores + self.routing_bias.float()).topk(self.top_k, -1).indices
        weights = scores.gather(-1, selected)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                if self.step_load.device != selected.device:
                    self.step_load = torch.zeros(self.num_experts, dtype=torch.int64, device=selected.device)
                self.step_load.add_(torch.bincount(selected.reshape(-1), minlength=self.num_experts))
        return logits, weights.to(logits.dtype), selected

    @torch.no_grad()
    def update_bias(self, distributed=False):
        counts = self.step_load.clone()
        if distributed:
            torch.distributed.all_reduce(counts)
        error = counts.float().mean() - counts.float()
        self.routing_bias.add_(error.sign(), alpha=self.update_rate)
        # Adding a common scalar leaves top-k unchanged.
        self.routing_bias.sub_(self.routing_bias.mean())
        self.step_load.zero_()
        return counts


def neural_cache_distribution(hidden, ids, vocab, theta, window=1024, eos_id=53, query_chunk=128):
    """Return input-only cache probabilities; CPU prototype, chunked on GPU.

    At output position t, key i is paired with visible ids[i+1], and i<t.
    Every call starts empty and each batch row has independent state.
    """
    b, t, _ = hidden.shape
    keys = F.normalize(hidden.float(), dim=-1)
    positions = torch.arange(t, device=ids.device)
    starts = torch.cummax(torch.where(ids == eos_id, positions[None, :], -1), 1).values.clamp_min(0)
    output = torch.zeros((b, t, vocab), device=hidden.device, dtype=torch.float32)
    active = torch.zeros((b, t), device=hidden.device, dtype=torch.bool)
    for lo in range(0, t, query_chunk):
        hi = min(lo + query_chunk, t)
        klo, khi = max(0, lo - window), hi - 1
        if khi <= klo:
            continue
        qp = positions[lo:hi, None]
        kp = positions[None, klo:khi]
        allowed = (kp < qp) & (kp >= qp - window)
        allowed = allowed[None, :, :] & (kp[None, :, :] >= starts[:, lo:hi, None])
        similarity = torch.bmm(keys[:, lo:hi], keys[:, klo:khi].transpose(1, 2)) * theta
        present = allowed.any(-1)
        similarity.masked_fill_(~allowed, -torch.inf)
        similarity = torch.where(present[:, :, None], similarity, torch.zeros_like(similarity))
        weights = similarity.softmax(-1) * allowed
        values = ids[:, klo+1:khi+1, None].transpose(1, 2).expand(b, hi-lo, khi-klo)
        output[:, lo:hi].scatter_add_(2, values, weights)
        active[:, lo:hi] = present
    return output, active
