"""Muon (Keller Jordan 的 Newton–Schulz 正交化 + Moonlight 的 update-RMS 对齐)。
- Nesterov 动量 0.95; NS 5 步, 系数 (3.4445, -4.7750, 2.0315); 3 维张量(MoE 专家 [E, A, B])按第 0 维逐片正交化。
- update ×= 0.2·sqrt(max(A,B)) 对齐 AdamW 的 update RMS(≈0.2), 因此可沿用 AdamW 的 lr 与解耦 weight decay。
- NS 的 FLOPs 属于优化器开销, 不计入模型 FLOPs。梯度已在 DDP 里同步后才调用 step()。"""
import os
import torch

NS_COEF = (3.4445, -4.7750, 2.0315)
MUON_MAX_CHUNK_ELEMENTS = 64 * 1024 * 1024


@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7):
    """G: [..., A, B] → 近似正交化(奇异值≈1), 在 bf16 里算。"""
    a, b, c = NS_COEF
    X = G.to(torch.bfloat16)
    tr = X.size(-2) > X.size(-1)
    if tr: X = X.transpose(-2, -1)
    X = X / (X.flatten(-2).norm(dim=-1).clamp(min=eps)[..., None, None])
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    if tr: X = X.transpose(-2, -1)
    return X


def _ns_iterations(X, steps):
    a, b, c = NS_COEF
    for _ in range(steps):
        gram = X @ X.transpose(-2, -1)
        polynomial = b * gram + c * (gram @ gram)
        X = a * X + polynomial @ X
    return X


_ns_iterations_cuda = torch.compile(_ns_iterations, fullgraph=True, dynamic=True)


@torch.no_grad()
def _newton_schulz_cuda(G, steps=5, eps=1e-7):
    X = G.to(torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    # Keep this large reduction outside Inductor's fused graph. A single fused
    # reduction over 448x224 expert matrices spills registers on A100 (~40 ms).
    scale = X.flatten(-2).norm(dim=-1).clamp_min(eps)
    X = X / scale[..., None, None]
    X = _ns_iterations_cuda(X, steps)
    return X.transpose(-2, -1) if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.1, ns_steps=5, rms_scale=0.2):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, rms_scale=rms_scale))

    @torch.no_grad()
    def step(self):
        if os.environ.get("Q38_MUON_BASELINE") or not any(p.is_cuda for g in self.param_groups for p in g["params"]):
            return self._reference_step()
        for group in self.param_groups:
            buckets = {}
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(parameter)
                key = (parameter.device, parameter.dtype, parameter.shape[-2:])
                buckets.setdefault(key, []).append(parameter)
            for (_, _, matrix_shape), parameters in buckets.items():
                rows, cols = matrix_shape
                # Bound temporary NS/update buffers for large C3/C4 expert buckets.
                # Split only between complete matrices; expert slices stay independent.
                limit = MUON_MAX_CHUNK_ELEMENTS
                matrices_per_chunk = max(1, limit // (rows * cols))
                chunks, chunk, used = [], [], 0
                for parameter in parameters:
                    pviews = parameter.reshape(-1, rows, cols).split(matrices_per_chunk)
                    gviews = parameter.grad.reshape(-1, rows, cols).split(matrices_per_chunk)
                    bviews = self.state[parameter]["buf"].reshape(-1, rows, cols).split(matrices_per_chunk)
                    for pview, gview, bview in zip(pviews, gviews, bviews):
                        if chunk and used + pview.numel() > limit:
                            chunks.append(chunk)
                            chunk, used = [], 0
                        chunk.append((pview, gview, bview))
                        used += pview.numel()
                if chunk:
                    chunks.append(chunk)
                for chunk in chunks:
                    pviews, gradients, buffers = map(list, zip(*chunk))
                    torch._foreach_mul_(buffers, group["momentum"])
                    torch._foreach_add_(buffers, gradients)
                    updates = torch._foreach_add(gradients, buffers, alpha=group["momentum"])
                    counts = [u.shape[0] for u in updates]
                    batched = torch.cat(updates)
                    orthogonal = _newton_schulz_cuda(batched, group["ns_steps"]).float()
                    pieces = list(orthogonal.split(counts))
                    del updates, batched
                    if group["weight_decay"]:
                        torch._foreach_mul_(pviews, 1 - group["lr"] * group["weight_decay"])
                    scale = group["rms_scale"] * max(rows, cols) ** .5
                    torch._foreach_add_(pviews, pieces, alpha=-group["lr"] * scale)

    @torch.no_grad()
    def _reference_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if "buf" not in st: st["buf"] = torch.zeros_like(p)
                buf = st["buf"]; buf.mul_(g["momentum"]).add_(p.grad)
                u = p.grad.add(buf, alpha=g["momentum"])            # Nesterov
                ns = newton_schulz if not p.is_cuda or bool(os.environ.get("Q38_MUON_BASELINE")) else _newton_schulz_cuda
                o = ns(u, g["ns_steps"]).float()
                o.mul_(g["rms_scale"] * max(p.shape[-2], p.shape[-1]) ** 0.5)
                if g["weight_decay"]: p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o, alpha=-g["lr"])


class HybridOptimizer:
    """Muon for hidden matrices; AdamW for embeddings, routers and scalar parameters."""

    def __init__(self, model, lr, weight_decay, muon=True):
        matrices, decay, no_decay = [], [], []
        for name, parameter in model.named_parameters():
            hidden_matrix = parameter.ndim >= 2 and not any(
                key in name for key in ("embed_tokens", "lm_head", "emb2", "emb3", ".mlp.gate.weight", "shared_expert_gate", "in_proj_a", "in_proj_b"))
            if ".mhc." in name:
                hidden_matrix = name.endswith((".A.weight", ".B.weight"))
            if muon and hidden_matrix:
                matrices.append(parameter)
            elif parameter.ndim >= 2 and not any(key in name for key in ("emb2", "emb3", ".mhc.")):
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        self.adam = torch.optim.AdamW([
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=next(model.parameters()).is_cuda)
        self.muon = Muon(matrices, lr=lr, weight_decay=weight_decay) if matrices else None
        self.param_groups = self.adam.param_groups + (self.muon.param_groups if self.muon else [])

    def step(self):
        self.adam.step()
        if self.muon:
            self.muon.step()

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"adam": self.adam.state_dict(), "muon": self.muon.state_dict() if self.muon else None}

    def load_state_dict(self, state):
        self.adam.load_state_dict(state["adam"])
        if self.muon:
            self.muon.load_state_dict(state["muon"])
        self.param_groups = self.adam.param_groups + (self.muon.param_groups if self.muon else [])
