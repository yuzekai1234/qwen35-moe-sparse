"""Pretrain the scaled Qwen3.8-MoE text model (single GPU or torchrun DDP).

  python -m q38.train --preset q38_c2_m                                    # 单卡
  torchrun --standalone --nproc_per_node=2 -m q38.train --preset q38_c2_m  # 多卡, 全局 batch 不变
预设 configs/<name>.json 给出模型形状和学习率等默认值, 命令行参数覆盖预设。--tokens 缺省时按预设算力预算
和实际词表/序列长度重新核算(向下取整到完整全局 batch); --warmup 缺省为 1% 步数(至少 10)。
默认 --resume 1: 重跑同一命令从 run_dir/ckpt.pt 续训; 改模型/数据/预算/学习率策略时用新的 --run_dir。
--max_steps N 只把此次运行截止到绝对第 N 步, 不改变原预算的学习率曲线。
--micro_bs 是每卡 micro batch, 超过每卡序列数 batch_tokens/(seq×卡数) 时取后者。
--grad_ckpt 1: 按层 activation checkpointing, 层间只保存 mHC 残差流, 反向时重算各层 (约多一次前向, 重算不计入 FLOPs)。
  仅为在 40GB 卡上放下 C4 的临时措施, 正式提交代码前需改回。
--auto_micro_bs 1: 训练前从 --micro_bs 往小探测能放下的最大 micro batch (显存占用 ≤ --probe_mem_frac),
结果缓存到 run_dir/micro_bs_probe.json, 卡型/卡数/模型不变时续训直接复用。
默认数据: 共享 gzy 10K 语料 train.bin 全部用于训练, 过程验证取独立留出集 --val_data 的前 --eval_tokens;
--val_data "" 时改为从 train.bin 末尾留出 --eval_tokens。
输出 run_dir/: ckpt.pt (FP32 主权重+优化器+各 rank 随机状态), training_metrics.csv, tb/ (TensorBoard, --tensorboard 0 关闭),
run_config.json, summary.json, flops.json, hf/ (BF16 HF 导出, 携带模型源码)。
监控: tensorboard --logdir runs --port 6006
"""
import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .data import DEFAULT_DATA, DEFAULT_TOKENIZER, DEFAULT_VAL_DATA, TOKEN_DTYPE, load_tokenizer, open_tokens, tokenizer_sha256
from .flops import count as count_flops, validate_resume_accounting
from .models.qwen3_8_moe import Qwen3_8MoeForCausalLM, source_info
from .models.qwen3_8_moe import modeling_qwen3_8_moe as M
from .presets import MODEL_DEFAULTS, ROOT, budget_tokens, load_preset, load_training_config, model_config
from .sampling import ShuffledBlockSampler, sequential_windows

TRAIN_DEFAULTS = dict(run_dir="", data=str(DEFAULT_DATA), val_data=str(DEFAULT_VAL_DATA), tokenizer=str(DEFAULT_TOKENIZER),
                      seq=4096, tokens=0, compute=0.0, batch_tokens=524288, micro_bs=8, lr=3.6e-3, min_lr_ratio=0.1,
                      warmup=0, wd=0.1, clip=1.0, seed=0, log_every=10, eval_every=500, eval_tokens=2_000_000,
                      ckpt_every=200, resume=1, max_steps=0, tensorboard=1,
                      warmup_ratio=0.01, warmup_min=10, data_sampling="random", data_seed=-1,
                      eval_sampling="random", eval_offset=0, final_eval_tokens=0, final_eval_offset=0,
                      require_ready_pool=0, auto_micro_bs=0, probe_mem_frac=0.92, mtp_coef=0.3, kl_coef=0.05, muon=1, dense_warmup=1,
                      grad_ckpt=0)
# A resumed run must agree on everything that shapes the model, data stream or LR schedule.
RESUME_KEYS = ("layers", "hidden", "heads", "kv_heads", "head_dim", "lin_k_heads", "lin_v_heads", "lin_head_dim", "experts",
               "topk", "moe_ffn", "shared_ffn", "layer_types", "aux_coef", "rope_theta", "partial_rotary", "seq", "tokens",
               "batch_tokens", "seed", "data", "val_data", "tokenizer", "eval_tokens", "lr", "min_lr_ratio", "warmup", "wd", "clip",
               "data_sampling", "data_seed", "eval_sampling", "eval_offset", "final_eval_tokens", "final_eval_offset")
RESUME_KEYS += ("n_branch", "mhc_iters", "qsa_blocks", "indexer_heads", "indexer_head_dim", "ngram_rows", "mtp", "mtp_coef", "kl_coef", "muon", "dense_warmup")
NEW_RESUME_DEFAULTS = {k: TRAIN_DEFAULTS[k] for k in (
    "data_sampling", "data_seed", "eval_sampling", "eval_offset", "final_eval_tokens", "final_eval_offset")}


def int_like(s):   # accept 3e9-style token counts
    return int(float(s))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--preset", default="", help="configs/<preset>.json")
    source.add_argument("--config", default="", help="Training JSON file or directory containing config.json")
    ap.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                    help="Validate data, model shape and compute budget on CPU; do not create outputs or train")
    for k, v in {**MODEL_DEFAULTS, **TRAIN_DEFAULTS}.items():
        ap.add_argument(f"--{k}", type=int_like if k in ("tokens", "eval_tokens") else type(v), default=v)
    known, _ = ap.parse_known_args(argv)
    if known.preset or known.config:
        p = load_preset(known.preset) if known.preset else load_training_config(known.config)[1]
        bad_model = set(p["model"]) - set(MODEL_DEFAULTS)
        # Legacy presets also keep experts_impl (a model option) in train.
        bad_train = set(p["train"]) - ((set(TRAIN_DEFAULTS) | set(MODEL_DEFAULTS)) - {"compute"})
        if bad_model or bad_train:
            ap.error(f"Unknown config fields: model={sorted(bad_model)}, train={sorted(bad_train)}")
        defaults = dict(compute=p["compute"], run_dir=str(ROOT / "runs" / known.preset) if known.preset else "")
        defaults.update(p["model"]); defaults.update(p["train"])
        ap.set_defaults(**defaults)
    a = ap.parse_args(argv)
    if a.config:
        a.config = str(load_training_config(a.config)[0])
    for key in ("run_dir", "data", "val_data", "tokenizer"):
        value = getattr(a, key)
        if value:
            path = Path(value).expanduser()
            setattr(a, key, str(path if path.is_absolute() else ROOT / path))
    if not a.run_dir:
        ap.error("--run_dir is required without --preset")
    for name in ("layers", "hidden", "seq", "batch_tokens", "micro_bs", "log_every", "eval_every"):
        if getattr(a, name) <= 0:
            ap.error(f"--{name} must be positive")
    if a.grad_ckpt not in (0, 1):
        ap.error("--grad_ckpt must be 0 or 1")
    if a.auto_micro_bs not in (0, 1) or not 0 < a.probe_mem_frac <= 1:
        ap.error("--auto_micro_bs must be 0 or 1; --probe_mem_frac must be in (0,1]")
    if a.tokens < 0 or a.max_steps < 0 or a.ckpt_every < 0:
        ap.error("token and step limits must be nonnegative")
    if not a.tokens and not a.compute:
        ap.error("give --tokens or --compute (or a --preset)")
    if a.eval_tokens < a.seq + 2:
        ap.error("--eval_tokens must be at least seq + 2")
    if a.compute < 0 or a.lr <= 0 or not 0 <= a.min_lr_ratio <= 1:
        ap.error("compute must be nonnegative; lr must be positive; min_lr_ratio must be in [0,1]")
    if not 0 <= a.warmup_ratio < 1 or min(a.warmup, a.warmup_min, a.eval_offset, a.final_eval_tokens, a.final_eval_offset) < 0:
        ap.error("Invalid warmup or validation partition")
    if a.data_sampling not in ("random", "shuffled_blocks") or a.eval_sampling not in ("random", "sequential"):
        ap.error("Invalid training or evaluation sampling mode")
    if a.data_seed < -1:
        ap.error("data_seed must be -1 (use model seed) or nonnegative")
    return a


def _data_identity(path):
    path = Path(path)
    stat = path.stat()
    result = dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    meta_path = path.parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        result["declared_sha256"] = meta.get("train_sha256" if path.name == "train.bin" else "valid_sha256")
    return result


def micro_bs_candidates(per_rank, limit):
    """Divisors of the per-rank sequence count up to limit, largest first."""
    return [m for m in range(min(per_rank, limit), 0, -1) if per_rank % m == 0]


def prepare_inputs(a, world=1):
    """Read-only preflight, also used by --dry-run before any CUDA initialization."""
    if world < 1:
        raise ValueError("WORLD_SIZE must be positive")
    tok = load_tokenizer(a.tokenizer)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS ID")
    tok_sha = tokenizer_sha256(a.tokenizer)
    cfg = model_config(vars(a), len(tok), a.seq, tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
    fc = count_flops(cfg.to_dict(), a.seq)
    if not a.tokens:
        a.tokens = budget_tokens(fc["train_flops_per_token"], a.compute, a.batch_tokens)
    steps = a.tokens // a.batch_tokens
    if steps < 1:
        raise ValueError("Budget must cover at least one global batch")
    per_rank, rem = divmod(a.batch_tokens, a.seq * world)
    if rem or not per_rank:
        raise ValueError("batch_tokens must be divisible by seq*WORLD_SIZE")
    a.micro_bs = min(a.micro_bs, per_rank)
    if not a.auto_micro_bs and per_rank % a.micro_bs:
        raise ValueError("batch_tokens must be divisible by micro_bs*seq*WORLD_SIZE")
    if not a.warmup:
        a.warmup = min(steps, max(a.warmup_min, round(steps * a.warmup_ratio)))
    if a.warmup > steps:
        raise ValueError("Warmup exceeds the full training horizon")
    a.data_seed = a.seed if a.data_seed == -1 else a.data_seed
    data = open_tokens(a.data, tok_sha)
    validation = open_tokens(a.val_data, tok_sha) if a.val_data else data[-a.eval_tokens:]
    train = data if a.val_data else data[:-a.eval_tokens]
    if len(train) < a.seq + 2 or len(validation) < a.seq + 2:
        raise ValueError("Not enough training or validation tokens")
    if a.require_ready_pool:
        pool = Path(a.data).parent
        meta = json.loads((pool / "meta.json").read_text())
        verified = json.loads((pool / "verification.json").read_text())
        if not (pool / "READY").is_file() or meta.get("state") != "ready" or verified.get("state") != "ready":
            raise ValueError("Training pool is not verified READY")
        if meta["tokenizer_sha256"] != tok_sha or len(data) != meta["train_tokens"]:
            raise ValueError("Training pool identity does not match its manifest")
        if verified["train_sha256"] != meta["train_sha256"]:
            raise ValueError("Pool verification and manifest disagree")
    sampler = ShuffledBlockSampler(len(train), a.seq, a.data_seed) if a.data_sampling == "shuffled_blocks" else None
    if sampler and steps * (a.batch_tokens // a.seq) > sampler.blocks:
        raise ValueError("Training budget exceeds unique sequence blocks; corpus repetition is disabled")

    def partition(offset, targets):
        if offset < 0 or offset + targets + 1 > len(validation):
            raise ValueError("Requested evaluation partition exceeds validation corpus")
        return validation[offset:offset + targets + 1]

    dev = partition(a.eval_offset, a.eval_tokens) if a.eval_sampling == "sequential" else validation[:a.eval_tokens]
    final = partition(a.final_eval_offset, a.final_eval_tokens) if a.final_eval_tokens else None
    if final is not None and not a.val_data:
        raise ValueError("Final evaluation requires an independent validation file")
    identity = dict(train=_data_identity(a.data), validation=_data_identity(a.val_data) if a.val_data else None)
    plan = dict(config=a.config, run_dir=a.run_dir, compute_budget=a.compute,
                train_tokens=steps * a.batch_tokens, steps=steps, warmup_steps=a.warmup,
                actual_train_flops=fc["train_flops_per_token"] * steps * a.batch_tokens,
                train_flops_per_token=fc["train_flops_per_token"],
                flops_accounting=fc["accounting_method"],
                megatron_reference_train_flops_per_token=fc["megatron_reference_train_flops_per_token"],
                total_params=fc["total_nonvocab_params"] + fc["embed_params"],
                active_nonvocab_params=fc["active_nonvocab_params"],
                seq=a.seq, batch_tokens=a.batch_tokens, micro_bs="auto" if a.auto_micro_bs else a.micro_bs, world=world,
                micro_bs_candidates=micro_bs_candidates(per_rank, a.micro_bs) if a.auto_micro_bs else [a.micro_bs],
                accumulation=None if a.auto_micro_bs else per_rank // a.micro_bs,
                lr=a.lr, data=a.data, val_data=a.val_data, tokenizer=a.tokenizer,
                tokenizer_sha256=tok_sha, data_identity=identity,
                sampling=sampler.metadata() if sampler else dict(mode="random", replacement=True, seed=a.data_seed),
                eval_sampling=a.eval_sampling, eval_offset=a.eval_offset, eval_targets=len(dev)-1,
                final_eval_offset=a.final_eval_offset, final_eval_targets=len(final)-1 if final is not None else 0,
                config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest() if a.config else None)
    return tok, cfg, fc, train, dev, final, sampler, plan


class _LinearCESum(torch.nn.Module):
    def forward(self, weight, hidden, target):
        return torch.nn.functional.cross_entropy(torch.nn.functional.linear(hidden, weight), target, reduction="sum")


class LossModel(torch.nn.Module):
    """Fused-linear main/MTP CE, indexer KL and router balancing; DDP wraps all losses.

    It uses the Q38 trunk directly to avoid materialising vocabulary logits."""

    def __init__(self, lm, experts, topk, ce=None):
        super().__init__()
        if ce is None:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            ce = LigerFusedLinearCrossEntropyLoss()
        self.lm, self.experts, self.topk = lm, experts, topk
        self.ce = ce
        self._router, self._recording = [], False
        self._router_hooks = [layer.mlp.gate.register_forward_hook(self._record_router)
                              for layer in list(lm.model.layers) + ([lm.mtp["block"]] if lm.mtp else [])]
        if lm.mtp is not None:
            if type(ce).__name__ == "LigerFusedLinearCrossEntropyLoss":
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
                self.mtp_ce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
            else:
                self.mtp_ce = _LinearCESum()

    def _record_router(self, module, args, output):
        # _recording excludes activation-checkpoint recomputation, which reruns the gates during backward.
        if self._recording and self.training and torch.is_grad_enabled():
            self._router.append(output[0])

    def forward(self, x, y):
        m = self.lm
        router = self._router
        router.clear()
        self._recording = True
        try:
            hs, kls = m.model.trunk(x)
            mh = m.mtp_hidden(hs, y) if m.mtp is not None else None
        finally:
            self._recording = False
        h = m.model.norm(hs)
        ce = self.ce(m.lm_head.weight, h.reshape(-1, h.size(-1)), y.reshape(-1))
        mtp_loss = ce.new_zeros(())
        if m.mtp is not None:
            target = torch.cat([y[:, 1:], torch.full_like(y[:, :1], -100)], 1)
            target = target.masked_fill(y == m.config.eos_token_id, -100)
            # Sum / clamped count handles all-EOS batches without a GPU->CPU branch.
            valid = (target != -100).sum()
            mtp_loss = self.mtp_ce(m.lm_head.weight, mh.reshape(-1, mh.size(-1)), target.reshape(-1)) / valid.clamp_min(1)
        kl = torch.stack(kls).mean() if kls else ce.new_zeros(())
        aux = M.load_balancing_loss_func(tuple(router), self.experts, self.topk)
        router.clear()
        return ce, aux, mtp_loss, kl


def gdn_kernel():
    """Transformers swaps in fla's chunk_gated_delta_rule when fla is installed; the torch fallback is far slower."""
    impl = [c.cell_contents.__module__ for c in (M.torch_chunk_gated_delta_rule.__closure__ or [])
            if callable(c.cell_contents) and c.cell_contents.__module__.startswith("fla")]
    if not impl:
        raise RuntimeError("chunk_gated_delta_rule is not using the fla kernel")
    return impl


def _autocast_grouped_mm(input, weight, offs):
    """transformers' grouped_mm casts the input to the FP32 master weights, so on torch<2.9 / non-Hopper
    its per-expert fallback runs FP32 SGEMM; under autocast run it in the autocast dtype like F.linear."""
    if torch.is_autocast_enabled(input.device.type):
        dt = torch.get_autocast_dtype(input.device.type)
        input, weight = input.to(dt), weight.to(dt)
    return _autocast_grouped_mm.original(input, weight, offs=offs)


def install_fast_paths(model):
    """Training fast paths with the model's numerics up to BF16 rounding order (see tests/test_model.py):
    native grouped_mm on supported GPUs, padded batched GEMMs on older A100 Torch, FLA causal conv1d and
    grouped-value heads in Gated DeltaNet (shared with q38.inference), compiled RMSNorm / gated RMSNorm."""
    import types
    import transformers.integrations.moe as moe
    from .inference import _gdn_prefill_forward, compile_inference_norms, padded_batched_experts_forward
    if moe._grouped_mm is not _autocast_grouped_mm:
        _autocast_grouped_mm.original, moe._grouped_mm = moe._grouped_mm, _autocast_grouped_mm
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    ALL_EXPERTS_FUNCTIONS.register("q38_batched_mm", padded_batched_experts_forward)
    parameter = next(model.parameters())
    native_grouped = parameter.is_cuda and (
        torch.cuda.get_device_capability(parameter.device)[0] >= 9
        or tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 9))
    implementation = "grouped_mm" if native_grouped or os.environ.get("Q38_GROUPED_BASELINE") == "1" else "q38_batched_mm"
    model.set_experts_implementation(implementation)
    gdn = 0
    for module in model.modules():
        if isinstance(module, M.Qwen3_8MoeGatedDeltaNet):
            module._reference_forward = module.forward
            module.forward = types.MethodType(_gdn_prefill_forward, module)
            gdn += 1
    return dict(experts=implementation, gated_deltanet=gdn, norms=compile_inference_norms(model))


def sample_batch(rng, src, bs, seq, dev, offsets=None):
    ix = rng.integers(0, len(src) - seq - 1, bs) if offsets is None else offsets
    x = torch.from_numpy(np.stack([src[i:i + seq].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([src[i + 1:i + 1 + seq].astype(np.int64) for i in ix]))
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)


def probe_micro_bs(lm, model, train_data, a, dev, world, ddp, log0):
    """Largest micro_bs (at most --micro_bs) whose fwd+bwd peak plus AdamW/DDP state fits in probe_mem_frac of GPU memory.

    Runs on the unwrapped LossModel (no gradient all-reduce, so one rank's OOM cannot hang the others) and never
    steps the optimizer; ranks agree via all_reduce(MIN). Global RNG states are restored afterwards."""
    cands = micro_bs_candidates(a.batch_tokens // (a.seq * world), a.micro_bs)
    if len(cands) == 1:
        return cands[0], []
    total = torch.cuda.get_device_properties(dev).total_memory
    reserve = sum(p.numel() for p in model.parameters()) * 4 * (2 + ddp)   # exp_avg/exp_avg_sq (+ DDP grad buckets)
    rng_state = torch.get_rng_state(), torch.cuda.get_rng_state(dev)
    prng = np.random.default_rng(0); chosen = 0; trials = []
    for m in cands:
        peak = None
        try:
            torch.cuda.reset_peak_memory_stats(dev)
            for _ in range(2):   # the second pass accumulates into existing gradients, as in training
                x, y = sample_batch(prng, train_data, m, a.seq, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ce, aux, mtp_loss, kl = lm(x, y)
                (ce + a.aux_coef * aux + a.mtp_coef * mtp_loss + a.kl_coef * kl).backward()
                del x, y, ce, aux, mtp_loss, kl
            torch.cuda.synchronize(dev)
            peak = torch.cuda.max_memory_allocated(dev)
        except torch.cuda.OutOfMemoryError:
            pass
        model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        ok = peak is not None and peak + reserve <= a.probe_mem_frac * total
        if ddp:
            t = torch.tensor(int(ok), device=dev); dist.all_reduce(t, op=dist.ReduceOp.MIN); ok = bool(t.item())
        trials.append(dict(micro_bs=m, peak_bytes=peak, reserve_bytes=reserve, fits=ok))
        log0(f"micro_bs 探测 {m}: " + (f"峰值 {peak/2**30:.2f}G + 优化器 {reserve/2**30:.2f}G" if peak is not None else "OOM")
             + f" / {a.probe_mem_frac:.0%}×{total/2**30:.1f}G (rank0) -> {'可用' if ok else '放不下'}")
        if ok:
            chosen = m; break
    torch.set_rng_state(rng_state[0]); torch.cuda.set_rng_state(rng_state[1], dev)
    torch.cuda.reset_peak_memory_stats(dev)
    if not chosen:
        raise RuntimeError("micro_bs=1 does not fit in GPU memory")
    return chosen, trials


def lr_schedule(a, steps):
    def lr_at(s):   # linear warmup, cosine decay to min_lr_ratio × lr
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        pr = (s - a.warmup) / max(1, steps - a.warmup - 1)
        return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * pr)))
    return lr_at


@torch.no_grad()
def evaluate(model, eval_data, seq, dev, sampling="random", distributed=False):
    """Pure CE; sequential mode scores each target once, including the tail."""
    was_training = model.training
    model.eval(); tot = 0.0; n = 0
    erng = np.random.default_rng(1234)
    windows = sequential_windows(len(eval_data), seq) if sampling == "sequential" else [(None, seq)] * (len(eval_data) // seq)
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    try:
        for start, size in windows[rank::world]:
            if start is None:
                x, y = sample_batch(erng, eval_data, 1, size, dev)
            else:
                x = torch.from_numpy(np.asarray(eval_data[start:start+size], dtype=np.int64)[None].copy()).to(dev)
                y = torch.from_numpy(np.asarray(eval_data[start+1:start+size+1], dtype=np.int64)[None].copy()).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(input_ids=x, use_cache=False).logits
            fl = logits.view(-1, logits.size(-1)); yl = y.view(-1)
            chunk = max(256, (8192 * 32768) // logits.size(-1))
            for o in range(0, yl.numel(), chunk):
                tot += torch.nn.functional.cross_entropy(fl[o:o+chunk].float(), yl[o:o+chunk], reduction="sum").item()
            n += y.numel()
        if distributed:
            totals = torch.tensor([tot, n], dtype=torch.float64, device=dev)
            dist.all_reduce(totals)
            tot, n = totals.tolist()
        if not n:
            raise ValueError("No validation targets were scored")
        return tot / n
    finally:
        model.train(was_training)


def main(argv=None):
    a = parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tok, cfg, fc, train_data, eval_data, final_data, sampler, plan = prepare_inputs(a, world)
    if a.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan
    os.makedirs(a.run_dir, exist_ok=True)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if ddp:
        if local_rank >= torch.cuda.device_count():
            raise ValueError("DDP ranks exceed visible GPUs")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    torch.manual_seed(a.seed + rank)
    dev = torch.device("cuda", local_rank); torch.cuda.set_device(dev)

    def log0(*x):
        if rank == 0: print(*x, flush=True)

    # Tokenizer, data and budget were validated before allocating the GPU model.
    tok_sha = tokenizer_sha256(a.tokenizer)
    log0(f"Tokenizer: {a.tokenizer}, vocab={len(tok)}, bos={tok.bos_token_id}, eos={tok.eos_token_id}, pad={tok.pad_token_id}")
    model = Qwen3_8MoeForCausalLM(cfg).to(dev, dtype=torch.float32)
    model.model.gradient_checkpointing = bool(a.grad_ckpt)
    log0("Model source:", M.__file__)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "embeddings are not tied"
    log0("GDN kernel:", gdn_kernel())
    log0("Fast paths:", install_fast_paths(model))
    n_all = sum(p.numel() for p in model.parameters()); n_total_nv = n_all - model.get_input_embeddings().weight.numel()
    n_active_nv = fc["active_nonvocab_params"]
    flops_per_token = fc["train_flops_per_token"]
    assert (fc["total_nonvocab_params"], fc["active_nonvocab_params"]) == (n_total_nv, n_active_nv), "flops.py param count drifted from the model"
    log0(f"层型 {cfg.layer_types}\n参数 总={n_all/1e6:.2f}M 非词表 总={n_total_nv/1e6:.2f}M 激活={n_active_nv/1e6:.3f}M (每 token) | "
         f"每token训练 {flops_per_token/1e6:.1f} MFLOPs {fc['breakdown']} | 预算 {a.tokens:.3e} token ≈ {flops_per_token*a.tokens:.3e} FLOPs")

    log0(f"数据 训练 {len(train_data)/1e9:.2f}B token, 验证 {len(eval_data)/1e6:.1f}M token ({a.val_data or 'train.bin 末尾'}), mmap (world={world})")
    lm = LossModel(model, a.experts, a.topk)
    if a.auto_micro_bs:
        gpu = torch.cuda.get_device_properties(dev)
        key = dict(world=world, gpu=gpu.name, gpu_bytes=gpu.total_memory, torch=torch.__version__, fast_paths=1,
                   probe_mem_frac=a.probe_mem_frac, grad_ckpt=a.grad_ckpt, max_micro_bs=a.micro_bs, seq=a.seq, batch_tokens=a.batch_tokens,
                   model_sha256=hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest())
        probe_path = Path(a.run_dir) / "micro_bs_probe.json"
        cached = torch.tensor(0, device=dev)
        if rank == 0 and probe_path.exists():
            previous = json.loads(probe_path.read_text())
            if previous.get("key") == key:
                cached.fill_(previous["micro_bs"])
        if ddp: dist.broadcast(cached, 0)
        if cached.item():
            a.micro_bs = int(cached.item()); log0(f"micro_bs {a.micro_bs} (复用 {probe_path})")
        else:
            a.micro_bs, trials = probe_micro_bs(lm, model, train_data, a, dev, world, ddp, log0)
            if rank == 0:
                probe_path.write_text(json.dumps(dict(key=key, micro_bs=a.micro_bs, trials=trials), indent=1) + "\n")
    per_micro = a.micro_bs * a.seq * world
    if a.batch_tokens % per_micro:
        raise ValueError(f"batch_tokens {a.batch_tokens} must be divisible by micro_bs×seq×world={per_micro}")
    steps = a.tokens // a.batch_tokens; accum = a.batch_tokens // per_micro
    stop_step = min(steps, a.max_steps) if a.max_steps else steps
    rng = np.random.default_rng(a.data_seed + rank)

    if ddp:
        lm = torch.nn.parallel.DistributedDataParallel(lm, device_ids=[local_rank], find_unused_parameters=True)
    from .muon import HybridOptimizer
    opt = HybridOptimizer(model, lr=a.lr, weight_decay=a.wd, muon=bool(a.muon))
    lr_at = lr_schedule(a, steps)

    ckpt_path = f"{a.run_dir}/ckpt.pt"; resuming = a.resume and os.path.exists(ckpt_path)
    if resuming:
        previous = json.load(open(f"{a.run_dir}/run_config.json"))
        validate_resume_accounting(previous, fc)
        if previous.get("tokenizer_sha256", tok_sha) != tok_sha:
            raise ValueError("Unsafe resume: tokenizer contents changed; use a new run_dir")
        for key in RESUME_KEYS:
            fallback = previous.get("seed", 0) if key == "data_seed" else NEW_RESUME_DEFAULTS.get(key)
            if previous.get(key, fallback) != vars(a)[key]:
                raise ValueError(f"Unsafe resume: {key} changed; use a new run_dir")
        if "data_identity" in previous and previous["data_identity"] != plan["data_identity"]:
            raise ValueError("Unsafe resume: corpus identity changed")
        if sampler and previous.get("sampling") != plan["sampling"]:
            raise ValueError("Unsafe resume: shuffled block order changed")
    if ddp: dist.barrier()
    if rank == 0:
        json.dump({**vars(a), "tokenizer_sha256": tok_sha, "data_dtype": TOKEN_DTYPE.str, "model_code": source_info(),
                   "data_identity": plan["data_identity"], "sampling": plan["sampling"], "config_sha256": plan["config_sha256"],
                   "n_params": n_all, "n_nonvocab_total": n_total_nv, "n_nonvocab_active": n_active_nv,
                   "flops_per_token": flops_per_token, "flops_accounting": fc["accounting_method"],
                   "flops_breakdown": fc["breakdown"], "steps": steps, "accum": accum,
                   "config": cfg.to_dict()}, open(f"{a.run_dir}/run_config.json", "w"), indent=1, ensure_ascii=False)
    metrics = csv.writer(open(f"{a.run_dir}/training_metrics.csv", "a" if resuming else "w", newline="")) if rank == 0 else None
    if metrics and not resuming:
        metrics.writerow(["step", "tokens", "loss", "aux", "lr", "grad_norm", "tok_per_s", "elapsed_s", "val_loss", "active_nonvocab", "total_nonvocab", "mtp_loss", "indexer_kl"])

    model.train(); t0 = time.time(); t_last = t0; tok_seen = 0; start_step = 0
    if resuming:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_step = ck["step"] + 1; tok_seen = ck["tok_seen"]
        if sampler and tok_seen != start_step * a.batch_tokens:
            raise ValueError("Checkpoint step/token count does not match the deterministic data stream")
        if len(ck.get("rank_rng", [])) == world:   # same world size: exact per-rank sample stream
            state = ck["rank_rng"][rank]
            rng.bit_generator.state = state["numpy"]
            torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state(state["cuda"], dev)
        elif rank == 0:
            rng.bit_generator.state = ck["rng"]
        else:
            rng = np.random.default_rng([a.data_seed + rank, start_step])
        if ck.get("world", 1) != world:
            log0("World size changed: global shuffled-block order is preserved" if sampler else
                 "World size changed: optimizer/token count restored; random sample stream differs")
        log0(f"从 {ckpt_path} 恢复：step {start_step}, tok_seen {tok_seen}"); del ck
    tb = None
    if rank == 0 and a.tensorboard:   # purge_step drops events a crashed run logged past the restored checkpoint
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(f"{a.run_dir}/tb", purge_step=start_step if resuming else None)
        if not resuming:
            tb.add_text("run_config", f"```\n{json.dumps(vars(a), indent=1, ensure_ascii=False)}\n```")

    def save_ckpt(step):
        state = {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(dev)}
        states = [None] * world
        if ddp: dist.all_gather_object(states, state)
        else: states[0] = state
        if rank == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "tok_seen": tok_seen,
                        "rng": rng.bit_generator.state, "rank_rng": states, "world": world}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    log0(f"steps {steps} accum {accum} micro_bs {a.micro_bs} warmup {a.warmup} lr {a.lr}")
    for step in range(start_step, stop_step):
        model.model.qsa_dense = bool(a.dense_warmup and step < a.warmup)
        for g in opt.param_groups: g["lr"] = lr_at(step)
        totals = torch.zeros(4, dtype=torch.float32, device=dev)
        for mi in range(accum):
            offsets = sampler.micro_offsets(step, mi, rank, world, a.micro_bs, a.batch_tokens) if sampler else None
            x, y = sample_batch(rng, train_data, a.micro_bs, a.seq, dev, offsets=offsets)
            sync = lm.no_sync() if (ddp and mi < accum - 1) else contextlib.nullcontext()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                ce, aux, mtp_loss, kl = lm(x, y)
                loss = (ce + a.aux_coef * aux + a.mtp_coef * mtp_loss + a.kl_coef * kl) / accum
            loss.backward()
            totals += torch.stack([ce.detach(), torch.as_tensor(aux, device=dev).detach(),
                                   mtp_loss.detach(), kl.detach()]).float() / accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); opt.zero_grad(set_to_none=True); tok_seen += accum * per_micro
        if ddp:
            dist.all_reduce(totals); totals /= world
        # One host transfer per optimizer step, rather than four synchronizations per micro batch.
        loss_acc, aux_acc, mtp_acc, kl_acc, gn = torch.cat([totals, grad_norm.detach().reshape(1)]).tolist()
        if tb:
            for k, v in (("loss", loss_acc), ("aux", aux_acc), ("mtp_loss", mtp_acc), ("indexer_kl", kl_acc), ("lr", lr_at(step)), ("grad_norm", gn), ("tokens", tok_seen)):
                tb.add_scalar(f"train/{k}", v, step)
        if a.ckpt_every and step > 0 and step % a.ckpt_every == 0: save_ckpt(step)
        if (step % a.log_every == 0 or step == stop_step - 1) and rank == 0:
            now = time.time(); tps = min(a.log_every, step - start_step + 1) * accum * per_micro / max(1e-6, now - t_last); t_last = now
            vl = evaluate(model, eval_data, a.seq, dev, a.eval_sampling) if (step % a.eval_every == 0 and step > 0) or step == stop_step - 1 else ""
            metrics.writerow([step, tok_seen, f"{loss_acc:.8f}", f"{aux_acc:.8f}", f"{lr_at(step):.8e}", f"{gn:.6f}", f"{tps:.0f}",
                              f"{now-t0:.3f}", f"{vl:.8f}" if vl != "" else "", n_active_nv, n_total_nv,
                              f"{mtp_acc:.8f}", f"{kl_acc:.8f}"])
            print(f"step {step}/{steps} tok {tok_seen/1e6:.0f}M loss {loss_acc:.4f} mtp {mtp_acc:.3f} kl {kl_acc:.4f} aux {aux_acc:.3f} lr {lr_at(step):.2e} gn {gn:.2f} "
                  f"{tps:.0f} tok/s {(now-t0)/60:.1f}min mem {torch.cuda.max_memory_allocated(dev)/2**30:.1f}G"
                  + (f" | val {vl:.4f}" if vl != "" else ""), flush=True)
            if tb:
                tb.add_scalar("perf/tok_per_s", tps, step)
                tb.add_scalar("perf/max_mem_gib", torch.cuda.max_memory_allocated(dev) / 2**30, step)
                if vl != "": tb.add_scalar("val/loss", vl, step)
                tb.flush()
            # Validation latency belongs to evaluation, not the next training throughput interval.
            t_last = time.time()

    # Every rank participates; save FP32 master weights before the BF16 export.
    if stop_step > start_step: save_ckpt(stop_step - 1)
    final_loss = None
    if final_data is not None:
        final_loss = evaluate(model, final_data, a.seq, dev, "sequential", distributed=ddp)
        if rank == 0:
            json.dump(dict(loss=final_loss, metric="pure_cross_entropy", scored_tokens=len(final_data)-1,
                           train_flops=flops_per_token*tok_seen, flops_accounting=fc["accounting_method"],
                           offset=a.final_eval_offset, tokenizer_sha256=tok_sha,
                           tokens_trained=tok_seen, completed_budget=tok_seen >= steps*a.batch_tokens),
                      open(f"{a.run_dir}/final_eval.json", "w"), indent=2)
            log0(f"Final fixed validation CE: {final_loss:.8f} ({len(final_data)-1:,} targets)")
    if tb: tb.close()
    if ddp: dist.barrier()
    if rank == 0:
        model.to(torch.bfloat16).save_pretrained(f"{a.run_dir}/hf"); tok.save_pretrained(f"{a.run_dir}/hf")
        json.dump({"tokens_trained": tok_seen, "train_flops": flops_per_token * tok_seen, "active_nonvocab": n_active_nv,
                   "flops_accounting": fc["accounting_method"],
                   "total_nonvocab": n_total_nv, "elapsed_s": time.time() - t0, "world": world,
                   "completed_budget": tok_seen >= steps * a.batch_tokens,
                   "final_eval_loss": final_loss, "sampling": plan["sampling"]}, open(f"{a.run_dir}/summary.json", "w"), indent=1)
        json.dump({**fc, "train_tokens": tok_seen, "train_flops": flops_per_token * tok_seen}, open(f"{a.run_dir}/flops.json", "w"), indent=2)
        print("DONE 已导出 HF 目录", f"{a.run_dir}/hf", flush=True)
    if ddp: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
