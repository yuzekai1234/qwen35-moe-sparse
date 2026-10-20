"""Pretrain the scaled Qwen3.5-MoE text model (single GPU or torchrun DDP).

  python -m q35.train --preset q35_c2_m                                    # 单卡
  torchrun --standalone --nproc_per_node=2 -m q35.train --preset q35_c2_m  # 多卡, 全局 batch 不变
预设 configs/<name>.json 给出模型形状和学习率等默认值, 命令行参数覆盖预设。--tokens 缺省时按预设算力预算
和实际词表/序列长度重新核算(向下取整到完整全局 batch); --warmup 缺省为 1% 步数(至少 10)。
默认 --resume 1: 重跑同一命令从 run_dir/ckpt.pt 续训; 改模型/数据/预算/学习率策略时用新的 --run_dir。
--max_steps N 只把此次运行截止到绝对第 N 步, 不改变原预算的学习率曲线。
默认数据: 共享 gzy 30K 语料 train.bin 全部用于训练, 过程验证取独立留出集 --val_data 的前 --eval_tokens;
--val_data "" 时改为从 train.bin 末尾留出 --eval_tokens。
输出 run_dir/: ckpt.pt (FP32 主权重+优化器+各 rank 随机状态), training_metrics.csv, tb/ (TensorBoard, --tensorboard 0 关闭),
run_config.json, summary.json, flops.json, hf/ (BF16 HF 导出, 携带模型源码)。
监控: tensorboard --logdir runs --port 6006
"""
import argparse
import contextlib
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from .data import DEFAULT_DATA, DEFAULT_TOKENIZER, DEFAULT_VAL_DATA, TOKEN_DTYPE, load_tokenizer, open_tokens, tokenizer_sha256
from .flops import count as count_flops
from .models.qwen3_5_moe import Qwen3_5MoeForCausalLM, source_info
from .models.qwen3_5_moe import modeling_qwen3_5_moe as M
from .presets import MODEL_DEFAULTS, ROOT, budget_tokens, load_preset, model_config

TRAIN_DEFAULTS = dict(run_dir="", data=str(DEFAULT_DATA), val_data=str(DEFAULT_VAL_DATA), tokenizer=str(DEFAULT_TOKENIZER),
                      seq=8192, tokens=0, compute=0.0, batch_tokens=524288, micro_bs=8, lr=3.6e-3, min_lr_ratio=0.1,
                      warmup=0, wd=0.1, clip=1.0, seed=0, log_every=10, eval_every=500, eval_tokens=2_000_000,
                      ckpt_every=200, resume=1, max_steps=0, tensorboard=1)
# A resumed run must agree on everything that shapes the model, data stream or LR schedule.
RESUME_KEYS = ("layers", "hidden", "heads", "kv_heads", "head_dim", "lin_k_heads", "lin_v_heads", "lin_head_dim", "experts",
               "topk", "moe_ffn", "shared_ffn", "layer_types", "aux_coef", "rope_theta", "partial_rotary", "seq", "tokens",
               "batch_tokens", "seed", "data", "val_data", "tokenizer", "eval_tokens", "lr", "min_lr_ratio", "warmup", "wd", "clip")


def int_like(s):   # accept 3e9-style token counts
    return int(float(s))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="", help="configs/<preset>.json")
    for k, v in {**MODEL_DEFAULTS, **TRAIN_DEFAULTS}.items():
        ap.add_argument(f"--{k}", type=int_like if k in ("tokens", "eval_tokens") else type(v), default=v)
    known, _ = ap.parse_known_args(argv)
    if known.preset:
        p = load_preset(known.preset)
        ap.set_defaults(**p["model"], **p["train"], compute=p["compute"], run_dir=str(ROOT / "runs" / known.preset))
    a = ap.parse_args(argv)
    if not a.run_dir:
        ap.error("--run_dir is required without --preset")
    for name in ("layers", "hidden", "seq", "batch_tokens", "micro_bs", "log_every", "eval_every"):
        if getattr(a, name) <= 0:
            ap.error(f"--{name} must be positive")
    if a.tokens < 0 or a.max_steps < 0 or a.ckpt_every < 0:
        ap.error("token and step limits must be nonnegative")
    if not a.tokens and not a.compute:
        ap.error("give --tokens or --compute (or a --preset)")
    if a.eval_tokens < a.seq + 2:
        ap.error("--eval_tokens must be at least seq + 2")
    return a


class LossModel(torch.nn.Module):
    """Fused-linear CE over the backbone + router load-balancing loss (zero for dense, experts=0); DDP wraps this whole computation.

    It bypasses Qwen3_5MoeForCausalLM.forward() to avoid materialising logits; change it together with
    the output head or the routing auxiliary loss."""

    def __init__(self, lm, experts, topk):
        super().__init__()
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        self.lm, self.experts, self.topk = lm, experts, topk
        self.ce = LigerFusedLinearCrossEntropyLoss()

    def forward(self, x, y):
        out = self.lm.model(input_ids=x, output_router_logits=bool(self.experts))
        h = out.last_hidden_state
        ce = self.ce(self.lm.lm_head.weight, h.reshape(-1, h.size(-1)), y.reshape(-1))
        if not self.experts:
            return ce, ce.new_zeros(())
        return ce, M.load_balancing_loss_func(out.router_logits, self.experts, self.topk)


def gdn_kernel():
    """Transformers swaps in fla's chunk_gated_delta_rule when fla is installed; the torch fallback is far slower."""
    impl = [c.cell_contents.__module__ for c in (M.torch_chunk_gated_delta_rule.__closure__ or [])
            if callable(c.cell_contents) and c.cell_contents.__module__.startswith("fla")]
    if not impl:
        raise RuntimeError("chunk_gated_delta_rule is not using the fla kernel")
    return impl


def sample_batch(rng, src, bs, seq, dev):
    ix = rng.integers(0, len(src) - seq - 1, bs)
    x = torch.from_numpy(np.stack([src[i:i + seq].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([src[i + 1:i + 1 + seq].astype(np.int64) for i in ix]))
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)


def lr_schedule(a, steps):
    def lr_at(s):   # linear warmup, cosine decay to min_lr_ratio × lr
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        pr = (s - a.warmup) / max(1, steps - a.warmup)
        return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * pr)))
    return lr_at


@torch.no_grad()
def evaluate(model, eval_data, seq, dev):
    """Mean CE over len(eval_data) // seq fixed-seed windows of the validation tokens."""
    model.eval(); tot = 0.0; n = 0
    erng = np.random.default_rng(1234)
    for _ in range(len(eval_data) // seq):
        x, y = sample_batch(erng, eval_data, 1, seq, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
        fl = logits.view(-1, logits.size(-1)); yl = y.view(-1); chunk = max(256, (8192 * 32768) // logits.size(-1))
        for o in range(0, yl.numel(), chunk):
            tot += torch.nn.functional.cross_entropy(fl[o:o + chunk].float(), yl[o:o + chunk], reduction="sum").item()
        n += y.numel()
    model.train(); return tot / n


def main(argv=None):
    a = parse_args(argv)
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

    # Tokenizer and model.
    tok = load_tokenizer(a.tokenizer)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    tok_sha = tokenizer_sha256(a.tokenizer)
    log0(f"Tokenizer: {a.tokenizer}, vocab={len(tok)}, bos={tok.bos_token_id}, eos={tok.eos_token_id}, pad={tok.pad_token_id}")
    cfg = model_config(vars(a), len(tok), a.seq, tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
    model = Qwen3_5MoeForCausalLM(cfg).to(dev, dtype=torch.float32)
    log0("Model source:", M.__file__)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "embeddings are not tied"
    log0("GDN kernel:", gdn_kernel())
    n_all = sum(p.numel() for p in model.parameters()); n_total_nv = n_all - model.get_input_embeddings().weight.numel()
    n_active_nv = n_total_nv - (a.layers * (a.experts - a.topk) * 3 * a.hidden * a.moe_ffn if a.experts else 0)
    fc = count_flops(cfg.to_dict(), a.seq); flops_per_token = fc["train_flops_per_token"]
    assert (fc["total_nonvocab_params"], fc["active_nonvocab_params"]) == (n_total_nv, n_active_nv), "flops.py param count drifted from the model"
    if not a.tokens:
        a.tokens = budget_tokens(flops_per_token, a.compute, a.batch_tokens)
    if a.tokens < a.batch_tokens:
        raise ValueError("--tokens must cover at least one global batch")
    log0(f"层型 {cfg.layer_types}\n参数 总={n_all/1e6:.2f}M 非词表 总={n_total_nv/1e6:.2f}M 激活={n_active_nv/1e6:.3f}M (每 token) | "
         f"每token训练 {flops_per_token/1e6:.1f} MFLOPs {fc['breakdown']} | 预算 {a.tokens:.3e} token ≈ {flops_per_token*a.tokens:.3e} FLOPs")

    # Data: validate on the first eval_tokens of an independent held-out corpus, or else on the tail of train.bin.
    data = open_tokens(a.data, tok_sha)
    if a.val_data:
        train_data, eval_data = data, open_tokens(a.val_data, tok_sha)[:a.eval_tokens]
    else:
        train_data, eval_data = data[:-a.eval_tokens], data[-a.eval_tokens:]
    if len(train_data) < a.seq + 2 or len(eval_data) < a.seq + 2:
        raise ValueError("Not enough train or validation tokens for one sequence")
    log0(f"数据 训练 {len(train_data)/1e9:.2f}B token, 验证 {len(eval_data)/1e6:.1f}M token ({a.val_data or 'train.bin 末尾'}), mmap (world={world})")
    per_micro = a.micro_bs * a.seq * world
    if a.batch_tokens % per_micro:
        raise ValueError(f"batch_tokens {a.batch_tokens} must be divisible by micro_bs×seq×world={per_micro}")
    steps = a.tokens // a.batch_tokens; accum = a.batch_tokens // per_micro
    stop_step = min(steps, a.max_steps) if a.max_steps else steps
    if not a.warmup:
        a.warmup = max(10, round(steps * 0.01))
    rng = np.random.default_rng(a.seed + rank)

    lm = LossModel(model, a.experts, a.topk)
    if ddp:
        lm = torch.nn.parallel.DistributedDataParallel(lm, device_ids=[local_rank], find_unused_parameters=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    lr_at = lr_schedule(a, steps)

    ckpt_path = f"{a.run_dir}/ckpt.pt"; resuming = a.resume and os.path.exists(ckpt_path)
    if resuming:
        previous = json.load(open(f"{a.run_dir}/run_config.json"))
        if previous.get("tokenizer_sha256", tok_sha) != tok_sha:
            raise ValueError("Unsafe resume: tokenizer contents changed; use a new run_dir")
        for key in RESUME_KEYS:
            if previous.get(key) != vars(a)[key]:
                raise ValueError(f"Unsafe resume: {key} changed; use a new run_dir")
    if ddp: dist.barrier()
    if rank == 0:
        json.dump({**vars(a), "tokenizer_sha256": tok_sha, "data_dtype": TOKEN_DTYPE.str, "model_code": source_info(),
                   "n_params": n_all, "n_nonvocab_total": n_total_nv, "n_nonvocab_active": n_active_nv,
                   "flops_per_token": flops_per_token, "flops_breakdown": fc["breakdown"], "steps": steps, "accum": accum,
                   "config": cfg.to_dict()}, open(f"{a.run_dir}/run_config.json", "w"), indent=1, ensure_ascii=False)
    metrics = csv.writer(open(f"{a.run_dir}/training_metrics.csv", "a" if resuming else "w", newline="")) if rank == 0 else None
    if metrics and not resuming:
        metrics.writerow(["step", "tokens", "loss", "aux", "lr", "grad_norm", "tok_per_s", "elapsed_s", "val_loss", "active_nonvocab", "total_nonvocab"])

    model.train(); t0 = time.time(); t_last = t0; tok_seen = 0; start_step = 0
    if resuming:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_step = ck["step"] + 1; tok_seen = ck["tok_seen"]
        if len(ck.get("rank_rng", [])) == world:   # same world size: exact per-rank sample stream
            state = ck["rank_rng"][rank]
            rng.bit_generator.state = state["numpy"]
            torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state(state["cuda"], dev)
        elif rank == 0:
            rng.bit_generator.state = ck["rng"]
        else:
            rng = np.random.default_rng([a.seed + rank, start_step])
        if ck.get("world", 1) != world:
            log0("World size changed: optimizer/token count restored; sample stream is not bitwise identical")
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
        for g in opt.param_groups: g["lr"] = lr_at(step)
        loss_acc = aux_acc = 0.0
        for mi in range(accum):
            x, y = sample_batch(rng, train_data, a.micro_bs, a.seq, dev)
            sync = lm.no_sync() if (ddp and mi < accum - 1) else contextlib.nullcontext()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                ce, aux = lm(x, y)
                loss = (ce + a.aux_coef * aux) / accum
            loss.backward(); loss_acc += ce.item() / accum; aux_acc += float(aux) / accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
        opt.step(); opt.zero_grad(set_to_none=True); tok_seen += accum * per_micro
        if ddp:
            la = torch.tensor(loss_acc, device=dev); dist.all_reduce(la); loss_acc = la.item() / world
        if tb:
            for k, v in (("loss", loss_acc), ("aux", aux_acc), ("lr", lr_at(step)), ("grad_norm", gn), ("tokens", tok_seen)):
                tb.add_scalar(f"train/{k}", v, step)
        if a.ckpt_every and step > 0 and step % a.ckpt_every == 0: save_ckpt(step)
        if (step % a.log_every == 0 or step == stop_step - 1) and rank == 0:
            now = time.time(); tps = min(a.log_every, step - start_step + 1) * accum * per_micro / max(1e-6, now - t_last); t_last = now
            vl = evaluate(model, eval_data, a.seq, dev) if (step % a.eval_every == 0 and step > 0) or step == stop_step - 1 else ""
            metrics.writerow([step, tok_seen, f"{loss_acc:.4f}", f"{aux_acc:.4f}", f"{lr_at(step):.2e}", f"{gn:.2f}", f"{tps:.0f}",
                              f"{now-t0:.0f}", f"{vl:.4f}" if vl != "" else "", n_active_nv, n_total_nv])
            print(f"step {step}/{steps} tok {tok_seen/1e6:.0f}M loss {loss_acc:.4f} aux {aux_acc:.3f} lr {lr_at(step):.2e} gn {gn:.2f} "
                  f"{tps:.0f} tok/s {(now-t0)/60:.1f}min mem {torch.cuda.max_memory_allocated(dev)/2**30:.1f}G"
                  + (f" | val {vl:.4f}" if vl != "" else ""), flush=True)
            if tb:
                tb.add_scalar("perf/tok_per_s", tps, step)
                tb.add_scalar("perf/max_mem_gib", torch.cuda.max_memory_allocated(dev) / 2**30, step)
                if vl != "": tb.add_scalar("val/loss", vl, step)
                tb.flush()

    # Every rank participates; save FP32 master weights before the BF16 export.
    if stop_step > start_step: save_ckpt(stop_step - 1)
    if tb: tb.close()
    if ddp: dist.barrier()
    if rank == 0:
        model.to(torch.bfloat16).save_pretrained(f"{a.run_dir}/hf"); tok.save_pretrained(f"{a.run_dir}/hf")
        json.dump({"tokens_trained": tok_seen, "train_flops": flops_per_token * tok_seen, "active_nonvocab": n_active_nv,
                   "total_nonvocab": n_total_nv, "elapsed_s": time.time() - t0, "world": world,
                   "completed_budget": tok_seen >= steps * a.batch_tokens}, open(f"{a.run_dir}/summary.json", "w"), indent=1)
        json.dump({**fc, "train_tokens": tok_seen, "train_flops": flops_per_token * tok_seen}, open(f"{a.run_dir}/flops.json", "w"), indent=2)
        print("DONE 已导出 HF 目录", f"{a.run_dir}/hf", flush=True)
    if ddp: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
