"""Qwen3.5-MoE 文本模型(Qwen3.5-35B-A3B text_config 结构缩放)的预训练闭环, 由 train.py 改来: DDP/no_sync/ckpt/resume/cosine/token 预算/HF 导出不变。
结构: layer_types 每 4 层 3 层 linear_attention(Gated DeltaNet, conv 4, fla 快速 kernel) + 1 层 full_attention(GQA + 输出门, partial_rotary 0.25, rope_theta 1e7);
      MoE = E 个路由专家选 k + 1 个共享专家(带 sigmoid 门), loss 加 router_aux_loss_coef × load balancing loss; 无 MTP、无视觉部分; 绑定嵌入。
FLOPs/参数口径见 count_flops_q35.py (激活非词表参数 = 总非词表 − 未选中的路由专家)。
用法: train_q35.py --run_dir ../runs/q35_c1_s1 --data ../data/corpus_exaone154k/train.bin --tokenizer ../tokenizer/exaone_154k --layers 5 --hidden 96 --heads 2 --kv_heads 1 \
        --lin_k_heads 1 --lin_v_heads 2 --experts 64 --topk 4 --moe_ffn 96 --shared_ffn 48 --tokens 2.9e9 --batch_tokens 524288 --micro_bs 8 --lr 3.6e-3"""
import contextlib, argparse, os, time, math, json, csv, numpy as np, torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import PreTrainedTokenizerFast
from models.qwen3_5_moe import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM, source_info
from models.qwen3_5_moe import modeling_qwen3_5_moe as M
from count_flops_q35 import count as count_flops
ap = argparse.ArgumentParser()
for k, v in dict(run_dir="../runs/q35", data="../data/corpus_exaone154k/train.bin", tokenizer="../tokenizer/exaone_154k", layers=5, hidden=96, heads=2, kv_heads=1, head_dim=64,
                 lin_k_heads=1, lin_v_heads=2, lin_head_dim=64, experts=64, topk=4, moe_ffn=96, shared_ffn=48, layer_types="", aux_coef=0.001, rope_theta=1e7, partial_rotary=0.25,
                 experts_impl="eager", seq=8192, tokens=300e6, batch_tokens=524288, micro_bs=8, lr=3.6e-3, min_lr_ratio=0.1, warmup=0, wd=0.1, clip=1.0, seed=0,
                 log_every=10, eval_every=500, eval_tokens=2e6, gpu=0, ckpt_every=200, resume=1, max_steps=0).items():
    ap.add_argument(f"--{k}", type=type(v), default=v)
a = ap.parse_args()
for name in ("layers", "hidden", "seq", "tokens", "batch_tokens", "micro_bs", "log_every", "eval_every"):
    if getattr(a, name) <= 0: ap.error(f"--{name} must be positive")
if a.max_steps < 0 or a.ckpt_every < 0: ap.error("step limits must be nonnegative")
if a.eval_tokens < a.seq + 2: ap.error("--eval_tokens must be at least seq + 2")
if a.tokens < a.batch_tokens: ap.error("--tokens must cover at least one global batch")
os.makedirs(a.run_dir, exist_ok=True)
import torch.distributed as dist
DDP_ON = int(os.environ.get("WORLD_SIZE", "1")) > 1
if DDP_ON:
    a.gpu = int(os.environ["LOCAL_RANK"])
    if a.gpu >= torch.cuda.device_count(): raise ValueError("DDP ranks exceed visible GPUs")
    torch.cuda.set_device(a.gpu)
    dist.init_process_group("nccl"); RANK = dist.get_rank(); WORLD = dist.get_world_size()
else: RANK, WORLD = 0, 1
torch.manual_seed(a.seed + RANK)
dev = torch.device("cuda", a.gpu); torch.cuda.set_device(dev)
def log0(*x, **k):
    if RANK == 0: print(*x, flush=True)
tok = PreTrainedTokenizerFast.from_pretrained(a.tokenizer, local_files_only=True)
if len(tok) != 153600: raise ValueError(f"Expected EXAONE 153600 vocabulary; got {len(tok)}")
# 层型: 每 4 层 [lin,lin,lin,full]; 层数非 4 的倍数时余下的层接 lin (5 层 = [lin,lin,lin,full,lin])
types = a.layer_types.split(",") if a.layer_types else [("full_attention" if i % 4 == 3 else "linear_attention") for i in range(a.layers)]
cfg = Qwen3_5MoeTextConfig(vocab_size=len(tok), hidden_size=a.hidden, num_hidden_layers=a.layers, num_attention_heads=a.heads, num_key_value_heads=a.kv_heads, head_dim=a.head_dim,
                           linear_num_key_heads=a.lin_k_heads, linear_num_value_heads=a.lin_v_heads, linear_key_head_dim=a.lin_head_dim, linear_value_head_dim=a.lin_head_dim, linear_conv_kernel_dim=4,
                           num_experts=a.experts, num_experts_per_tok=a.topk, moe_intermediate_size=a.moe_ffn, shared_expert_intermediate_size=a.shared_ffn,
                           router_aux_loss_coef=a.aux_coef, output_router_logits=False, layer_types=types, max_position_embeddings=a.seq, rms_norm_eps=1e-6,
                           rope_parameters={"rope_type": "default", "rope_theta": a.rope_theta, "partial_rotary_factor": a.partial_rotary}, partial_rotary_factor=a.partial_rotary,
                           tie_word_embeddings=True, attention_bias=False, initializer_range=0.02, bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
cfg.use_cache = False
cfg._experts_implementation = a.experts_impl
cfg._attn_implementation = "sdpa"
model = Qwen3_5MoeForCausalLM(cfg).to(dev, dtype=torch.float32)
log0("Model source:", M.__file__)
assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "嵌入未绑定"
# fla 快速 kernel 检查: transformers 在装了 fla 时把 torch_chunk_gated_delta_rule 换成 fla 实现
_impl = [c.cell_contents.__module__ for c in (M.torch_chunk_gated_delta_rule.__closure__ or []) if callable(c.cell_contents) and c.cell_contents.__module__.startswith("fla")]
log0("GDN kernel:", _impl or "torch 慢路径!"); assert _impl, "chunk_gated_delta_rule 没走 fla"
n_all = sum(p.numel() for p in model.parameters()); n_total_nv = n_all - model.get_input_embeddings().weight.numel()
n_expert_each = 3 * a.hidden * a.moe_ffn; n_active_nv = n_total_nv - a.layers * (a.experts - a.topk) * n_expert_each
FC = count_flops(cfg.to_dict(), a.seq); flops_per_token = FC["train_flops_per_token"]
assert FC["total_nonvocab_params"] == n_total_nv and FC["active_nonvocab_params"] == n_active_nv, (FC["total_nonvocab_params"], n_total_nv, FC["active_nonvocab_params"], n_active_nv)
log0(f"层型 {types}\n参数 总={n_all/1e6:.2f}M 非词表 总={n_total_nv/1e6:.2f}M 激活={n_active_nv/1e6:.3f}M (每 token) | 每token训练 {flops_per_token/1e6:.1f} MFLOPs {FC['breakdown']} | 预算 {a.tokens:.3e} token ≈ {flops_per_token*a.tokens:.3e} FLOPs")
DT = np.uint32 if len(tok) > 65535 else np.uint16
if os.path.getsize(a.data) % np.dtype(DT).itemsize: raise ValueError("Misaligned binary token file")
data = np.memmap(a.data, dtype=DT, mode="r")
n_tok_data = len(data); log0(f"数据 {n_tok_data/1e9:.2f}B token mmap (world={WORLD})")
eval_n = int(a.eval_tokens)
if n_tok_data - eval_n < a.seq + 2: raise ValueError("Not enough train tokens after holding out eval_tokens")
train_data = data[:-eval_n]; eval_data = data[-eval_n:]
per_micro = a.micro_bs * a.seq * WORLD
assert a.batch_tokens % per_micro == 0, f"batch_tokens {a.batch_tokens} 不能被 micro_bs×seq×world={per_micro} 整除(accum 取整会少训 token)"
steps = int(a.tokens // a.batch_tokens); accum = a.batch_tokens // per_micro; rng = np.random.default_rng(a.seed + RANK)
stop_step = min(steps, a.max_steps) if a.max_steps else steps
if not a.warmup: a.warmup = max(10, round(steps * 0.01))
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
class LossModel(torch.nn.Module):   # 包一层让 DDP 包住整个 loss 计算(含 router aux loss)
    def __init__(self, lm):
        super().__init__(); self.lm = lm; self.ce = LigerFusedLinearCrossEntropyLoss()
    def forward(self, x, y):
        out = self.lm.model(input_ids=x, output_router_logits=True)
        h = out.last_hidden_state
        ce = self.ce(self.lm.lm_head.weight, h.reshape(-1, h.size(-1)), y.reshape(-1))
        aux = M.load_balancing_loss_func(out.router_logits, a.experts, a.topk)
        return ce, aux
lm = LossModel(model)
if DDP_ON: lm = torch.nn.parallel.DistributedDataParallel(lm, device_ids=[a.gpu], find_unused_parameters=True)
raw_model = model
def batch(src, bs):
    ix = rng.integers(0, len(src) - a.seq - 1, bs)
    x = torch.from_numpy(np.stack([src[i:i+a.seq].astype(np.int64) for i in ix])); y = torch.from_numpy(np.stack([src[i+1:i+1+a.seq].astype(np.int64) for i in ix]))
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
decay, no_decay = [p for n, p in raw_model.named_parameters() if p.dim() >= 2], [p for n, p in raw_model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
def lr_at(s):
    if s < a.warmup: return a.lr * (s + 1) / a.warmup
    pr = (s - a.warmup) / max(1, steps - a.warmup); return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * pr)))
@torch.no_grad()
def evaluate():
    raw_model.eval(); tot = 0.0; n = 0
    erng = np.random.default_rng(1234)
    for _ in range(eval_n // a.seq):
        ix = erng.integers(0, len(eval_data) - a.seq - 1, 1)
        x = torch.from_numpy(np.stack([eval_data[i:i+a.seq].astype(np.int64) for i in ix])).to(dev); y = torch.from_numpy(np.stack([eval_data[i+1:i+1+a.seq].astype(np.int64) for i in ix])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = raw_model(input_ids=x).logits
        fl = logits.view(-1, logits.size(-1)); yl = y.view(-1); CH = max(256, (8192 * 32768) // logits.size(-1))
        for o in range(0, yl.numel(), CH): tot += torch.nn.functional.cross_entropy(fl[o:o+CH].float(), yl[o:o+CH], reduction="sum").item()
        n += y.numel()
    raw_model.train(); return tot / n
CKPT = f"{a.run_dir}/ckpt.pt"; resuming = a.resume and os.path.exists(CKPT)
if resuming:
    previous = json.load(open(f"{a.run_dir}/run_config.json"))
    for key in ("layers", "hidden", "heads", "kv_heads", "head_dim", "lin_k_heads", "lin_v_heads", "lin_head_dim", "experts", "topk", "moe_ffn", "shared_ffn", "layer_types", "aux_coef", "rope_theta", "partial_rotary", "seq", "tokens", "batch_tokens", "seed", "data", "tokenizer", "eval_tokens", "lr", "min_lr_ratio", "warmup", "wd", "clip"):
        if previous[key] != vars(a)[key]: raise ValueError(f"Unsafe resume: {key} changed; use a new run_dir")
if DDP_ON: dist.barrier()
if RANK == 0: json.dump({**vars(a), "model_code": source_info(), "n_params": n_all, "n_nonvocab_total": n_total_nv, "n_nonvocab_active": n_active_nv, "flops_per_token": flops_per_token, "flops_breakdown": FC["breakdown"],
                         "steps": steps, "accum": accum, "config": cfg.to_dict()}, open(f"{a.run_dir}/run_config.json", "w"), indent=1, ensure_ascii=False)
log = csv.writer(open(f"{a.run_dir}/training_metrics.csv", "a" if resuming else "w", newline="")) if RANK == 0 else None
if log and not resuming: log.writerow(["step", "tokens", "loss", "aux", "lr", "grad_norm", "tok_per_s", "elapsed_s", "val_loss", "active_nonvocab", "total_nonvocab"])
model.train(); t0 = time.time(); t_last = t0; tok_seen = 0; start_step = 0
if resuming:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    raw_model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_step = ck["step"] + 1; tok_seen = ck["tok_seen"]
    if len(ck.get("rank_rng", [])) == WORLD:
        state = ck["rank_rng"][RANK]
        rng.bit_generator.state = state["numpy"]
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], dev)
    elif RANK == 0:
        rng.bit_generator.state = ck["rng"]
    else:
        rng = np.random.default_rng([a.seed + RANK, start_step])
    if ck.get("world", 1) != WORLD: log0("World size changed: optimizer/token count restored; sample stream is not bitwise identical")
    if os.environ.get("Q35_VERIFY_RESUME") == "1":
        def exact_state(x, y):
            if isinstance(x, torch.Tensor): return isinstance(y, torch.Tensor) and torch.equal(x.cpu(), y.cpu())
            if isinstance(x, dict): return x.keys() == y.keys() and all(exact_state(x[k], y[k]) for k in x)
            if isinstance(x, (tuple, list)): return len(x) == len(y) and all(exact_state(u, v) for u, v in zip(x, y))
            return x == y
        checks = {"rank": RANK, "model_exact": exact_state(raw_model.state_dict(), ck["model"]),
                  "optimizer_exact": exact_state(opt.state_dict(), ck["opt"])}
        if len(ck.get("rank_rng", [])) == WORLD:
            checks["rng_exact"] = exact_state({"numpy": rng.bit_generator.state, "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(dev)}, ck["rank_rng"][RANK])
        assert all(v for k, v in checks.items() if k != "rank"), checks
        verified = [None] * WORLD
        if DDP_ON: dist.all_gather_object(verified, checks)
        else: verified[0] = checks
        if RANK == 0: json.dump(verified, open(f"{a.run_dir}/resume_verification.json", "w"), indent=2)
    log0(f"从 {CKPT} 恢复：step {start_step}, tok_seen {tok_seen}"); del ck
def save_ckpt(step):
    state = {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(dev)}
    states = [None] * WORLD
    if DDP_ON: dist.all_gather_object(states, state)
    else: states[0] = state
    if RANK == 0:
        torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(), "step": step, "tok_seen": tok_seen, "rng": rng.bit_generator.state, "rank_rng": states, "world": WORLD}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)
log0(f"steps {steps} accum {accum} micro_bs {a.micro_bs} warmup {a.warmup} lr {a.lr}")
for step in range(start_step, stop_step):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    loss_acc = 0.0; aux_acc = 0.0
    for _mi in range(accum):
        x, y = batch(train_data, a.micro_bs)
        _ctx = lm.no_sync() if (DDP_ON and _mi < accum - 1) else contextlib.nullcontext()
        with _ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            ce, aux = lm(x, y)
            loss = (ce + a.aux_coef * aux) / accum
        loss.backward(); loss_acc += ce.item() / accum; aux_acc += float(aux) / accum
    gn = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), a.clip).item(); opt.step(); opt.zero_grad(set_to_none=True); tok_seen += accum * per_micro
    if DDP_ON:
        la = torch.tensor(loss_acc, device=dev); dist.all_reduce(la); loss_acc = la.item() / WORLD   # SUM/WORLD 等价 AVG, gloo 也能用
    if a.ckpt_every and step > 0 and step % a.ckpt_every == 0: save_ckpt(step)
    if (step % a.log_every == 0 or step == stop_step - 1) and RANK == 0:
        now = time.time(); tps = min(a.log_every, step - start_step + 1) * accum * per_micro / max(1e-6, now - t_last); t_last = now
        vl = evaluate() if (step % a.eval_every == 0 and step > 0) or step == stop_step - 1 else ""
        log.writerow([step, tok_seen, f"{loss_acc:.4f}", f"{aux_acc:.4f}", f"{lr_at(step):.2e}", f"{gn:.2f}", f"{tps:.0f}", f"{now-t0:.0f}", f"{vl:.4f}" if vl != "" else "", n_active_nv, n_total_nv])
        print(f"step {step}/{steps} tok {tok_seen/1e6:.0f}M loss {loss_acc:.4f} aux {aux_acc:.3f} lr {lr_at(step):.2e} gn {gn:.2f} {tps:.0f} tok/s {(now-t0)/60:.1f}min mem {torch.cuda.max_memory_allocated(dev)/2**30:.1f}G"
              + (f" | val {vl:.4f}" if vl != "" else ""), flush=True)
# Every rank participates; save FP32 master weights before BF16 export.
if stop_step > start_step: save_ckpt(stop_step - 1)
if DDP_ON: dist.barrier()
if RANK == 0:
    raw_model.to(torch.bfloat16).save_pretrained(f"{a.run_dir}/hf"); tok.save_pretrained(f"{a.run_dir}/hf")
    json.dump({"tokens_trained": tok_seen, "train_flops": flops_per_token * tok_seen, "active_nonvocab": n_active_nv, "total_nonvocab": n_total_nv, "elapsed_s": time.time() - t0, "world": WORLD, "completed_budget": tok_seen >= steps * a.batch_tokens},
              open(f"{a.run_dir}/summary.json", "w"), indent=1)
    print("DONE 已导出 HF 目录", f"{a.run_dir}/hf", flush=True)
if DDP_ON: dist.barrier(); dist.destroy_process_group()
