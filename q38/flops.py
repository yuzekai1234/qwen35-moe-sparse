"""Q38 sparse parameter and compute accounting.

Inherited GDN terms use the contest-pinned Megatron convention. Q38 adds
sparse QSA, indexer/KL, mHC, N-gram gates and training-only MTP.
Usage: python -m q38.flops q38_c2 --seq 4096
"""
import argparse
import json

MEGATRON_COMMIT = "b92c3ed70029005fd5769d2cf4c65bc3d061a74b"
ACCOUNTING_METHOD = "megatron_b92c3ed_plus_router_shared_gate_v1"


def _count_backbone(c, seq):
    c = c.get("text_config", c)
    L, h, V = c["num_hidden_layers"], c["hidden_size"], c["vocab_size"]
    E, k, fe, fs = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
    nq, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    nk, nv, dk, dv, K = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"]
    types = c["layer_types"][:L]
    if seq <= 0 or len(types) != L or any(t not in ("linear_attention", "full_attention") for t in types):
        raise ValueError("Positive sequence length and one supported layer type per layer are required")
    if c.get("attention_bias", False):
        raise ValueError("This accounting supports the bias-free Qwen attention projections")
    dense_width = c.get("intermediate_size", 0)
    if E == 0:
        if dense_width <= 0 or any((k, fe, fs)):
            raise ValueError("Dense FFN requires a positive width and zero expert fields")
        moe_active = moe_total = 3 * h * dense_width
    else:
        moe_active = E * h + 3 * h * fs + h + k * 3 * h * fe
        moe_total = E * h + 3 * h * fs + h + E * 3 * h * fe
    gdn = h * (2 * nk * dk + nv * dv) + h * nv * dv + 2 * h * nv + (2 * nk * dk + nv * dv) * K + 2 * nv + dv + nv * dv * h
    full = h * 2 * nq * hd + 2 * h * nkv * hd + nq * hd * h + 2 * hd   # q 含输出门(×2), q/k norm
    n_lin, n_full = types.count("linear_attention"), types.count("full_attention")
    mix = n_lin * gdn + n_full * full + L * 2 * h + h              # + 每层两个 RMSNorm + 末层 norm
    active, total = mix + L * moe_active, mix + L * moe_total
    # Operation counts are independent of norm/decay parameters counted above.
    full_proj = n_full * 2 * h * (3 * nq * hd + 2 * nkv * hd)
    gdn_proj = n_lin * 2 * h * (2 * nk * dk + 3 * nv * dv + 2 * nv)
    gdn_conv = n_lin * 2 * K * (2 * nk * dk + nv * dv)
    # Preserve the pinned reference's dv**2 convention, including when dk != dv.
    gdn_core = n_lin * 8 * nv * dv**2
    expert_gemm = L * 6 * h * (k * fe + fs if E else dense_width)
    router = L * 2 * h * E
    shared_gate = L * 2 * h if E else 0
    attn_score = n_full * 2 * seq * nq * hd
    logits = 2 * h * V
    breakdown = dict(full_projections=full_proj, gdn_projections=gdn_proj, gdn_conv=gdn_conv,
                     gdn_core=gdn_core, expert_gemm=expert_gemm, router=router,
                     shared_gate=shared_gate, attn_score=attn_score, logits=logits)
    fwd = sum(breakdown.values())
    reference_fwd = fwd - router - shared_gate
    return dict(layers=L, hidden=h, layer_types=types, vocab=V, seq=seq, experts=E, topk=k, expert_ffn=fe, shared_ffn=fs,
                attn_heads=nq, kv_heads=nkv, head_dim=hd, lin_k_heads=nk, lin_v_heads=nv,
                active_nonvocab_params=active, total_nonvocab_params=total, nonvocab_params=active, embed_params=h * V,
                accounting_method=ACCOUNTING_METHOD, megatron_commit=MEGATRON_COMMIT,
                fwd_flops_per_token=fwd, train_flops_per_token=3 * fwd,
                megatron_reference_train_flops_per_token=3 * reference_fwd,
                extra_router_shared_gate_train_flops_per_token=3 * (router + shared_gate),
                breakdown=breakdown)


def count(c, seq):
    """Q38 budget: sparse attention, indexer, mHC, N-gram gate and training-only MTP.

    Total non-vocabulary parameters include N-gram tables and MTP. Active parameters
    describe the evaluation main head: two accessed hash rows, no training-only MTP.
    Dense warmup and optimizer operations are excluded, matching the reference.
    """
    c = c.get("text_config", c)
    r = _count_backbone(c, seq)
    h, L = c["hidden_size"], c["num_hidden_layers"]
    nf = r["layer_types"].count("full_attention")
    n, rank = c.get("n_branch", 4), max(1, round(h / 8))
    ni, di = c.get("indexer_heads", 4), c.get("indexer_head_dim", 64)
    rows, budget = c.get("ngram_rows", 0), c.get("qsa_blocks", 8)
    if rows < 0 or budget < 1 or seq < 1:
        raise ValueError("Invalid Q38 N-gram rows, sparse block budget or sequence length")
    blocks = (seq + 63) // 64
    avg = sum(64 * (min(i + 1, budget) - 1) + 32.5 for i in range(blocks)) / blocks
    bd = dict(r["breakdown"])
    bd["attn_score"] = round(nf * 4 * c["num_attention_heads"] * c["head_dim"] * avg)
    ip = nf * h * (ni * di + di + ni)
    mp = 2 * L * (n*h*rank + rank*(2*n+n*n) + 2*n+n*n+3)
    gp, tables = (h*h + 2*h, 2*rows*h) if rows else (0, 0)
    bd.update(indexer_projections=2*ip, indexer_score=nf*2*seq*ni*di,
              mhc_projections=2*2*L*(n*h*rank + rank*(2*n+n*n)),
              mhc_mix=2*L*2*h*(2*n+n*n), ngram_gate=2*h*h if rows else 0)
    mtp_params = mtp_active = 0
    if c.get("mtp", True):
        mc = dict(c, num_hidden_layers=1, layer_types=["linear_attention"])
        mr = _count_backbone(mc, seq)
        mtp_params = mr["total_nonvocab_params"] + 2*h*h + 2*h
        mtp_active = mr["active_nonvocab_params"] + 2*h*h + 2*h
        bd["mtp"] = mr["fwd_flops_per_token"] + 4*h*h
    distill = nf * 2 * seq * c["num_attention_heads"] * c["head_dim"]
    active = r["active_nonvocab_params"] + ip + mp + gp + (2*h if rows else 0)
    total = r["total_nonvocab_params"] + ip + mp + gp + tables + mtp_params
    fwd = sum(bd.values())
    # Indexer inputs are detached: projections need only forward and weight
    # gradients (2x), while the score GEMMs still backpropagate to both operands.
    train = 3*fwd - bd["indexer_projections"] + distill
    r.update(accounting_method="q38_sparse_mhc_ngram_mtp_v2" if c["num_experts"] else "q38_dense_mhc_ngram_mtp_v1", breakdown=bd,
             active_nonvocab_params=active, nonvocab_params=active, total_nonvocab_params=total,
             total_params_all=total+r["embed_params"], ngram_params=tables, ngram_rows=rows,
             indexer_params=ip, mhc_params=mp, mtp_params=mtp_params,
             mtp_active_params=mtp_active, distill_target_flops=distill,
             fwd_flops_per_token=fwd, main_fwd_flops_per_token=fwd-bd.get("mtp", 0),
             train_flops_per_token=train)
    # Keep the historical nonvocab fields for training/checkpoint accounting;
    # evaluation excludes every Embedding, including N-gram tables/rows.
    r.update(active_params_excluding_embeddings=active-(2*h if rows else 0),
             total_params_excluding_embeddings=total-tables)
    return r


def validate_resume_accounting(previous, current):
    """Do not silently mix accounting conventions in one checkpoint history."""
    if (previous.get("flops_per_token") != current["train_flops_per_token"]
            or previous.get("flops_accounting", current["accounting_method"]) != current["accounting_method"]):
        raise ValueError("Unsafe resume: FLOPs accounting changed; use a new run_dir")


def main():
    from .data import DEFAULT_TOKENIZER
    from .presets import preset_budget, preset_names
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="preset name or config.json")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--tokens", type=float, default=0)
    ap.add_argument("--batch_tokens", type=int, default=None)
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    a = ap.parse_args()
    if a.target in preset_names():
        from tokenizers import Tokenizer
        vocab = Tokenizer.from_file(f"{a.tokenizer}/tokenizer.json").get_vocab_size()
        tokens, res = preset_budget(a.target, vocab, a.seq, a.batch_tokens)
        res["budget_tokens"] = tokens
        a.tokens = a.tokens or tokens
    else:
        res = count(json.load(open(a.target)), a.seq)
    if a.tokens:
        res["train_tokens"] = a.tokens
        res["train_flops"] = res["train_flops_per_token"] * a.tokens
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
