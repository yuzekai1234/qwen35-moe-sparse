"""Qwen3.5-MoE 参数量及比赛 Megatron 参考口径的 FLOPs。

参考固定提交 b92c3ed70029005fd5769d2cf4c65bc3d061a74b，training.py:821-1407。
前向每 token：全注意力投影包含输出门，causal 核心为 2*s*nq*hd；GDN 投影和
depthwise conv 按实际尺寸计，核心严格采用参考的 8*nv*dv**2；SwiGLU 为三次 GEMM。
另计原函数未列出的实际 MoE router 和共享专家门投影。所有参考项前反向合计乘 3。
参数量统计独立于 FLOPs；norm 等标量参数不再按 GEMM 权重计算。

这是指定的 Megatron 模型 FLOPs 约定，不统计 FLA chunk 的物理 kernel 指令数；
与未修改原函数的差异仅为 router/shared gate，返回值中单列参考值和差异。
独立原函数交叉核对见 scripts/audit_flops.py。改变结构时须同步参数统计。

用法: python -m q35.flops q35_c2_m [--seq 8192] [--tokenizer tokenizer/gzy_50k]   # 预设按实际词表核算预算
      python -m q35.flops runs/q35_c2_m/hf/config.json [--seq 8192] [--tokens 3e9]"""
import argparse
import json

MEGATRON_COMMIT = "b92c3ed70029005fd5769d2cf4c65bc3d061a74b"
ACCOUNTING_METHOD = "megatron_b92c3ed_plus_router_shared_gate_v1"


def count(c, seq):
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
    moe_active = E * h + 3 * h * fs + h + k * 3 * h * fe          # router + 共享专家 + 共享专家门 + top-k 路由专家
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
    expert_gemm = L * 6 * h * (k * fe + fs)
    router = L * 2 * h * E
    shared_gate = L * 2 * h
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
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--tokens", type=float, default=0)
    ap.add_argument("--batch_tokens", type=int, default=524288)
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
