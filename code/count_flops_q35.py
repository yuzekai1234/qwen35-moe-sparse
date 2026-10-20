"""Qwen3.5-MoE 文本模型(Gated DeltaNet 3:1 full attention + 路由专家/共享专家)的每 token FLOPs 与参数核算, 口径对齐 count_flops.py(Megatron num_floating_point_operations)。
用法: count_flops_q35.py --model ../runs/q35_c1_s1/hf --seq 8192 [--tokens 2.9e9] | [--budget 3e17 --batch_tokens 524288 反推 token 数]
口径(前向+反向 = 3 × 前向; 前向每 token):
  稠密 GEMM: 2 × 激活非词表参数 (每层 norm/投影/conv/router/共享专家/门 + top-k 个路由专家; 即 "6×激活非词表参数" 的 6N 项)
  全注意力层分数: 2·s·(heads·head_dim)  (同 count_flops.py: Megatron 不打因果对折, 与 FMA×2 抵消)
  线性注意力层(chunked GDN, chunk C=64, 按 fla chunk_gated_delta_rule 的实际 matmul, 每个 v 头):
     状态项 6·dk·dv  (W·S 求 v_new、Q·S 读出、K^T·v_new 更新状态, 各 2·dk·dv)
     块内项 6·C·dk + 4·C·dv  (KK^T、QK^T、A·(βK) 各 2·C·dk; A·(βV)、块内注意力×v_new 各 2·C·dv)
  词表投影: 2·h·V   (嵌入查表不计)"""
import argparse, json

CHUNK = 64

def count(c, seq):
    c = c.get("text_config", c)
    L, h, V = c["num_hidden_layers"], c["hidden_size"], c["vocab_size"]
    E, k, fe, fs = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
    nq, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    nk, nv, dk, dv, K = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"]
    types = c["layer_types"][:L]
    moe_active = E * h + 3 * h * fs + h + k * 3 * h * fe          # router + 共享专家 + 共享专家门 + top-k 路由专家
    moe_total = E * h + 3 * h * fs + h + E * 3 * h * fe
    gdn = h * (2 * nk * dk + nv * dv) + h * nv * dv + 2 * h * nv + (2 * nk * dk + nv * dv) * K + 2 * nv + dv + nv * dv * h
    full = h * 2 * nq * hd + 2 * h * nkv * hd + nq * hd * h + 2 * hd   # q 含输出门(×2), q/k norm
    n_lin, n_full = types.count("linear_attention"), types.count("full_attention")
    mix = n_lin * gdn + n_full * full + L * 2 * h + h              # + 每层两个 RMSNorm + 末层 norm
    active, total = mix + L * moe_active, mix + L * moe_total
    gdn_core = n_lin * nv * (6 * dk * dv + 6 * CHUNK * dk + 4 * CHUNK * dv)
    attn_score = n_full * 2 * seq * nq * hd
    logits = 2 * h * V
    fwd = 2 * active + gdn_core + attn_score + logits
    return dict(layers=L, hidden=h, layer_types=types, vocab=V, seq=seq, experts=E, topk=k, expert_ffn=fe, shared_ffn=fs,
                attn_heads=nq, kv_heads=nkv, head_dim=hd, lin_k_heads=nk, lin_v_heads=nv,
                active_nonvocab_params=active, total_nonvocab_params=total, nonvocab_params=active, embed_params=h * V,
                fwd_flops_per_token=fwd, train_flops_per_token=3 * fwd,
                breakdown={"dense_2Na": 2 * active, "gdn_core": gdn_core, "attn_score": attn_score, "logits": logits})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--tokens", type=float, default=0); ap.add_argument("--budget", type=float, default=0); ap.add_argument("--batch_tokens", type=int, default=524288)
    a = ap.parse_args(); c = json.load(open(a.model if a.model.endswith(".json") else f"{a.model}/config.json"))
    res = count(c, a.seq)
    if a.budget:
        steps = round(a.budget / res["train_flops_per_token"] / a.batch_tokens); res["steps"] = steps; a.tokens = steps * a.batch_tokens
    if a.tokens: res["train_tokens"] = a.tokens; res["train_flops"] = res["train_flops_per_token"] * a.tokens
    print(json.dumps(res, indent=1))
