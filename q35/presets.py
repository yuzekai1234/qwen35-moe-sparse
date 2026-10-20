"""Presets (configs/<name>.json) and model-config construction.

A preset holds the model shape, preset-specific training defaults, the compute
budget and the historical reference numbers.  Everything else uses the shared
defaults below, which reproduce the original scaling rules:
  每 4 层 [lin, lin, lin, full]，层数非 4 的倍数时余下接 lin；64 选 4 路由专家 + 共享专家；
  全注意力 head_dim 64、输出门、partial_rotary 0.25、rope_theta 1e7；绑定嵌入。
"""
import json
from pathlib import Path

from .flops import count
from .models.qwen3_5_moe import Qwen3_5MoeTextConfig

ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "configs"

# Model-shape arguments shared by train.py and the presets.
MODEL_DEFAULTS = dict(layers=5, hidden=96, heads=2, kv_heads=1, head_dim=64, lin_k_heads=1, lin_v_heads=2,
                      lin_head_dim=64, experts=64, topk=4, moe_ffn=96, shared_ffn=48, layer_types="",
                      aux_coef=0.001, rope_theta=1e7, partial_rotary=0.25, experts_impl="eager")


def preset_names():
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def load_preset(name):
    path = PRESET_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown preset {name!r}; available: {', '.join(preset_names())}")
    return json.loads(path.read_text())


def default_layer_types(layers):
    return ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)]


def model_config(shape, vocab_size, seq, bos_token_id=None, eos_token_id=None, pad_token_id=None):
    """Build the text config from shape fields (a dict such as vars(args) or MODEL_DEFAULTS | preset)."""
    s = {**MODEL_DEFAULTS, **shape}
    types = s["layer_types"].split(",") if s["layer_types"] else default_layer_types(s["layers"])
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=vocab_size, hidden_size=s["hidden"], num_hidden_layers=s["layers"],
        num_attention_heads=s["heads"], num_key_value_heads=s["kv_heads"], head_dim=s["head_dim"],
        linear_num_key_heads=s["lin_k_heads"], linear_num_value_heads=s["lin_v_heads"],
        linear_key_head_dim=s["lin_head_dim"], linear_value_head_dim=s["lin_head_dim"], linear_conv_kernel_dim=4,
        num_experts=s["experts"], num_experts_per_tok=s["topk"], moe_intermediate_size=s["moe_ffn"],
        shared_expert_intermediate_size=s["shared_ffn"], router_aux_loss_coef=s["aux_coef"],
        output_router_logits=False, layer_types=types, max_position_embeddings=seq, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": s["rope_theta"], "partial_rotary_factor": s["partial_rotary"]},
        partial_rotary_factor=s["partial_rotary"], tie_word_embeddings=True, attention_bias=False, initializer_range=0.02,
        bos_token_id=bos_token_id if bos_token_id is not None else eos_token_id, eos_token_id=eos_token_id,
        pad_token_id=pad_token_id if pad_token_id is not None else eos_token_id)
    cfg.use_cache = False
    cfg._experts_implementation = s["experts_impl"]
    cfg._attn_implementation = "sdpa"
    return cfg


def budget_tokens(train_flops_per_token, compute, batch_tokens):
    """Largest whole number of global batches that fits in the compute budget."""
    return int(compute) // (train_flops_per_token * batch_tokens) * batch_tokens


def preset_budget(name, vocab_size, seq=8192, batch_tokens=524288):
    """Recalculate a preset's token budget for another vocabulary / sequence length."""
    preset = load_preset(name)
    fc = count(model_config(preset["model"], vocab_size, seq).to_dict(), seq)
    return budget_tokens(fc["train_flops_per_token"], preset["compute"], batch_tokens), fc
