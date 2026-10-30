"""Presets (configs/<name>.json) and model-config construction.

A preset holds the model shape, preset-specific training defaults, the compute
budget and training defaults.  Everything else uses the shared
defaults below, which reproduce the original scaling rules:
  每 4 层 [lin, lin, lin, full]，层数非 4 的倍数时余下接 lin；64 选 5 路由专家 + 共享专家；
  全注意力 head_dim 64、输出门、partial_rotary 0.25、rope_theta 1e7；绑定嵌入。
"""
import json
from pathlib import Path

from .flops import count
from .models.qwen3_8_moe import Qwen3_8MoeTextConfig

ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "configs"

# Legacy defaults preserve unspecified fields in historical configs/checkpoints.
# configs/baseline.json explicitly selects the current dense/full-attention model.
MODEL_DEFAULTS = dict(layers=5, hidden=96, heads=2, kv_heads=1, head_dim=64, lin_k_heads=1, lin_v_heads=2,
                      lin_head_dim=64, experts=64, topk=5, moe_ffn=24, shared_ffn=24, layer_types="",
                      aux_coef=0.001, rope_theta=1e7, partial_rotary=0.25, experts_impl="eager",
                      n_branch=4, mhc_iters=20, mhc_enabled=1, qsa_mode="learned",
                      qsa_blocks=8, indexer_heads=4, indexer_head_dim=64,
                      ngram_rows=-1, mtp=1, dense_ffn=0)


def preset_names():
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def load_preset(name):
    path = PRESET_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown preset {name!r}; available: {', '.join(preset_names())}")
    return json.loads(path.read_text())


def load_training_config(target):
    """Load an explicit JSON file or a directory containing config.json."""
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        path /= "config.json"
    path = path.resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or not {"compute", "model", "train"} <= config.keys():
        raise ValueError(f"Training config requires compute/model/train sections: {path}")
    if not isinstance(config["model"], dict) or not isinstance(config["train"], dict):
        raise ValueError("model and train must be JSON objects")
    return path, config


def default_layer_types(layers):
    return ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)]


def model_config(shape, vocab_size, seq, bos_token_id=None, eos_token_id=None, pad_token_id=None):
    """Build the text config from shape fields (a dict such as vars(args) or MODEL_DEFAULTS | preset)."""
    s = {**MODEL_DEFAULTS, **shape}
    if s["experts"] == 0:
        if s["dense_ffn"] <= 0 or any(s[k] for k in ("topk", "moe_ffn", "shared_ffn", "aux_coef")):
            raise ValueError("Dense shape requires dense_ffn>0 and topk/moe_ffn/shared_ffn/aux_coef=0")
        if s["ngram_rows"] < 0:
            raise ValueError("Resolve the paired MoE N-gram table size before constructing dense")
    elif s["dense_ffn"]:
        raise ValueError("dense_ffn is only valid when experts=0")
    types = s["layer_types"].split(",") if s["layer_types"] else default_layer_types(s["layers"])
    cfg = Qwen3_8MoeTextConfig(
        vocab_size=vocab_size, hidden_size=s["hidden"], num_hidden_layers=s["layers"],
        num_attention_heads=s["heads"], num_key_value_heads=s["kv_heads"], head_dim=s["head_dim"],
        linear_num_key_heads=s["lin_k_heads"], linear_num_value_heads=s["lin_v_heads"],
        linear_key_head_dim=s["lin_head_dim"], linear_value_head_dim=s["lin_head_dim"], linear_conv_kernel_dim=4,
        num_experts=s["experts"], num_experts_per_tok=s["topk"], moe_intermediate_size=s["moe_ffn"],
        intermediate_size=s["dense_ffn"],
        shared_expert_intermediate_size=s["shared_ffn"], router_aux_loss_coef=s["aux_coef"],
        output_router_logits=False, layer_types=types, max_position_embeddings=seq, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": s["rope_theta"], "partial_rotary_factor": s["partial_rotary"]},
        partial_rotary_factor=s["partial_rotary"], tie_word_embeddings=True, attention_bias=False, initializer_range=0.02,
        bos_token_id=bos_token_id if bos_token_id is not None else eos_token_id, eos_token_id=eos_token_id,
        pad_token_id=pad_token_id if pad_token_id is not None else eos_token_id)
    if s["mhc_enabled"] not in (0, 1) or s["qsa_mode"] not in ("learned", "full"):
        raise ValueError("Invalid mHC enable flag or attention mode")
    if s["mhc_enabled"] and (s["n_branch"] < 1 or s["mhc_iters"] < 1):
        raise ValueError("Enabled mHC requires positive branches and Sinkhorn iterations")
    for key in ("n_branch", "mhc_iters", "mhc_enabled", "qsa_mode", "qsa_blocks", "indexer_heads", "indexer_head_dim", "mtp"):
        setattr(cfg, key, bool(s[key]) if key in ("mtp", "mhc_enabled") else s[key])
    cfg.ngram_rows = max(1024, round(0.205 * count({**cfg.to_dict(), "ngram_rows": 0, "mtp": False}, seq)["total_nonvocab_params"] / s["hidden"] / 1024) * 1024) if s["ngram_rows"] == -1 else s["ngram_rows"]
    if ((cfg.qsa_mode == "learned" and (cfg.indexer_head_dim != cfg.head_dim or cfg.qsa_blocks < 1))
            or cfg.ngram_rows < 0):
        raise ValueError("Invalid Q38 indexer head dimension, sparse budget or N-gram rows")
    cfg.use_cache = False
    cfg._experts_implementation = s["experts_impl"]
    cfg._attn_implementation = "sdpa"
    return cfg


def dense_shape(shape, vocab_size, seq):
    """Replace only the FFN, resolving shared non-FFN dimensions from the MoE first."""
    reference = model_config(shape, vocab_size, seq)
    if reference.num_experts <= 0:
        raise ValueError("dense_shape expects a MoE reference")
    width = (reference.num_experts_per_tok * reference.moe_intermediate_size
             + reference.shared_expert_intermediate_size)
    return {**shape, "experts": 0, "topk": 0, "moe_ffn": 0, "shared_ffn": 0,
            "dense_ffn": width, "aux_coef": 0.0, "experts_impl": "eager",
            "ngram_rows": reference.ngram_rows}


def budget_tokens(train_flops_per_token, compute, batch_tokens):
    """Largest whole number of global batches that fits in the compute budget."""
    return int(compute) // (train_flops_per_token * batch_tokens) * batch_tokens


def preset_budget(name, vocab_size, seq=None, batch_tokens=None):
    """Recalculate a preset's token budget for another vocabulary / sequence length."""
    preset = load_preset(name)
    seq = seq if seq is not None else preset["train"].get("seq", 4096)
    batch_tokens = batch_tokens if batch_tokens is not None else preset["train"].get("batch_tokens", 524288)
    fc = count(model_config(preset["model"], vocab_size, seq).to_dict(), seq)
    reserve = preset.get("accounting", {}).get("compute_reserve_fraction", 0.0)
    return budget_tokens(fc["train_flops_per_token"], int(preset["compute"] * (1-reserve)), batch_tokens), fc
