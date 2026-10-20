"""Editable Qwen3.5-MoE implementation, based on transformers 5.19.0."""
import hashlib
from pathlib import Path

from .configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from .modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

# HF exports carry this implementation and its config, including future edits.
# Generic AutoModel consumers explicitly opt into exported code with
# trust_remote_code=True; repository training/evaluation import local classes.
Qwen3_5MoeTextConfig.register_for_auto_class("AutoConfig")
Qwen3_5MoeForCausalLM.register_for_auto_class("AutoModelForCausalLM")


def source_info():
    root = Path(__file__).resolve().parent
    return {
        "module": Qwen3_5MoeForCausalLM.__module__,
        "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in ("modeling_qwen3_5_moe.py", "configuration_qwen3_5_moe.py")},
    }


__all__ = ["Qwen3_5MoeTextConfig", "Qwen3_5MoeForCausalLM", "source_info"]
