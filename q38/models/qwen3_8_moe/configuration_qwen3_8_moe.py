# Vendored from transformers 5.19.0 (text model only); this repository owns this editable copy.
# The inherited Q35 backbone was generated from modular_qwen3_5_moe.py. This repository
# does not run that generator: edit this file directly for model experiments.
# See UPSTREAM.json for original hashes and LICENSE for Apache-2.0 terms.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import auto_docstring


@strict
class Qwen3_8MoeTextConfig(PreTrainedConfig):
    """Q38 text config: inherited GDN/MoE plus QSA, mHC, N-gram and MTP fields."""

    # A distinct type prevents AutoModel silently falling back to the stock model.
    model_type = "qwen38_moe_sparse"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_a": "colwise_gather_output",
        "layers.*.linear_attn.out_proj": "colwise_gather_output",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }

    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = False
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    intermediate_size: int = 0  # Dense SwiGLU width; num_experts=0 selects this FFN.
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 5
    num_experts: int = 256
    n_branch: int = 4
    mhc_iters: int = 20
    # Missing fields retain the architecture of historical checkpoints.
    # The current baseline preset explicitly disables mHC and selects full SDPA.
    mhc_enabled: bool = True
    qsa_blocks: int = 8
    qsa_mode: str = "learned"
    indexer_heads: int = 4
    indexer_head_dim: int = 64
    ngram_rows: int = 0
    mtp: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __post_init__(self, **kwargs):
        if self.qsa_mode not in ("learned", "full"):
            raise ValueError("qsa_mode must be learned or full")
        if self.mhc_enabled and (self.n_branch < 1 or self.mhc_iters < 1):
            raise ValueError("Enabled mHC requires positive branches and Sinkhorn iterations")
        if self.num_experts < 0:
            raise ValueError("num_experts must be nonnegative")
        if self.num_experts == 0:
            if self.intermediate_size <= 0 or any((self.num_experts_per_tok,
                    self.moe_intermediate_size, self.shared_expert_intermediate_size)):
                raise ValueError("Dense FFN requires intermediate_size>0 and zero routed/shared expert fields")
            if self.router_aux_loss_coef != 0:
                raise ValueError("Dense FFN has no router auxiliary loss")
        elif not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("MoE requires 1 <= num_experts_per_tok <= num_experts")
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


__all__ = ["Qwen3_8MoeTextConfig"]
