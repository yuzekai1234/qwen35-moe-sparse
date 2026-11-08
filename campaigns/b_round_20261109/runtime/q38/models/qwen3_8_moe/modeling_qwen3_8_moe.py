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

import copy
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import (
    use_experts_implementation,
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub_with_fallback,
    use_kernelized_func,
)
from transformers.integrations.accelerate import force_accelerate_hooks
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_exporting,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs
from .configuration_qwen3_8_moe import Qwen3_8MoeTextConfig


class Qwen3_8MoeTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_8MoeTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3_8MoeTextConfig, device=None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3_8MoeText has different position ids for the grids
        # So position_ids is broadcast to (3, bs, positions) and freqs has shape (3, bs, positions, dim // 2)
        position_ids = position_ids.expand(3, -1, -1)
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling

        sin = self.recomposition_frequencies(sin)
        cos = self.recomposition_frequencies(cos)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        freqs_thw = freq[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


# NOTE: the FLA package does not re-cast to `input_dtype` in its implementation, maybe we should do the same
@use_kernel_forward_from_hub("RMSNormGated")
class Qwen3_8MoeRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


@use_kernel_func_from_hub_with_fallback("causal_conv1d_update", "causal_conv1d")
def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
    **kwargs,
):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


# NOTE: the FLA package computes `x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)` instead, so if we align
# with the GatedRMSNorm, maybe we can make that change as well.
def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes the gated delta rule, by chunking along the sequence dimension.
    Args:
        query: Query tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        key: Key tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        value: Value tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]. num_v_heads can be equal
            to num_k_heads, same for v_head_dim and k_head_dim.
        g: Decay (in log space) tensor of shape [batch_size, sequence_length, num_v_heads]: the recurrent state is
            multiplied by exp(g) at each step, so entries must be <= 0.
        beta: Beta tensor of shape [batch_size, sequence_length, num_v_heads]
        chunk_size: Size of the chunks along the sequence dimension.
        initial_state: The recurrent state, an optional tensor of shape [batch_size, num_v_heads, k_head_dim, v_head_dim]
        output_final_state: Whether to output the new recurrent state along with the output.
        use_qk_l2norm_in_kernel: If this flag is set to True, query and key vectors are L2-normalized.
    Returns:
        - The output tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]
        - Either None or the new recurrent state tensor of shape [batch_size, num_v_heads, k_head_dim, v_head_dim]
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)  # -1 is the padded sequence length
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (in fp32 to match the FLA library)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # And always normalize queries by the head dimension
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    # Pad sequence length to be a multiple of chunk_size. Padding is described as (left_pad, right_pad) for each dim.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # Apply beta to K and V, which is the "learning rate" of the recurrent state for a given token, ie. how much the new
    # state influence the old state. Beta is often normalized to (0, 1) where 0 = no update; 1 = overwrite old state.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape all tensors to chunk the sequence dimension (adds a new dimension of size chunk_size)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    # Create a chunk-sized strictly upper triangular mask, ie. the mask of what a causal chunk may not attend to
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)

    # Cumulative decay within each chunk (dim 3 is the position inside the chunk). Since decay is in log space,
    # cum_decay[..., t] is the log of a product of decays between the start of the chunk and position t
    cum_decay = decay.cumsum(dim=3)

    # First phase: compute intra-chunk quantities.
    # The pairwise decays: pairwise_decay[..., i, j] = exp(cum_decay_i - cum_decay_j) is the decay accumulated between
    # positions j and i of a chunk. Positive values are masked to -inf before exp to avoid overflow
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()  # with the exp, we exit log space, so we can apply this decay to the states

    # Compute auxiliary tensors: the Upper Triangular (ut) transform system and the intra-chunk attn (QK dot product)
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Gated delta attention uses a UT transform to condense several delta rule updates into a few matmuls. After the UT
    # system is solved, we can then compute the new_values (called "u" in the DeltaNet paper) and the decayed keys
    # reading the old state (k_cumdecay). In the update, the part of new_values that the old state already predicts is
    # subtracted out, so that only the correction is written to the recurrent state: this is the delta rule.

    # Not all export targets support the fast triangular solver, so we build the inverse by forward substitution then
    if not is_torchdynamo_exporting():
        new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    else:
        ut_system = -ut_system.tril(-1)  # ut_system is masked to only keep the strictly lower triangle
        for i in range(1, chunk_size):
            row = ut_system[..., i, :i].clone()
            sub = ut_system[..., :i, :i].clone()
            ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut_system = ut_system + torch.eye(chunk_size, dtype=ut_system.dtype, device=ut_system.device)
        new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    # Apply decay once rather than in each chunk
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Second phase: the sequential scan over chunks
    for i in range(num_chunks):
        # Compute attention output for the current chunk: add the read of the previous recurrent state
        # (inter_chunk_attn) with the within-chunk attention (intra_chunk_attn)
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        # Update the recurrent state: new recurrent state (S_t+1) = decayed old state (S_t * (I-βkk^T)) + update (βvk^T)
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Reshape the output to the orignal shape: flatten the chunk dimension, then drop padding
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


@use_kernel_func_from_hub_with_fallback("fused_recurrent_gated_delta_rule", "fla")
def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes linear attention using the gated delta rule, by iterating over each token in the sequence dimension.
    Same args and return value as torch_chunk_gated_delta_rule, except for `chunk_size` because the sequence dim is not
    chunked."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (done once in fp32 for better accuracy)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # And always normalize queries by the head dimension
    query = query / (query.shape[-1] ** 0.5)

    # Create the storage for the last recurrent state, which will be updated in place. If a previous state is provided,
    # it is the starting point, otherwise start with a zeroed buffer.
    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    # Loop over each token and update the recurrent state
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # Decay the recurrent state
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        # Update the recurrent state with the current token
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


@use_kernel_forward_from_hub("Qwen3_8GatedDeltaNet")
@use_kernelized_func(
    [torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update]
)
class Qwen3_8MoeGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_8MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # Lower bound kept away from 0 so log(A) never becomes -inf
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_8MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            # Drop the additional previous states
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )

        # Update cache
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3_8MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3_8MoeTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_8MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3_8MoeRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_8MoeMLP(nn.Module):
    def __init__(self, config: Qwen3_8MoeTextConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@use_experts_implementation
class Qwen3_8MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class PaperLossFreeRouter(nn.Module):
    """Bias changes selection only; the trainer calls update_bias after a step."""

    def __init__(self, hidden, experts, topk, update_rate=0.001):
        super().__init__()
        self.hidden_dim, self.num_experts, self.top_k = hidden, experts, topk
        self.update_rate = update_rate
        self.weight = nn.Parameter(torch.zeros(experts, hidden))
        # A persistent non-gradient scalar per expert is still counted in N.
        self.routing_bias = nn.Parameter(torch.zeros(experts), requires_grad=False)
        # Plain temporary tensor: DDP must not broadcast rank0's accumulated
        # load over other ranks at the start of every microbatch forward.
        self.step_load = torch.zeros(experts, dtype=torch.int64)

    def forward(self, hidden):
        hidden = hidden.reshape(-1, self.hidden_dim)
        logits = F.linear(hidden, self.weight)
        scores = logits.float().sigmoid()
        selected = (scores + self.routing_bias.float()).topk(self.top_k, -1).indices
        weights = scores.gather(-1, selected)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                if self.step_load.device != selected.device:
                    self.step_load = torch.zeros(self.num_experts, dtype=torch.int64, device=selected.device)
                self.step_load.add_(torch.bincount(selected.reshape(-1), minlength=self.num_experts))
        return logits, weights.to(logits.dtype), selected

    @torch.no_grad()
    def update_bias(self, distributed=False):
        counts = self.step_load.clone()
        if distributed:
            torch.distributed.all_reduce(counts)
        error = counts.float().mean() - counts.float()
        self.routing_bias.add_(error.sign(), alpha=self.update_rate)
        # Adding a common scalar leaves top-k unchanged.
        self.routing_bias.sub_(self.routing_bias.mean())
        self.step_load.zero_()
        return counts


@use_kernel_forward_from_hub("SoftmaxTopKRouter")
class Qwen3_8MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen3_8MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = (PaperLossFreeRouter(config.hidden_size, config.num_experts, config.num_experts_per_tok,
                     config.paper_bias_rate) if config.paper_loss_free else Qwen3_8MoeTopKRouter(config))
        self.experts = Qwen3_8MoeExperts(config)
        self.shared_expert = Qwen3_8MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


@use_kernel_forward_from_hub("RMSNormZeroCentered")
class Qwen3_8MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3_8Moe is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_8MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_8MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_8MoeGatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_8MoeAttention(config, layer_idx)
        self.mlp = (Qwen3_8MoeSparseMoeBlock(config) if config.num_experts
                    else Qwen3_8MoeMLP(config, intermediate_size=config.intermediate_size))
        self.input_layernorm = Qwen3_8MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_8MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif self.block_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3_8MoePreTrainedModel(PreTrainedModel):
    config: Qwen3_8MoeTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_8MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3_8MoeTopKRouter, index=0),
        "hidden_states": Qwen3_8MoeDecoderLayer,
        "attentions": Qwen3_8MoeAttention,
    }
    _is_stateful = True
    _can_compile_fullgraph = True

    def _initialize_weights(self, module, is_custom_code: bool = False):
        # This vendored implementation uses Transformers initialization helpers.
        # The framework's compatibility guard for arbitrary remote code would
        # skip parent-only initialization (e.g. the MoE block router redraw),
        # changing seeded initialization solely because the module moved here.
        # Preserve the exact built-in initialization traversal instead.
        return super()._initialize_weights(module, is_custom_code=False)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_8MoeGatedDeltaNet):
            init.ones_(module.dt_bias)
            # Lower bound kept away from 0 so log(A) never becomes -inf
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device).uniform_(0.01, 16).log_(),
            )
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, Qwen3_8MoeRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen3_8MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3_8MoeSparseMoeBlock):
            init.normal_(module.gate.weight, mean=0.0, std=self.config.initializer_range)


@dataclass
class Qwen3_8MoeModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
        The attribute is deprecated and will be removed in v5.20, use `model.base_model.rope_deltas` instead.
    """

    rope_deltas: torch.LongTensor | None = None
    router_logits: tuple[torch.FloatTensor] | None = None


from torch.nn.attention.flex_attention import flex_attention, BlockMask
BLK = 64
# Torch 2.7 validates its default tile *before* reading kernel_options. Its
# A100 defaults use 128-wide tiles and reject the model's 64-token blocks.
# Keep the compatibility adjustment scoped to that pinned Torch release.
if torch.__version__.startswith("2.7."):
    import torch._inductor.lowering  # initialize Inductor before its kernel modules
    import torch._inductor.kernel.flex_attention as _flex_codegen
    _flex_codegen._get_default_config_fwd = lambda query: (64, 64, 4, 3)
    _flex_codegen._get_default_config_bwd = lambda query: (32, 64, 4, 3)

_flex = torch.compile(flex_attention, dynamic=False)
_KO = {}


def _causal(b, h, q, kv): return q >= kv


# ---------------------------------------------------------------- 分支残差 (Hyper-Connections / mHC)
def _sinkhorn(logits, iters=20):
    P = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        P = P / P.sum(-1, keepdim=True).clamp(min=1e-9); P = P / P.sum(-2, keepdim=True).clamp(min=1e-9)
    return P


def _coeffs(Xf, A, B, b_pre, b_post, b_res, alphas, n, iters):
    """Xf: [B,S,n*h] 已 RMSNorm。dyn = tanh(B(A(Xf))) → (pre[B,S,n], post[B,S,n], res[B,S,n,n])。"""
    z = torch.tanh(F.linear(F.linear(Xf, A), B).float())
    zp, zq, zr = z[..., :n], z[..., n:2 * n], z[..., 2 * n:].reshape(*z.shape[:-1], n, n)
    pre = torch.sigmoid(b_pre + alphas[0] * zp)
    post = 2 * torch.sigmoid(b_post + alphas[1] * zq)
    res = _sinkhorn(b_res + alphas[2] * zr, iters)
    return pre, post, res


def _agg(X, pre): return (pre.unsqueeze(-1) * X).sum(2)          # X [B,S,n,h] → [B,S,h]


def _mix(X, res, post, y):                                        # X' = H_res X + H_post^T y
    return torch.einsum("bsmn,bsnh->bsmh", res, X) + post.unsqueeze(-1) * y.float().unsqueeze(2)


_agg_c = torch.compile(_agg); _mix_c = torch.compile(_mix); _coeffs_c = torch.compile(_coeffs)


def _mhc_pre(X, A, B, b_pre, b_post, b_res, alphas, n, iters):
    X = X.float()
    Xf = X.flatten(2)
    Xf = Xf * torch.rsqrt(Xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    coeffs = _coeffs(Xf, A.float(), B.float(), b_pre.float(), b_post.float(), b_res.float(), alphas.float(), n, iters)
    return _agg(X, coeffs[0]), (X,) + coeffs


_mhc_pre_c = torch.compile(_mhc_pre)


class MHC(nn.Module):
    """一个子层(注意力/GDN 或 MoE)的分支残差系数。静态参数 + 动态部分(RMSNorm(flatten 4 路) → 低秩 r → 24 个系数, tanh 限幅)。
    初始化 = 普通残差: 动态部分输出 0(B 零初始化); H_res≈I(对角 0/非对角 −8 过 Sinkhorn); H_pre 和为 1(本层"主读"的那路 0.7、其余 0.1);
    H_post 全 1(每路都写子层输出)。embedding 复制 4 路时前向与普通残差一致(差异来自 Sinkhorn 的 ~1e-3 泄漏)。"""
    def __init__(self, h, n=4, r=None, slot=0, iters=20):
        super().__init__()
        self.n, self.iters = n, iters; r = r or round(h / 8)
        self.A = nn.Linear(n * h, r, bias=False); self.B = nn.Linear(r, 2 * n + n * n, bias=False)
        nn.init.normal_(self.A.weight, std=0.02); nn.init.zeros_(self.B.weight)
        pre = torch.full((n,), 0.1); pre[slot % n] = 0.7
        self.b_pre = nn.Parameter(torch.log(pre / (1 - pre))); self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.full((n, n), -8.0) + 8.0 * torch.eye(n)); self.alphas = nn.Parameter(torch.full((3,), 0.1))

    def n_params(self): return sum(p.numel() for p in self.parameters())

    def pre(self, X):
        fn = _mhc_pre_c if X.is_cuda else _mhc_pre
        with torch.autocast("cuda", enabled=False):
            return fn(X, self.A.weight, self.B.weight, self.b_pre, self.b_post,
                      self.b_res, self.alphas, self.n, self.iters)

    def post(self, X, y, c): return (_mix_c if X.is_cuda else _mix)(c[0], c[3], c[2], y)


# ---------------------------------------------------------------- QSA: lightning indexer + 块稀疏注意力
def _idx_score_fn(q, k, w, scale):
    s = torch.matmul(q, k.transpose(-1, -2))                                  # [B,nh,S,S] bf16
    return torch.einsum("bhts,bth->bts", F.relu(s.float()) * scale, w)


_idx_score = torch.compile(_idx_score_fn)


class Indexer(nn.Module):
    """DSA 风格: MQA, n_ih 个 q 头 + 1 个共享 k 头, head_dim d_ih; 打分 I[t,s] = Σ_j w_j(t)·ReLU(q_j(t)·k(s)) (RoPE 同主干的 partial rope)。
    只靠 KL 蒸馏训练(输入 detach, 不回传主干)。"""
    def __init__(self, h, n_ih=4, d_ih=64):
        super().__init__()
        self.n_ih, self.d_ih = n_ih, d_ih
        self.wq = nn.Linear(h, n_ih * d_ih, bias=False); self.wk = nn.Linear(h, d_ih, bias=False); self.ww = nn.Linear(h, n_ih, bias=False)

    def forward(self, x, cos, sin, block_queries=False):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_ih, self.d_ih).transpose(1, 2); k = self.wk(x).view(B, S, 1, self.d_ih).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))                          # q [B,nh,S,d]  k [B,1,S,d]
        w = self.ww(x).float() * (self.n_ih ** -0.5)                            # [B,S,nh]
        if block_queries:
            q, w = q[:, :, ::BLK], w[:, ::BLK]
        return (_idx_score if q.is_cuda else _idx_score_fn)(q, k, w, self.d_ih ** -0.5)                           # [B,S,S]


@torch.no_grad()
def block_mass(q, k, scale, chunk=256):
    """主注意力(稠密, 因果)在每个 64 块上的概率质量, 头平均: q [B,nq,S,d], k [B,nkv,S,d] → [B,S,nb] (fp32, detach)。"""
    B, nq, S, d = q.shape; nkv = k.shape[1]; rep = nq // nkv; nb = S // BLK
    qg = q.view(B, nkv, rep, S, d); out = []
    kk = k[:, :, None]
    cols = torch.arange(S, device=q.device)
    for t0 in range(0, S, chunk):
        sc = torch.matmul(qg[:, :, :, t0:t0 + chunk], kk.transpose(-1, -2)).float() * scale          # [B,nkv,rep,c,S]
        rows = torch.arange(t0, min(S, t0 + chunk), device=q.device)
        sc = sc.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
        p = torch.softmax(sc, dim=-1).view(B, nkv, rep, -1, nb, BLK).sum(-1)
        out.append(p.mean(dim=(1, 2)))
    return torch.cat(out, dim=1)


def select_blocks(lp_blk, budget_blocks):
    """lp_blk: [B,S,nb] indexer 在块上的 log 概率(−inf=非因果)。按 query 块首 token 的概率 → 每个 query 块 top-k 块, 总含当前块。
    返回 BlockMask 需要的 (kv_num_blocks [B,1,nbq], kv_indices [B,1,nbq,k]) (int32)。"""
    B, queries, nb = lp_blk.shape; nbq = nb; k = min(budget_blocks, nb)
    # Select using the first query of each block: later tokens must not affect earlier outputs.
    p = lp_blk.detach().exp() if queries == nb else lp_blk.detach().exp().view(B, nbq, BLK, nb)[:, :, 0]
    qb = torch.arange(nbq, device=p.device)[:, None]; kb = torch.arange(nb, device=p.device)[None, :]
    p = torch.where(kb > qb, torch.full_like(p, -1.0), p)
    p = torch.where(kb == qb, torch.full_like(p, 2.0), p)
    idx = p.topk(k, dim=-1).indices
    num = torch.clamp(qb[:, 0] + 1, max=k).to(torch.int32)[None].expand(B, -1)
    full = torch.zeros(B, 1, nbq, nb, dtype=torch.int32, device=p.device); full[..., :k] = idx[:, None].to(torch.int32)   # from_kv_blocks 要求最后一维 = 块总数(尾部不读)
    return num[:, None].contiguous(), full


QSA_STAT = {}


def qsa_attention(attn, x, cos, sin, indexer, budget_blocks, dense, want_kl):
    """用 attn(Qwen3_5MoeAttention 的权重) 做注意力; dense=True 走全注意力(预热/测试), 否则 indexer 选块 + flex_attention。
    返回 (输出, kl 或 None)。KL = mean_t Σ_blk P_main(blk|t)·(log P_main − log P_idx), P_main 为稠密主注意力(头平均, detach)。"""
    B, S, _ = x.shape; hd = attn.head_dim
    q, gate = torch.chunk(attn.q_proj(x).view(B, S, -1, hd * 2), 2, dim=-1); gate = gate.reshape(B, S, -1)
    q = attn.q_norm(q).transpose(1, 2); k = attn.k_norm(attn.k_proj(x).view(B, S, -1, hd)).transpose(1, 2); v = attn.v_proj(x).view(B, S, -1, hd).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))
    kl = None; lp_blk = None
    if indexer is not None and (want_kl or not dense):
        block_queries = not want_kl and not bool(os.environ.get("Q38_INDEXER_BASELINE"))
        I = indexer(x.detach(), cos, sin, block_queries=block_queries)
        ar = torch.arange(S, device=x.device)
        query_positions = ar[::BLK] if block_queries else ar
        I = I.masked_fill(ar[None, :] > query_positions[:, None], -1e9)    # 有限大负数而非 -inf: 避免 logsumexp/log_softmax 反传出 NaN
        lp = torch.log_softmax(I, dim=-1)
        lp_blk = torch.logsumexp(lp.view(B, I.shape[1], S // BLK, BLK), dim=-1)                  # [B,S,nb]
        if want_kl:
            P = block_mass(q.detach(), k.detach(), attn.scaling)
            kl = torch.where(P > 0, P * (torch.log(P.clamp(min=1e-12)) - lp_blk), torch.zeros_like(P)).sum(-1).mean()
            if QSA_STAT.get("on") and not dense:
              with torch.no_grad():   # 统计: 稠密主注意力概率质量落在 indexer 选中块里的比例(recall), 对照"随机选同样多块"的期望
                  num_, idx_ = select_blocks(lp_blk, budget_blocks); nbq_, nb_ = S // BLK, S // BLK
                  sel = torch.zeros(B, nbq_, nb_, device=x.device).scatter_(-1, idx_[:, 0, :, :min(budget_blocks, nb_)].long(), 1.0)
                  sel = sel * (torch.arange(nb_, device=x.device)[None, :] <= torch.arange(nbq_, device=x.device)[:, None])
                  rec = (P.view(B, nbq_, BLK, nb_) * sel[:, :, None, :]).sum(-1).mean()
                  QSA_STAT.setdefault("recall", []).append(rec.item())
    if dense or indexer is None:
        o = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling, is_causal=True, enable_gqa=True)
    else:
        num, idx = select_blocks(lp_blk, budget_blocks)
        bm = BlockMask.from_kv_blocks(num, idx, BLOCK_SIZE=BLK, mask_mod=_causal, seq_lengths=(S, S))
        if not q.is_cuda:
            nb = S // BLK
            chosen = torch.zeros(B, nb, nb, dtype=torch.bool, device=x.device)
            for j in range(nb):
                count = min(j + 1, budget_blocks)
                chosen[:, j].scatter_(1, idx[:, 0, j, :count].long(), True)
            mask = chosen.repeat_interleave(BLK, 1).repeat_interleave(BLK, 2)
            mask = mask & torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], scale=attn.scaling, enable_gqa=True)
        else:
            kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32,
                              "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}
            # Torch 2.7 switches short GQA queries to a decoding kernel, whose
            # BLOCK_M must cover S * (query_heads / kv_heads), not just S.
            # These are uncached prefill tails: use the prefill kernel with the
            # exact same selected-block/causal mask, rather than dropping it.
            if torch.__version__.startswith("2.7.") and S < 128:
                kernel_options["FORCE_USE_FLEX_ATTENTION"] = True
            o = _flex(q, k, v, scale=attn.scaling, block_mask=bm,
                      enable_gqa=True, kernel_options=kernel_options)
    o = o.transpose(1, 2).reshape(B, S, -1) * torch.sigmoid(gate)
    return attn.o_proj(o), kl



P2 = (1000003, 998244353); P3 = (1000033, 998244353, 1000000007)
MIX = 2654435761


class PaperEngram(nn.Module):
    """Orders 2/3, independent hash heads, context gate, causal dilated conv.

    Raw token IDs and EOS boundaries are preserved. Unlike the paper, there is
    no tokenizer canonicalization and no multi-branch residual backbone.
    """

    def __init__(self, hidden, vocab, rows, eos_id, heads=4, head_dim=128):
        super().__init__()
        self.hidden, self.vocab, self.eos_id = hidden, vocab, eos_id
        self.heads, self.head_dim = heads, head_dim
        # Distinct prime table sizes reduce correlated modulo collisions.
        def previous_prime(n):
            while n > 2:
                if all(n % d for d in range(2, math.isqrt(n) + 1)):
                    return n
                n -= 1
            raise ValueError('Table too small')
        sizes = []
        n = rows
        for _ in range(2 * heads):
            n = previous_prime(n)
            sizes.append(n)
            n -= 2
        self.sizes = tuple(sizes)
        self.tables = nn.ModuleList([nn.Embedding(n, head_dim) for n in sizes])
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.02)
        memory_dim = 2 * heads * head_dim
        self.key = nn.Linear(memory_dim, hidden, bias=False)
        self.value = nn.Linear(memory_dim, hidden, bias=False)
        self.query_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.key_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.conv_norm = nn.RMSNorm(hidden, eps=1e-6)
        self.conv = nn.Conv1d(hidden, hidden, 4, groups=hidden, dilation=3, bias=False)
        # Start at the original N-gram module's small residual scale.
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.value.weight, std=0.002)
        nn.init.zeros_(self.conv.weight)

    def indices(self, ids):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).expand(b, t)
        start = torch.cummax(torch.where(ids == self.eos_id, pos, -1), 1).values.clamp_min(0)
        prior = []
        for shift in (2, 1, 0):
            count = min(shift, t)
            x = F.pad(ids[:, :t-count], (count, 0), value=self.vocab) if shift else ids
            prior.append(torch.where(pos - shift >= start, x, self.vocab))
        result = []
        for order in (2, 3):
            for head in range(self.heads):
                # Signed-int64 arithmetic intentionally wraps modulo 2**64,
                # as in the existing XOR/multiply N-gram hash.
                z = torch.zeros_like(ids)
                for j, x in enumerate(prior[3-order:]):
                    z = z ^ (x * (1000003 + 104729 * (head + 1) + 13007 * (j + 1)))
                z = (z ^ (z >> 17)) * (2654435761 + 2 * head)
                result.append(z % self.sizes[len(result)])
        return result

    def delta(self, hidden, ids):
        memory = torch.cat([tab(ix) for tab, ix in zip(self.tables, self.indices(ids))], -1)
        key = self.key_norm(self.key(memory))
        query = self.query_norm(hidden.to(key.dtype))
        gate = torch.sigmoid((query.float() * key.float()).sum(-1, keepdim=True) / math.sqrt(self.hidden))
        value = (gate * self.value(memory).float()).to(hidden.dtype)
        # Explicit left padding and per-tap EOS masking prevent future access
        # and prevent the short convolution from crossing document boundaries.
        normed = self.conv_norm(value.to(self.conv_norm.weight.dtype)).to(value.dtype)
        b, t, h = normed.shape
        pos = torch.arange(t, device=ids.device).expand(b, t)
        start = torch.cummax(torch.where(ids == self.eos_id, pos, -1), 1).values.clamp_min(0)
        convolved = torch.zeros_like(normed)
        for tap in range(4):
            lag = (3 - tap) * 3
            n = min(lag, t)
            shifted = F.pad(normed[:, :t-n], (0, 0, n, 0)) if lag else normed
            mask = (pos - lag >= start).unsqueeze(-1)
            convolved = convolved + shifted * mask * self.conv.weight[:, 0, tap].to(value.dtype)
        return value + F.silu(convolved)


class NGramEmbedding(nn.Module):
    def __init__(self, hidden, vocab, rows, eos_id, gate_bias=-1.0, init_std=0.02, norm_init=0.02):
        super().__init__()
        self.hidden, self.vocab, self.rows, self.eos_id = hidden, vocab, rows, eos_id
        self.emb2 = nn.Embedding(rows, hidden); self.emb3 = nn.Embedding(rows, hidden)
        nn.init.normal_(self.emb2.weight, std=init_std); nn.init.normal_(self.emb3.weight, std=init_std)
        self.norm = nn.RMSNorm(hidden, eps=1e-6); nn.init.constant_(self.norm.weight, norm_init)   # e 的 RMS 起步 ≈ 嵌入量级, 不压过残差流
        self.gate = nn.Linear(hidden, hidden, bias=True)
        nn.init.normal_(self.gate.weight, std=0.02); nn.init.constant_(self.gate.bias, gate_bias)
        self._ids = None; self.stat = {}

    def table_params(self): return self.emb2.weight.numel() + self.emb3.weight.numel()
    def gate_params(self): return sum(p.numel() for p in self.gate.parameters()) + self.norm.weight.numel()

    def indices(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).expand(B, T)
        last = torch.cummax(torch.where(x == self.eos_id, pos, torch.full_like(pos, -1)), dim=1).values.clamp(min=0)
        def prev(k):
            shift = min(k, T)
            xs = torch.cat([torch.full((B, shift), self.vocab, device=x.device, dtype=x.dtype), x[:, :T-shift]], dim=1)
            return torch.where((pos - k) >= last, xs, torch.full_like(xs, self.vocab))
        a, b = prev(1), x
        i2 = (a * P2[0]) ^ (b * P2[1])
        c = prev(2)
        i3 = (c * P3[0]) ^ (a * P3[1]) ^ (b * P3[2])
        i2 = (i2 ^ (i2 >> 17)) * MIX; i3 = (i3 ^ (i3 >> 17)) * MIX
        return (i2 % self.rows), (i3 % self.rows)

    def delta(self, h, x):
        """返回要加到残差流上的增量 g*e (h 只用来算门; 4 路分支残差下传各路的均值)。"""
        return self.forward(h, x) - h

    def forward(self, h, x):
        i2, i3 = self.indices(x)
        e = self.norm(self.emb2(i2) + self.emb3(i3)).to(h.dtype)
        g = torch.sigmoid(self.gate(h))
        with torch.no_grad():
            self.stat = dict(gate_mean=g.float().mean().item(), gate_frac_lt01=(g < 0.1).float().mean().item(), gate_frac_gt09=(g > 0.9).float().mean().item(),
                             ratio=((g * e).float().norm() / h.float().norm().clamp(min=1e-9)).item()) if getattr(self, "collect", False) else self.stat
        return h + g * e


def fft_segments(length, taps):
    """FFT length and number of output segments of causal_fft_conv for a sequence of `length` positions."""
    if taps < length and length % taps == 0:
        return 2 * taps, length // taps
    return 1 << (length + taps - 2).bit_length(), 1


def causal_fft_conv(u, taps):
    """z[:, t] = sum over j < min(t + 1, K) of u[:, t - j] @ taps[j], for u [B, L, c] and taps [K, c, c].

    When K < L divides L, the sequence is cut into L/K output segments of K positions, each computed from the 2K
    inputs ending with it (overlap-save; zeros before the start): length-2K real FFTs and one c x c complex product
    per frequency and segment. Otherwise one zero-padded FFT of length n >= L + K - 1 covers the sequence. In both
    cases the circular convolution equals the causal one on the kept positions. FP32 outside autocast."""
    B, L, c = u.shape
    K = taps.shape[0]
    n, segments = fft_segments(L, K)
    with torch.autocast(u.device.type, enabled=False):
        W = torch.fft.rfft(taps.float(), n=n, dim=0)                                    # [n/2+1, c, c]
        if segments == 1:
            U = torch.fft.rfft(u.float(), n=n, dim=1)                                   # [B, n/2+1, c]
            Z = torch.matmul(U.transpose(0, 1), W).transpose(0, 1)
            return torch.fft.irfft(Z, n=n, dim=1)[:, :L]
        windows = F.pad(u.float(), (0, 0, K, 0)).unfold(1, n, K)                        # [B, S, c, 2K]
        U = torch.fft.rfft(windows, dim=-1).permute(3, 0, 1, 2).reshape(K + 1, B * segments, c)
        Z = torch.matmul(U, W).reshape(K + 1, B, segments, c).permute(1, 2, 3, 0)      # [B, S, c, K+1]
        z = torch.fft.irfft(Z, n=n, dim=-1)[..., K:]                                    # [B, S, c, K]
        return z.permute(0, 1, 3, 2).reshape(B, L, c)


def _fft_inputs(u, n, K, segments):
    """Frequency representation [n/2+1, rows, c] of u [B, L, c]; rows = B * segments (see causal_fft_conv)."""
    B, L, c = u.shape
    if segments == 1:
        return torch.fft.rfft(u, n=n, dim=1).transpose(0, 1)
    windows = F.pad(u, (0, 0, K, 0)).unfold(1, n, K)                                    # [B, S, c, 2K]
    return torch.fft.rfft(windows, dim=-1).permute(3, 0, 1, 2).reshape(n // 2 + 1, B * segments, c)


def _fft_outputs(Z, n, K, segments, B, L):
    """Positions [B, L, c] of the frequency products Z [n/2+1, rows, c] (inverse of _fft_inputs)."""
    c = Z.shape[-1]
    if segments == 1:
        return torch.fft.irfft(Z.transpose(0, 1), n=n, dim=1)[:, :L]
    Z = Z.reshape(n // 2 + 1, B, segments, c).permute(1, 2, 3, 0)
    return torch.fft.irfft(Z, n=n, dim=-1)[..., K:].permute(0, 1, 3, 2).reshape(B, L, c)


def _fft_columns(U, taps, scale, n, K, segments, B, L):
    return _fft_outputs(torch.matmul(U, torch.fft.rfft(taps * scale, n=n, dim=0)), n, K, segments, B, L)


class FFTConv(torch.autograd.Function):
    """causal_fft_conv(u, taps * scale) computed over output-column chunks of the taps.

    Only one chunk's frequency response [n/2+1, c, chunk] exists at a time; the backward pass recomputes it per chunk
    instead of saving it, so memory stays bounded for large c and K at the cost of one more filter transform."""

    @staticmethod
    def forward(ctx, u, taps, scale, chunk):
        B, L, c = u.shape
        K = taps.shape[0]
        n, segments = fft_segments(L, K)
        U = _fft_inputs(u, n, K, segments)
        z = torch.cat([_fft_columns(U, taps[:, :, s:s + chunk], scale, n, K, segments, B, L)
                       for s in range(0, taps.shape[2], chunk)], -1)
        ctx.save_for_backward(u, taps, scale)
        ctx.chunk = chunk
        return z

    @staticmethod
    def backward(ctx, grad_z):
        u, taps, scale = ctx.saved_tensors
        B, L, c = u.shape
        K, chunk = taps.shape[0], ctx.chunk
        n, segments = fft_segments(L, K)
        grad_taps = torch.empty_like(taps)
        with torch.enable_grad():
            u_ = u.detach().requires_grad_()
            U = _fft_inputs(u_, n, K, segments)
            U_leaf = U.detach().requires_grad_()
            grad_U = torch.zeros_like(U)
            for s in range(0, taps.shape[2], chunk):
                t = taps[:, :, s:s + chunk].detach().contiguous().requires_grad_()
                z = _fft_columns(U_leaf, t, scale, n, K, segments, B, L)
                g_U, g_t = torch.autograd.grad(z, (U_leaf, t), grad_z[..., s:s + chunk])
                grad_U += g_U
                grad_taps[:, :, s:s + chunk] = g_t
            (grad_u,) = torch.autograd.grad(U, u_, grad_U)
        return grad_u, grad_taps, None, None


def _collective(op, out, inp, group):
    """NCCL collective on the real views of complex tensors; Gloo (CPU tests) emulates the missing ones."""
    import torch.distributed as dist
    out_r, inp_r = torch.view_as_real(out), torch.view_as_real(inp)
    if dist.get_backend(group) == "nccl":
        {"gather": dist.all_gather_into_tensor, "alltoall": dist.all_to_all_single,
         "reduce_scatter": dist.reduce_scatter_tensor}[op](out_r, inp_r, group=group)
        return
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if op == "gather":
        dist.all_gather(list(out_r.unbind(0)), inp_r, group=group)
    elif op == "alltoall":
        everything = inp_r.new_empty((world,) + inp_r.shape)                 # [source, destination, ...]
        dist.all_gather(list(everything.unbind(0)), inp_r.contiguous(), group=group)
        out_r.copy_(everything[:, rank])
    else:
        total = inp_r.clone()
        dist.all_reduce(total, group=group)
        out_r.copy_(total[rank])


class _StepFFTConv(torch.autograd.Function):
    """Training convolution of u [B, L, c] with the block's frequency response cached for the optimizer step.

    Rank r of `world` holds the response W_r [f, c, c/world] of output columns r*c/world ... of the lag filter. The
    forward pass gathers every rank's input transform U, multiplies it by W_r and returns each rank's rows of these
    columns to it (all-to-all); the backward pass sends the output gradient back the same way, accumulates the
    response gradient U^H G into the block's step accumulator and reduce-scatters the input gradient. The taps'
    gradient is formed once per step from the accumulator (LongConv.end_step), so the filter is transformed twice
    per step in total and the large response is only read per micro batch."""

    @staticmethod
    def forward(ctx, u, block):
        B, L, c = u.shape
        K, world, group = block.taps.shape[0], block._world, block._group
        n, segments = fft_segments(L, K)
        U = _fft_inputs(u, n, K, segments).contiguous()                                 # [f, R, c]
        f, R = U.shape[:2]
        if world > 1:
            gathered = U.new_empty((world, f, R, c))
            _collective("gather", gathered, U, group)
            U = gathered.permute(1, 0, 2, 3).reshape(f, world * R, c)
        Z = torch.bmm(U, block._W)                                                     # [f, world*R, c/world]
        if world > 1:
            send = Z.view(f, world, R, -1).permute(1, 0, 2, 3).contiguous()
            received = torch.empty_like(send)
            _collective("alltoall", received, send, group)                              # [p] = rank p's columns
            Z = received.permute(1, 2, 0, 3).reshape(f, R, c)
        ctx.save_for_backward(U)
        ctx.block, ctx.shape = block, (B, L, c, n, K, segments, f, R)
        return _fft_outputs(Z, n, K, segments, B, L)

    @staticmethod
    def backward(ctx, grad_z):
        (U,) = ctx.saved_tensors
        block = ctx.block
        B, L, c, n, K, segments, f, R = ctx.shape
        world, group = block._world, block._group
        with torch.enable_grad():
            Z = torch.zeros((f, R, c), dtype=U.dtype, device=U.device, requires_grad=True)
            (G,) = torch.autograd.grad(_fft_outputs(Z, n, K, segments, B, L), Z, grad_z)
        if world > 1:
            send = G.view(f, R, world, c // world).permute(2, 0, 1, 3).contiguous()   # [q] = rank q's columns
            received = torch.empty_like(send)
            _collective("alltoall", received, send, group)                              # [p] = rank p's rows
            G = received.permute(1, 0, 2, 3).reshape(f, world * R, c // world)
        # Complex matmul gradients as in autograd (conjugate transposes, taken by cuBLAS without copies): the
        # response gradient U^H G accumulates over the micro batches, grad_U = G W^H.
        block._grad.baddbmm_(U.mH, G)
        grad_U = torch.bmm(G, block._W.mH)
        if world > 1:
            local = grad_U.new_empty((f, R, c))
            _collective("reduce_scatter", local, grad_U.view(f, world, R, c).permute(1, 0, 2, 3).contiguous(), group)
            grad_U = local
        with torch.enable_grad():
            u = torch.zeros((B, L, c), dtype=grad_z.dtype, device=grad_z.device, requires_grad=True)
            (grad_u,) = torch.autograd.grad(_fft_inputs(u, n, K, segments), u, grad_U)
        return grad_u, None


class LongConv(nn.Module):
    """Causal convolution over the last `taps` positions with a full channels x channels matrix per lag.

    u, g = split(norm(x) W_in), z_t = sum_j u_{t-j} T_j, y = (z * silu(g)) W_out is added to the residual stream.
    Lag j uses T_j = s_j P_j with learned P_j and the fixed scale s_j = ((j + 1) H)^(-1/2), H = sum_j 1 / (j + 1):
    the squared scales sum to one, so the initial output scale and the summed update of all lags match one
    channels x channels matrix, and near lags start larger than far ones. W_out starts at zero."""

    def __init__(self, hidden, channels, taps, eps, length):
        super().__init__()
        self.length = length   # training sequence length (frequency response of the step cache)
        self._W = self._grad = None
        self.norm = Qwen3_8MoeRMSNorm(hidden, eps=eps)
        self.in_proj = nn.Linear(hidden, 2 * channels, bias=False)
        self.taps = nn.Parameter(torch.randn(taps, channels, channels) / channels ** .5)
        self.out_proj = nn.Linear(channels, hidden, bias=False)
        nn.init.zeros_(self.norm.weight); nn.init.normal_(self.in_proj.weight, std=0.02); nn.init.zeros_(self.out_proj.weight)

    chunk = 128   # output columns per filter-response chunk in FFTConv

    def lag_scale(self, device=None):
        inverse = 1 / torch.arange(1, self.taps.shape[0] + 1, device=device or self.taps.device, dtype=torch.float32)
        return (inverse / inverse.sum()).sqrt().view(-1, 1, 1)

    def lag_filter(self):
        return self.taps.float() * self.lag_scale()

    def _columns(self):
        width = self.taps.shape[2] // self._world
        return slice(self._rank * width, (self._rank + 1) * width)

    @torch.no_grad()
    def begin_step(self, world=1, rank=0, group=None, column_taps=None):
        """Cache the frequency response of this rank's output columns, from column_taps [K, c, c/world] (by default
        those columns of the module's taps), and a zero accumulator of its gradient for the micro batches of one
        optimizer step."""
        if self.taps.shape[2] % world:
            raise ValueError("LongConv channels must be divisible by the data-parallel world size")
        self._world, self._rank, self._group = world, rank, group
        taps = self.taps[:, :, self._columns()] if column_taps is None else column_taps
        n, _ = fft_segments(self.length, taps.shape[0])
        scale = self.lag_scale(taps.device)
        self._W = torch.empty((n // 2 + 1,) + tuple(taps.shape[1:]), device=taps.device, dtype=torch.complex64)
        for start in range(0, taps.shape[2], self.chunk):
            part = slice(start, start + self.chunk)
            self._W[:, :, part] = torch.fft.rfft(taps[:, :, part].float() * scale, n=n, dim=0)
        self._grad = torch.zeros_like(self._W)

    def end_step(self):
        """Gradient of this rank's output columns of the taps, [K, c, c/world], as the mean over ranks (like DDP):
        the rfft adjoint of the step's accumulated response gradient, a chunk of columns at a time; q38.longconv_train
        regroups it by lags for the sharded optimizer."""
        K = self.taps.shape[0]
        n, _ = fft_segments(self.length, K)
        response_grad, self._W, self._grad = self._grad, None, None
        _, c, width = response_grad.shape
        grad = torch.empty((K, c, width), device=response_grad.device, dtype=torch.float32)
        scale = self.lag_scale(response_grad.device)
        with torch.enable_grad():   # the transform is linear: any value of t gives its adjoint
            for start in range(0, width, self.chunk):
                part = slice(start, start + self.chunk)
                t = torch.zeros((K, c, grad[:, :, part].shape[2]), device=grad.device, requires_grad=True)
                grad[:, :, part] = torch.autograd.grad(torch.fft.rfft(t * scale, n=n, dim=0), t, response_grad[:, :, part])[0]
        return grad.div_(self._world)

    def forward(self, x):
        u, g = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        with torch.autocast(u.device.type, enabled=False):
            if self._W is not None and self.training and torch.is_grad_enabled():
                if u.shape[1] != self.length:
                    raise ValueError("LongConv step cache was built for another sequence length")
                z = _StepFFTConv.apply(u.float(), self)
            else:
                if self.taps.device != u.device:
                    raise RuntimeError("LongConv taps are on another device (q38.longconv_train keeps them on the CPU "
                                       "outside gather_taps)")
                z = FFTConv.apply(u.float(), self.taps, self.lag_scale(), self.chunk)
        return self.out_proj(z.to(g.dtype) * F.silu(g))


def longconv_layer_ids(config):
    return [int(i) for i in config.longconv_layers.split(",")] if getattr(config, "longconv_layers", "") else []


class PaperDepthWeightedAverage(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.weights=nn.ParameterList([nn.Parameter(torch.cat([torch.zeros(i+1),torch.ones(1)])) for i in range(layers)])

    def mix(self,index,raw):
        if len(raw)!=index+2:raise ValueError('Expected embedding plus each raw block output')
        weights=self.weights[index]
        # No softmax/normalization: paper coefficients are independent learned scalars.
        result=raw[-1]*weights[-1].to(raw[-1].dtype)
        for value,weight in zip(raw[:-1],weights[:-1]):result=result+value*weight.to(value.dtype)
        return result


class Qwen3_8MoeTextModel(Qwen3_8MoePreTrainedModel):
    config: Qwen3_8MoeTextConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen3_8MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_8MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_8MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()
        h, L = config.hidden_size, config.num_hidden_layers
        self.mhc = nn.ModuleList([MHC(h, config.n_branch, max(1, round(h / 8)), slot=i,
                                      iters=config.mhc_iters) for i in range(2 * L)]) if config.mhc_enabled else nn.ModuleList()
        self.indexers = nn.ModuleDict({str(i): Indexer(h, config.indexer_heads, config.indexer_head_dim)
                                     for i, kind in enumerate(config.layer_types) if kind == "full_attention" and config.qsa_mode == "learned"})
        self.ngram = ((PaperEngram(h, config.vocab_size, config.ngram_rows, config.eos_token_id,
                          config.paper_hash_heads, config.paper_hash_dim) if config.paper_engram else
              NGramEmbedding(h, config.vocab_size, config.ngram_rows, config.eos_token_id))
              if config.ngram_rows else None)
        layers = longconv_layer_ids(config)
        if layers and (config.mhc_enabled or not all(0 <= i < L for i in layers) or len(set(layers)) != len(layers)
                       or config.longconv_channels < 1 or config.longconv_taps < 1):
            raise ValueError("LongConv needs mHC disabled, distinct valid layer indices, channels and taps")
        self.longconv = nn.ModuleDict({str(i): LongConv(h, config.longconv_channels, config.longconv_taps, config.rms_norm_eps,
                                                        config.max_position_embeddings) for i in layers})
        self.depth_average = PaperDepthWeightedAverage(L) if config.paper_dwa else None
        for extra in (self.mhc, self.indexers, self.ngram, self.longconv, self.depth_average):
            if extra is not None:
                for module in extra.modules():
                    module._is_hf_initialized = True
        self.qsa_dense = False

    def trunk(self, ids):
        # Packed prefill only. Right padding also covers short validation tails.
        length = ids.shape[1]
        if not length:
            raise ValueError("Q38 requires at least one input token")
        padding = (-length) % BLK if self.config.qsa_mode == "learned" else 0
        if padding:
            ids = F.pad(ids, (0, padding), value=self.config.pad_token_id or 0)
        e = self.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, ids.shape[0], -1)
        cos, sin = self.rotary_emb(e, positions)
        if self.config.qsa_mode == "learned" and self.config.indexer_head_dim != self.config.head_dim:
            raise ValueError("Indexer and attention head dimensions must match for shared partial RoPE")
        mhc_enabled = self.config.mhc_enabled
        X = (e.float().unsqueeze(2).expand(-1, -1, self.config.n_branch, -1).contiguous()
             if mhc_enabled else e.float())
        raw_depth = [X] if self.depth_average is not None else None
        kls = []
        for i, layer in enumerate(self.layers):
            xin, state = self.mhc[2*i].pre(X) if mhc_enabled else (X, None)
            x = layer.input_layernorm(xin).to(e.dtype)
            if self.config.layer_types[i] == "linear_attention":
                y = layer.linear_attn(hidden_states=x, cache_params=None, attention_mask=None)
            else:
                full_attention = self.config.qsa_mode == "full"
                y, kl = qsa_attention(layer.self_attn, x, cos, sin,
                                      None if full_attention else self.indexers[str(i)],
                                      self.config.qsa_blocks, full_attention or (self.training and self.qsa_dense),
                                      not full_attention and self.training and torch.is_grad_enabled())
                if kl is not None:
                    kls.append(kl)
            X = self.mhc[2*i].post(X, y, state) if mhc_enabled else X + y.float()
            if str(i) in self.longconv:
                X = X + self.longconv[str(i)](X.to(e.dtype)).float()
            xin, state = self.mhc[2*i+1].pre(X) if mhc_enabled else (X, None)
            y = layer.mlp(layer.post_attention_layernorm(xin).to(e.dtype))
            X = self.mhc[2*i+1].post(X, y, state) if mhc_enabled else X + y.float()
            if i == 0 and self.ngram is not None:
                delta = self.ngram.delta((X.mean(2) if mhc_enabled else X).to(e.dtype), ids)
                X = X + (delta.unsqueeze(2) if mhc_enabled else delta)
            if raw_depth is not None:
                raw_depth.append(X)
                X = self.depth_average.mix(i, raw_depth)
        hs = X.sum(2) if mhc_enabled else X
        return hs[:, :length].to(e.dtype), kls

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=False, **kwargs):
        if input_ids is None or inputs_embeds is not None:
            raise ValueError("Q38 packed prefill requires input_ids")
        if past_key_values is not None or use_cache or attention_mask is not None:
            raise ValueError("Q38 supports uncached packed prefill without attention_mask")
        hs, _ = self.trunk(input_ids)
        return Qwen3_8MoeModelOutputWithPast(last_hidden_state=self.norm(hs))


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # Accumulate assignment counts and probability sums layer by layer, normalizing at the end,
    # so peak memory stays O(seq_len * num_experts) regardless of the number of layers.
    compute_device = gate_logits[0].device
    tokens_per_expert_sum = torch.zeros(num_experts, dtype=torch.float32, device=compute_device)
    router_prob_sum = torch.zeros(num_experts, dtype=torch.float32, device=compute_device)
    total_rows = 0.0

    if attention_mask is not None:
        # The same flat mask applies to every layer's [batch_size * sequence_length] rows.
        flat_mask = attention_mask.reshape(-1).to(device=compute_device, dtype=torch.float32)

    for layer_gate in gate_logits:
        routing_weights = torch.nn.functional.softmax(layer_gate.to(compute_device), dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if attention_mask is None:
            # Count of top-k assignments per expert
            # scatter_add rather than bincount: no host synchronization, and integer counts are exact in FP32
            flat = selected_experts.reshape(-1)
            tokens_per_expert_sum = tokens_per_expert_sum + torch.zeros(
                num_experts, dtype=torch.float32, device=compute_device
            ).scatter_add_(0, flat, torch.ones(flat.shape, dtype=torch.float32, device=compute_device))
            # Sum of routing probabilities per expert
            router_prob_sum = router_prob_sum + routing_weights.float().sum(dim=0)
            total_rows = total_rows + routing_weights.shape[0]
        else:
            # Same reductions, weighted by the attention mask to exclude padding tokens
            tokens_per_expert_sum = tokens_per_expert_sum + torch.zeros(
                num_experts, dtype=torch.float32, device=compute_device
            ).scatter_add_(0, selected_experts.reshape(-1), flat_mask.repeat_interleave(top_k))
            router_prob_sum = router_prob_sum + (routing_weights.float() * flat_mask.unsqueeze(-1)).sum(dim=0)
            total_rows = total_rows + flat_mask.sum()

    tokens_per_expert = tokens_per_expert_sum / total_rows
    router_prob_per_expert = router_prob_sum / total_rows

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Qwen3_8MoeForCausalLM(Qwen3_8MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}
    config: Qwen3_8MoeTextConfig
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_8MoeTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.mtp = None
        if config.mtp:
            mtp_config = copy.deepcopy(config)
            mtp_config.layer_types = list(config.layer_types) + ["linear_attention"]
            h = config.hidden_size
            self.mtp = nn.ModuleDict(dict(
                enorm=Qwen3_8MoeRMSNorm(h, eps=config.rms_norm_eps),
                hnorm=Qwen3_8MoeRMSNorm(h, eps=config.rms_norm_eps),
                proj=nn.Linear(2*h, h, bias=False),
                block=Qwen3_8MoeDecoderLayer(mtp_config, config.num_hidden_layers),
                norm=Qwen3_8MoeRMSNorm(h, eps=config.rms_norm_eps)))

        # Initialize weights and apply final processing
        self.post_init()

    def mtp_hidden(self, hs, ids_next):
        m = self.mtp
        z = m["proj"](torch.cat([m["enorm"](self.model.embed_tokens(ids_next)), m["hnorm"](hs)], -1))
        z = m["block"](z, position_embeddings=None, attention_mask=None)
        return m["norm"](z)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        """Main-head packed prefill; training auxiliary losses live in q38.train.LossModel."""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        ) and self.num_experts > 0

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "Qwen3_8MoeTextModel",
    "Qwen3_8MoeForCausalLM",
    "Qwen3_8MoePreTrainedModel",
]
