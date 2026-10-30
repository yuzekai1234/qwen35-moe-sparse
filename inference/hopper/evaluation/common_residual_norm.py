"""Inference-only add/RMSNorm/BF16 fusion for an ordinary FP32 residual stream."""
import sys
import types
import torch


def _add_norm(residual, pending, weight, eps):
    if pending is not None:
        residual = residual + pending.float()
    value = residual * torch.rsqrt(residual.pow(2).mean(-1, keepdim=True) + eps)
    value = value * (1.0 + weight.float())
    return residual, value.to(torch.bfloat16)


def _finish(residual, pending):
    return (residual if pending is None else residual + pending.float()).to(torch.bfloat16)


_compiled = torch.compile(_add_norm, fullgraph=True, dynamic=True,
                          options={'emulate_precision_casts': True})
_finish_compiled = torch.compile(_finish, fullgraph=True, dynamic=True,
                                 options={'emulate_precision_casts': True})


def _trunk(self, ids):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Residual norm fusion requires eval and inference_mode')
    if ids.ndim != 2 or ids.shape[1] < 1:
        raise ValueError('Expected a nonempty packed token matrix')
    embedding = self.embed_tokens(ids)
    positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, ids.shape[0], -1)
    cos, sin = self.rotary_emb(embedding, positions)
    residual = embedding.float()
    del embedding, positions
    pending = None
    for index, layer in enumerate(self.layers):
        residual, x = _compiled(residual, pending, layer.input_layernorm.weight, layer.input_layernorm.eps)
        pending = None
        if self.config.layer_types[index] == 'linear_attention':
            y = layer.linear_attn(hidden_states=x, cache_params=None, attention_mask=None)
        else:
            y, _ = self._common_full_qsa(layer.self_attn, x, cos, sin, None,
                                         self.config.qsa_blocks, True, False)
        del x
        residual, x = _compiled(residual, y, layer.post_attention_layernorm.weight,
                                layer.post_attention_layernorm.eps)
        del y
        pending = layer.mlp(x)
        del x
        if index == 0 and self.ngram is not None:
            # Keep the exact delta() operation, including its BF16 subtraction;
            # replacing it with gate*embedding would change rounding.
            residual = residual + pending.float()
            pending = None
            delta = self.ngram.delta(residual.to(torch.bfloat16), ids)
            residual = residual + delta
            del delta
    return _finish_compiled(residual, pending), []


def install(model):
    config = model.config
    if model.training or config.num_experts != 0 or config.mhc_enabled or config.qsa_mode != 'full':
        raise ValueError('Qualified path requires eval Dense checkpoint, no mHC, full attention')
    if model.model.embed_tokens.weight.dtype != torch.bfloat16:
        raise ValueError('Qualified parameter dtype is BF16 with FP32 residuals')
    implementation = sys.modules[type(model).__module__]
    trunk = model.model
    trunk._common_full_qsa = implementation.qsa_attention
    trunk.trunk = types.MethodType(_trunk, trunk)
    return dict(implementation='compiled_add_RMSNorm_BF16_cast', residual_storage='original_FP32',
                residual_addition_order='unchanged', normalization='original_1_plus_weight_FP32',
                ngram_delta_rounding='preserved', expert_or_router_changes=False,
                inference_only=True, checkpoint_parameters_changed=False)
