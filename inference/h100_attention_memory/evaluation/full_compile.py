"""Full C1 prefill graph; vendor/library GPU kernels remain explicit extern ops."""
import inspect
import os
from pathlib import Path
import types
import torch
from fla.modules.conv.causal_conv1d import causal_conv1d as _original_conv
from gdn_flashqla_backend import run as _original_core


@torch.library.custom_op('q38infra::causal_conv', mutates_args=())
def _conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
          activation: str) -> torch.Tensor:
    output, _ = _original_conv(x, weight=weight, bias=bias, activation=activation)
    return output


@_conv.register_fake
def _(x, weight, bias, activation):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op('q38infra::flashqla', mutates_args=())
def _core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
          g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    output, _ = _original_core(q, k, v, g=g, beta=beta,
                               use_qk_l2norm_in_kernel=False)
    return output


@_core.register_fake
def _(q, k, v, g, beta):
    return torch.empty_like(v, memory_format=torch.contiguous_format)


def causal_conv1d(x, weight, bias=None, activation=None, **kwargs):
    assert not kwargs and activation in ('silu', 'swish')
    return _conv(x, weight, bias, activation), None


def core(q, k, v, g, beta, scale=None, initial_state=None,
         output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    assert scale is None and initial_state is None and not output_final_state
    assert not use_qk_l2norm_in_kernel and not kwargs
    return _core(q, k, v, g, beta), None


@torch.library.custom_op('q38infra::cudnn_attention', mutates_args=())
def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               scale: float) -> torch.Tensor:
    from cudnn_attention import attention
    return attention(q, k, v, scale)


@_attention.register_fake
def _(q, k, v, scale):
    return torch.empty_like(q)


def cudnn_attention(q, k, v, scale):
    # cuDNN preserves input layout. Pin BSHD storage and retain native GQA.
    q = q.transpose(1, 2).contiguous().transpose(1, 2)
    k = k.transpose(1, 2).contiguous().transpose(1, 2)
    v = v.transpose(1, 2).contiguous().transpose(1, 2)
    return _attention(q, k, v, scale)


def functional_gdn(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    assert cache_params is None and attention_mask is None and not kwargs
    from dev_optimizations import _piece
    batch = hidden_states.shape[0]
    if batch <= 256:
        return _piece(self, hidden_states)
    return torch.cat([_piece(self, hidden_states[start:start + 256])
                      for start in range(0, batch, 256)], 0)


def prepare(model):
    import dev_optimizations
    import gdn_flashqla_backend
    from compiled_attention import attention
    assert not model.training and not model.config.num_experts
    assert not model.config.mhc_enabled and model.config.qsa_mode == 'full'
    for layer in model.model.layers:
        if hasattr(layer, 'linear_attn'):
            layer.linear_attn.forward = types.MethodType(functional_gdn, layer.linear_attn)
    source = inspect.getsource(dev_optimizations._piece)
    needle = 'from fla.modules.conv.causal_conv1d import causal_conv1d'
    assert source.count(needle) == 1
    source = source.replace(needle, 'from full_compile import causal_conv1d')
    exec(compile(source, dev_optimizations.__file__ + ':full_compile_conv', 'exec'),
         dev_optimizations.__dict__)
    gdn_flashqla_backend.run = core
    model.model._common_full_qsa = attention

    def prefill(ids):
        hidden, _ = model.model.trunk(ids)
        return model.lm_head(model.model.norm(hidden))

    return prefill


def install(model, forward, paths):
    if paths['variant'] != 'general_dense_fused_norm':
        raise ValueError('Full compile release requires the qualified C1 FP8/BF16 path')
    forward = prepare(model)
    options = dict(emulate_precision_casts=True, **{
        'triton.cudagraphs': False,
        'reorder_for_peak_memory': False,
        'layout_optimization': False,
        # Retaining old large buffers for a later reuse overlaps FFN and
        # attention allocations and prevents batch942 fitting on H100 80GB.
        'allow_buffer_reuse': False,
    })
    info = dict(fullgraph=True, dynamic=False, cudagraphs=False,
                silent_eager_fallback=False, functional_gdn=True,
                fused_qk_norm_rope_gate_pack=True,
                rope_token_head_tile=int(os.environ.get('Q38_ROPE_BT', '16')),
                attention_batch_limit=int(os.environ.get('Q38_ATTENTION_BATCH_LIMIT', '512')),
                attention_memory_strategy='batch_tiled_projection_through_output',
                external_ops=['q38infra::causal_conv', 'q38infra::flashqla',
                              'q38infra::cudnn_attention'], graphs=[], options=options)

    def backend(graph, inputs, **kwargs):
        shapes = [list(x.shape) for x in inputs
                  if isinstance(x, torch.Tensor) and x.dtype == torch.int64 and x.ndim == 2]
        info['graphs'].append(dict(nodes=len(list(graph.graph.nodes)), input_token_shapes=shapes))
        evidence = Path(os.environ.get('Q38_COMPILE_EVIDENCE', 'compile-evidence'))
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / f'graph-{len(info["graphs"])}.py').write_text(graph.code)
        return torch._inductor.compile(graph, inputs, options=options)

    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    paths['full_compile'] = info
    paths['variant_details']['development']['gdn'].update(
        direct_out=False, functional_output_assembly=True)
    return torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
