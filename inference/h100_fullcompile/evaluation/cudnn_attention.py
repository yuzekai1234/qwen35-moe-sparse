"""Force PyTorch's cuDNN fused causal GQA while retaining qualified output fusion."""
import inspect
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend


def attention(q,k,v,scale):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(q,k,v,scale=scale,is_causal=True,enable_gqa=True)


def install(model):
    assert model.config.num_experts==0 and model.config.qsa_mode=='full'
    original=sys.modules[type(model).__module__].qsa_attention
    source=inspect.getsource(original)
    old='o = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling, is_causal=True, enable_gqa=True)'
    assert source.count(old)==1
    source=source.replace(old,'o = _infra_cudnn_attention(q, k, v, attn.scaling)')
    tail='o = o.transpose(1, 2).reshape(B, S, -1) * torch.sigmoid(gate)\n    return attn.o_proj(o), kl'
    assert source.count(tail)==1
    source=source.replace(tail,'return _infra_attention_project(attn, o, gate), kl')
    from infra_attention import project
    namespace=dict(original.__globals__)
    namespace.update(_infra_cudnn_attention=attention,_infra_attention_project=project)
    exec(compile(source,__file__+':qualified_full_attention','exec'),namespace)
    model.model._common_full_qsa=namespace[original.__name__]
    return dict(implementation='PyTorch_cuDNN_fused_BF16_causal_GQA',fallback_allowed=False,
                output_fusion=True,checkpoint_parameters_changed=False,expert_or_router_changes=False)
