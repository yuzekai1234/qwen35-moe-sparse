"""Existing FA3 only; no projection/norm/deferred-scale/quant-kernel rewrite."""
from pathlib import Path
import sys,torch
H=Path('/data/q38/experiments/c4-lowprecision-20261031/fp8-attention/round2/src/flash-attention-4616f5116d33e3ba3cbb71d582caffcfcb49649c/hopper')
sys.path.insert(0,str(H))
from flash_attn_interface import flash_attn_func
@torch.library.custom_op('q38fp8::fa3_fused_baseline_native',mutates_args=())
def native(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,qs:torch.Tensor,ks:torch.Tensor,vs:torch.Tensor,scale:float)->torch.Tensor:
    assert q.dtype==k.dtype==v.dtype==torch.float8_e4m3fn
    assert q.shape[-2:]==(44,64) and k.shape[-2:]==v.shape[-2:]==(2,64)
    out=flash_attn_func(q,k,v,q_descale=qs,k_descale=ks,v_descale=vs,causal=True,softmax_scale=scale,num_splits=1,pack_gqa=True)
    assert out.dtype==torch.bfloat16
    return out
@native.register_fake
def _(q,k,v,qs,ks,vs,scale):return torch.empty(q.shape,device=q.device,dtype=torch.bfloat16)
def quant(x):
    b,s,h,d=x.shape;g=x.reshape(b,s,2,h//2,d)
    descale=(g.float().abs().amax(dim=(1,3,4))/448).clamp_min(1e-12)
    return (g/descale[:,None,:,None,None]).to(torch.float8_e4m3fn).reshape(b,s,h,d),descale
def attention(q,k,v,scale):
    q8,qs=quant(q.transpose(1,2).contiguous());k8,ks=quant(k.transpose(1,2).contiguous());v8,vs=quant(v.transpose(1,2).contiguous())
    return native(q8,k8,v8,qs,ks,vs,scale).transpose(1,2)
def install():
    import full_compile
    full_compile.cudnn_attention=attention
