"""Fuse residual/RMSNorm/FP8 input production for ordinary Dense projections."""
import os
import types
import torch
import triton
import triton.language as tl
from low_precision import quant_rows as original_quant_rows
from common_residual_norm import _compiled, _finish_compiled


class FP8Activation:
    def __init__(self,q,scale,shape):
        self.q=q;self.scale=scale;self.shape=tuple(shape);self.device=q.device

    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],(tuple,list)):shape=tuple(shape[0])
        shape=list(shape)
        if -1 in shape:
            n=self.q.numel()
            for v in shape:
                if v!=-1:n//=v
            shape[shape.index(-1)]=n
        assert shape[-1]==self.q.shape[-1]
        return FP8Activation(self.q,self.scale,shape)

    def __getitem__(self,index):
        assert len(self.shape)==3
        b,s,h=self.shape
        q=self.q.reshape(b,s,h)[index]
        scale=self.scale.reshape(b,s)[index]
        return FP8Activation(q.reshape(-1,h),scale.reshape(-1),q.shape)


def quant_rows(x):
    if isinstance(x,FP8Activation):return x.q,x.scale
    return original_quant_rows(x)


@triton.jit
def _kernel(R,P,W,OUT,Q,S,K:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr,HAS_PENDING:tl.constexpr,ROUND_RESIDUAL:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);col=tl.arange(0,BLOCK)
    r=tl.load(R+row*K+col,col<K,other=0).to(tl.float32)
    if HAS_PENDING:
        p=tl.load(P+row*K+col,col<K,other=0).to(tl.float32)
        value=r+p
    else:
        value=r
    if ROUND_RESIDUAL:value=value.to(tl.bfloat16).to(tl.float32)
    variance=tl.sum(value*value,0)/K
    normalized=value*tl.rsqrt(variance+EPS)
    weight=tl.load(W+col,col<K,other=0).to(tl.float32)
    x=(normalized*(1.+weight)).to(tl.bfloat16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0)/448.,1.e-12)
    if HAS_PENDING:tl.store(OUT+row*K+col,value,col<K)
    tl.store(Q+row*K+col,tl.minimum(tl.maximum(x/scale,-448.),448.),col<K)
    tl.store(S+row,scale)


def produce(residual,pending,weight,eps):
    assert residual.dtype in (torch.float32,torch.bfloat16) and residual.is_contiguous()
    if pending is not None:assert pending.dtype==torch.bfloat16 and pending.is_contiguous()
    shape=tuple(residual.shape);width=shape[-1];rows=residual.numel()//width
    updated=torch.empty_like(residual) if pending is not None else residual
    q=torch.empty((rows,width),device=residual.device,dtype=torch.float8_e4m3fn)
    scale=torch.empty(rows,device=residual.device,dtype=torch.float32)
    _kernel[(rows,)](residual,pending if pending is not None else residual,weight,
        updated,q,scale,width,eps,triton.next_power_of_2(width),pending is not None,residual.dtype==torch.bfloat16,
        num_warps=4,enable_fp_fusion=False)
    return updated,FP8Activation(q,scale,shape)


def _add_store(residual,pending):
    return (residual.float()+pending.float()).to(residual.dtype)


_add_store_compiled=torch.compile(_add_store,fullgraph=True,dynamic=True,options={'emulate_precision_casts':True})


def validate_configuration():
    mode=os.environ.get('Q38_RESIDUAL_DTYPE','fp32')
    if mode not in ('fp32','bf16'):raise ValueError('Unknown residual dtype')
    if mode=='bf16' and any(os.environ.get(name,'1')!='1' for name in ('Q38_PREQUANT_NORM','Q38_PREQUANT_INPUT','Q38_PREQUANT_FFN')):
        raise ValueError('BF16 residual requires prequant NORM=1, INPUT=1 and FFN=1; fallback would promote residual to FP32')


def _trunk(self,ids):
    validate_configuration()
    if self.training or torch.is_grad_enabled():raise RuntimeError('Inference-only prequantized norm path')
    embedding=self.embed_tokens(ids)
    positions=torch.arange(ids.shape[1],device=ids.device).view(1,1,-1).expand(3,ids.shape[0],-1)
    cos,sin=self.rotary_emb(embedding,positions)
    mode=os.environ.get('Q38_RESIDUAL_DTYPE','fp32')
    if mode not in ('fp32','bf16'):raise ValueError('Unknown residual dtype')
    residual=embedding.to(torch.bfloat16 if mode=='bf16' else torch.float32);del embedding,positions
    pending=None
    use_inputs=os.environ.get('Q38_PREQUANT_INPUT','1')=='1'
    use_ffn=os.environ.get('Q38_PREQUANT_FFN','1')=='1'
    for index,layer in enumerate(self.layers):
        norm=produce if use_inputs else _compiled
        residual,x=norm(residual,pending,layer.input_layernorm.weight,layer.input_layernorm.eps)
        pending=None
        if self.config.layer_types[index]=='linear_attention':
            y=layer.linear_attn(hidden_states=x,cache_params=None,attention_mask=None)
        else:
            y,_=self._common_full_qsa(layer.self_attn,x,cos,sin,None,self.config.qsa_blocks,True,False)
        del x
        norm=produce if use_ffn else _compiled
        residual,x=norm(residual,y,layer.post_attention_layernorm.weight,layer.post_attention_layernorm.eps)
        del y
        pending=layer.mlp(x);del x
        if index==0 and self.ngram is not None:
            residual=_add_store_compiled(residual,pending);pending=None
            delta=self.ngram.delta(residual.to(torch.bfloat16),ids)
            residual=_add_store_compiled(residual,delta);del delta
    return _finish_compiled(residual,pending),[]


def install(model):
    validate_configuration()
    assert model.config.num_experts==0 and not model.config.mhc_enabled
    import common_gdn,fp8_linear,dense_ffn
    common_gdn.quant_rows=quant_rows
    fp8_linear.quant_rows=quant_rows
    dense_ffn.quant_rows=quant_rows
    model.model.trunk=types.MethodType(_trunk,model.model)
    return dict(implementation='residual_RMSNorm_BF16_rounding_FP8_producer_for_all_Dense_inputs',
                residual_storage=os.environ.get('Q38_RESIDUAL_DTYPE','fp32'),
                residual_addition='FP32_add_then_round_to_storage_dtype',norm_accumulation='FP32',
                experts_modified=False,router_modified=False,checkpoint_parameters_changed=False,
                full_attention_callback_preserved=True)
