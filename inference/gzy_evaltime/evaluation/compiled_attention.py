"""Tiled D64 full-prefill Q/K RMSNorm + partial RoPE + Q gate packing."""
import os
import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _norm_rope(X,W,COS,SIN,Q,G,N:tl.constexpr,S:tl.constexpr,H:tl.constexpr,R:tl.constexpr,
               CB:tl.constexpr,CS:tl.constexpr,SB:tl.constexpr,SS:tl.constexpr,
               EPS:tl.constexpr,HAS_GATE:tl.constexpr,BT:tl.constexpr,XS:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)*BT+tl.arange(0,BT);d=tl.arange(0,64)
    token=row//H;b=token//S;t=token%S
    step=128 if HAS_GATE else 64
    source=token*XS+(row%H)*step
    x=tl.load(X+source[:,None]+d[None,:],row[:,None]<N,0.).to(tl.float32)
    w=tl.load(W+d).to(tl.float32)
    inv=libdevice.rsqrt(tl.sum(x*x,1)/64.+EPS)
    norm=(x*inv[:,None]*(1.+w[None,:])).to(tl.bfloat16).to(tl.float32)
    idx=tl.where(d<R,(d+R//2)%R,d)
    idx=tl.broadcast_to(idx[None,:],(BT,64))
    rotated=tl.gather(norm,idx,1)*tl.where(d[None,:]<R//2,-1.,1.)
    c=tl.load(COS+b[:,None]*CB+t[:,None]*CS+d[None,:],(row<N)[:,None]&(d<R)[None,:],0.).to(tl.float32)
    s=tl.load(SIN+b[:,None]*SB+t[:,None]*SS+d[None,:],(row<N)[:,None]&(d<R)[None,:],0.).to(tl.float32)
    a=(norm*c).to(tl.bfloat16).to(tl.float32)
    z=(rotated*s).to(tl.bfloat16).to(tl.float32)
    y=(a+z).to(tl.bfloat16)
    tl.store(Q+row[:,None]*64+d[None,:],tl.where((d<R)[None,:],y,norm),row[:,None]<N)
    if HAS_GATE:
        gate=tl.load(X+source[:,None]+64+d[None,:],row[:,None]<N,0.)
        tl.store(G+row[:,None]*64+d[None,:],gate,row[:,None]<N)

def norm_rope(x,weight,eps,cos,sin,has_gate):
    b,s,h,width=x.shape
    # Token-major rows, possibly a column slice of a wider projection output.
    assert width==(128 if has_gate else 64) and x.stride(3)==1 and x.stride(2)==width
    assert x.stride(0)==s*x.stride(1)
    assert cos.shape==sin.shape and cos.shape[:2]==(b,s) and cos.shape[-1] in (16,32,64)
    cos=cos.to(torch.bfloat16);sin=sin.to(torch.bfloat16)
    q=torch.empty((b,s,h,64),device=x.device,dtype=torch.bfloat16)
    gate=torch.empty((b,s,h*64),device=x.device,dtype=torch.bfloat16) if has_gate else None
    bt=int(os.environ.get('Q38_ROPE_BT','16'));warps=8 if bt>=32 else 4
    _norm_rope[(triton.cdiv(b*s*h,bt),)](x,weight,cos,sin,q,gate if has_gate else q,b*s*h,s,h,cos.shape[-1],
        cos.stride(0),cos.stride(1),sin.stride(0),sin.stride(1),eps,has_gate,bt,x.stride(1),num_warps=warps,enable_fp_fusion=False)
    return q.transpose(1,2),gate

def _attention_piece(attn,x,cos,sin,indexer,budget_blocks,dense,want_kl):
    from full_compile import cudnn_attention
    from infra_attention import project
    assert indexer is None and dense and not want_kl and attn.head_dim==64
    b,s,_=x.shape
    if os.environ.get('Q38_ATTN_MERGED_PROJ','0')=='1':
        from prequant_norm import quant_rows
        q8,scale=quant_rows(x.reshape(-1,x.shape[-1]))
        y=torch._scaled_mm(q8,attn._merged_q8_weight.t(),scale_a=scale[:,None],
            scale_b=attn._merged_q8_scale[None,:],out_dtype=torch.bfloat16,use_fast_accum=True)
        (q0,q1),(k0,k1),(v0,v1)=attn._merged_slices
        qraw=y[:,q0:q1].view(b,s,-1,128);kraw=y[:,k0:k1].view(b,s,-1,64)
        v=y[:,v0:v1].view(b,s,-1,64).transpose(1,2)
    else:
        qraw=attn.q_proj(x).view(b,s,-1,128)
        kraw=attn.k_proj(x).view(b,s,-1,64)
        v=attn.v_proj(x).view(b,s,-1,64).transpose(1,2)
    q,gate=norm_rope(qraw,attn.q_norm.weight,attn.q_norm.eps,cos,sin,True)
    k,_=norm_rope(kraw,attn.k_norm.weight,attn.k_norm.eps,cos,sin,False)
    o=cudnn_attention(q,k,v,attn.scaling)
    return project(attn,o,gate),None


def attention(attn,x,cos,sin,indexer,budget_blocks,dense,want_kl):
    """Bound Q/gate liveness by completing each independent batch tile."""
    limit=int(os.environ.get('Q38_ATTENTION_BATCH_LIMIT','512'))
    if limit<0:
        raise ValueError('Q38_ATTENTION_BATCH_LIMIT must be nonnegative')
    batch=x.shape[0]
    if limit==0 or batch<=limit:
        return _attention_piece(attn,x,cos,sin,indexer,budget_blocks,dense,want_kl)
    assert cos.shape[0]==sin.shape[0]==batch
    parts=[]
    for start in range(0,batch,limit):
        stop=min(start+limit,batch)
        output,_=_attention_piece(attn,x[start:stop],cos[start:stop],sin[start:stop],
                                  indexer,budget_blocks,dense,want_kl)
        parts.append(output)
    return torch.cat(parts,dim=0),None


def merge_projections(attn):
    """Concatenate the cached row-wise FP8 q/k/v projections of one attention module."""
    parts=[attn.q_proj,attn.k_proj,attn.v_proj]
    assert all(p._q8_bias is None and p._q8_weight.shape[0]==p.out_features for p in parts)
    weight=torch.cat([p._q8_weight for p in parts],0);scale=torch.cat([p._q8_weight_scale for p in parts],0)
    attn.register_buffer('_merged_q8_weight',weight,persistent=False)
    attn.register_buffer('_merged_q8_scale',scale,persistent=False)
    starts=[0,parts[0].out_features,parts[0].out_features+parts[1].out_features]
    attn._merged_slices=[(start,start+p.out_features) for start,p in zip(starts,parts)]
    for start,p in zip(starts,parts):
        p._q8_weight=weight[start:start+p.out_features];p._q8_weight_scale=scale[start:start+p.out_features]
    return weight.shape[0]


def install(model):
    count=0
    if os.environ.get('Q38_ATTN_MERGED_PROJ','0')=='1':
        for layer in model.model.layers:
            if hasattr(layer,'self_attn'):
                merge_projections(layer.self_attn);count+=1
    return dict(merged_qkv_modules=count)
