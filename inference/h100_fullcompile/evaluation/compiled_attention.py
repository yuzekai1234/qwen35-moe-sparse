"""Tiled D64 full-prefill Q/K RMSNorm + partial RoPE + Q gate packing."""
import os
import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _norm_rope(X,W,COS,SIN,Q,G,N:tl.constexpr,S:tl.constexpr,H:tl.constexpr,R:tl.constexpr,
               CB:tl.constexpr,CS:tl.constexpr,SB:tl.constexpr,SS:tl.constexpr,
               EPS:tl.constexpr,HAS_GATE:tl.constexpr,BT:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)*BT+tl.arange(0,BT);d=tl.arange(0,64)
    token=row//H;b=token//S;t=token%S
    step=128 if HAS_GATE else 64
    x=tl.load(X+row[:,None]*step+d[None,:],row[:,None]<N,0.).to(tl.float32)
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
        gate=tl.load(X+row[:,None]*128+64+d[None,:],row[:,None]<N,0.)
        tl.store(G+row[:,None]*64+d[None,:],gate,row[:,None]<N)

def norm_rope(x,weight,eps,cos,sin,has_gate):
    b,s,h,width=x.shape
    assert width==(128 if has_gate else 64) and x.is_contiguous()
    assert cos.shape==sin.shape and cos.shape[:2]==(b,s) and cos.shape[-1] in (16,32,64)
    cos=cos.to(torch.bfloat16);sin=sin.to(torch.bfloat16)
    q=torch.empty((b,s,h,64),device=x.device,dtype=torch.bfloat16)
    gate=torch.empty((b,s,h*64),device=x.device,dtype=torch.bfloat16) if has_gate else None
    bt=int(os.environ.get('Q38_ROPE_BT','16'));warps=8 if bt>=32 else 4
    _norm_rope[(triton.cdiv(b*s*h,bt),)](x,weight,cos,sin,q,gate if has_gate else q,b*s*h,s,h,cos.shape[-1],
        cos.stride(0),cos.stride(1),sin.stride(0),sin.stride(1),eps,has_gate,bt,num_warps=warps,enable_fp_fusion=False)
    return q.transpose(1,2),gate

def attention(attn,x,cos,sin,indexer,budget_blocks,dense,want_kl):
    from full_compile import cudnn_attention
    from infra_attention import project
    assert indexer is None and dense and not want_kl and attn.head_dim==64
    b,s,_=x.shape
    qraw=attn.q_proj(x).view(b,s,-1,128)
    q,gate=norm_rope(qraw,attn.q_norm.weight,attn.q_norm.eps,cos,sin,True)
    kraw=attn.k_proj(x).view(b,s,-1,64)
    k,_=norm_rope(kraw,attn.k_norm.weight,attn.k_norm.eps,cos,sin,False)
    v=attn.v_proj(x).view(b,s,-1,64).transpose(1,2)
    o=cudnn_attention(q,k,v,attn.scaling)
    return project(attn,o,gate),None
