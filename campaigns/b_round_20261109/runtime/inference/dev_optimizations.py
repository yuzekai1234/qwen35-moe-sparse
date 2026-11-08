"""Opt-in C1 experiments. No checkpoint or installed dependency is modified."""
import os
import types
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_norm_qkv(Q,K,V,OQ,OK,OV, S:tl.constexpr,HQ:tl.constexpr,HV:tl.constexpr,
                     QB:tl.constexpr,QS:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr,
                     KB:tl.constexpr,KS:tl.constexpr,KH:tl.constexpr,KD:tl.constexpr,
                     VB:tl.constexpr,VS:tl.constexpr,VH:tl.constexpr,VD:tl.constexpr,
                     BT:tl.constexpr):
    t=(tl.program_id(0)*BT+tl.arange(0,BT)).to(tl.int64)
    h=tl.program_id(1);b=tl.program_id(2).to(tl.int64);d=tl.arange(0,64)
    if h < HQ:
        q=tl.load(Q+b*QB+t[:,None]*QS+h*QH+d[None,:]*QD,t[:,None]<S,0).to(tl.float32)
        k=tl.load(K+b*KB+t[:,None]*KS+h*KH+d[None,:]*KD,t[:,None]<S,0).to(tl.float32)
        qi=1./tl.sqrt(tl.sum(q*q,1)+1.e-6)
        ki=1./tl.sqrt(tl.sum(k*k,1)+1.e-6)
        off=((b*S+t[:,None])*HQ+h)*64+d[None,:]
        tl.store(OQ+off,q*qi[:,None],t[:,None]<S)
        tl.store(OK+off,k*ki[:,None],t[:,None]<S)
    v=tl.load(V+b*VB+t[:,None]*VS+h*VH+d[None,:]*VD,t[:,None]<S,0)
    tl.store(OV+((b*S+t[:,None])*HV+h)*64+d[None,:],v,t[:,None]<S)


def gather_norm_qkv(q,k,v):
    assert q.shape[-1]==k.shape[-1]==v.shape[-1]==64
    b,s,h,_=q.shape;hv=v.shape[2]
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    ok=torch.empty(k.shape,device=k.device,dtype=k.dtype)
    ov=torch.empty(v.shape,device=v.device,dtype=v.dtype)
    bt=int(os.environ.get('Q38_DEV_NORM_BT','16'))
    _gather_norm_qkv[(triton.cdiv(s,bt),hv,b)](q,k,v,oq,ok,ov,s,h,hv,*q.stride(),*k.stride(),*v.stride(),bt,
        num_warps=4,enable_fp_fusion=False)
    return oq,ok,ov


def _project_out(linear,x,out):
    from prequant_norm import quant_rows
    shape=x.shape;q,scale=quant_rows(x.reshape(-1,shape[-1]))
    flat=out.reshape(-1,linear.out_features)
    assert flat.is_contiguous() and linear.out_features%16==0
    torch.ops.aten._scaled_mm.out(q,linear._q8_weight.t(),scale[:,None],linear._q8_weight_scale[None,:],
        bias=linear._q8_bias,out_dtype=torch.bfloat16,use_fast_accum=True,out=flat)
    return out


def _piece(self,x,out=None):
    from common_gdn import _project,quant_rows
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from gdn_flashqla_backend import run
    b,s,_=x.shape;q,scale=quant_rows(x.reshape(-1,x.shape[-1]))
    projected=_project(self.in_proj_qkv,q,scale,b,s)
    mixed,_=causal_conv1d(projected,weight=self.conv1d.weight.squeeze(1),bias=self.conv1d.bias,activation=self.activation)
    del projected
    query,key,value=torch.split(mixed,[self.key_dim,self.key_dim,self.value_dim],-1)
    query=query.reshape(b,s,-1,self.head_k_dim);key=key.reshape(b,s,-1,self.head_k_dim);value=value.reshape(b,s,-1,self.head_v_dim)
    z=_project(self.in_proj_z,q,scale,b,s)
    beta=_project(self.in_proj_b,q,scale,b,s).sigmoid()
    a=_project(self.in_proj_a,q,scale,b,s);del q,scale
    g=-self.A_log.float().exp()*torch.nn.functional.softplus(a.float()+self.dt_bias);del a
    fused=os.environ.get('Q38_DEV_QKV_NORM')=='1'
    if fused:query,key,value=gather_norm_qkv(query,key,value)
    output,_=run(query,key,value,g=g,beta=beta,use_qk_l2norm_in_kernel=not fused)
    del query,key,value,mixed,g,beta
    output=self.norm(output.reshape(-1,self.head_v_dim),z.reshape(-1,self.head_v_dim));del z
    output=output.reshape(b,s,-1)
    return _project_out(self.out_proj,output,out) if out is not None else self.out_proj(output)


def _gdn_forward(self,hidden_states,cache_params=None,attention_mask=None,**kwargs):
    if cache_params is not None or attention_mask is not None or kwargs.get('cu_seq_lens_q') is not None:
        return self._common_original_gdn(hidden_states,cache_params=cache_params,attention_mask=attention_mask,**kwargs)
    if self.training or torch.is_grad_enabled():raise RuntimeError('C1 optimization is inference-only')
    b,s,_=hidden_states.shape;limit=int(os.environ.get('Q38_GDN_BATCH_LIMIT','64'))
    if not 1<=limit<=256:raise ValueError('Qualified GDN tile range is 1..256')
    if os.environ.get('Q38_DEV_DIRECT_OUT')=='1':
        out=torch.empty((b,s,self.out_proj.out_features),device=hidden_states.device,dtype=torch.bfloat16)
        for start in range(0,b,limit):_piece(self,hidden_states[start:start+limit],out[start:start+limit])
        return out
    if b>limit:return torch.cat([_piece(self,hidden_states[start:start+limit]) for start in range(0,b,limit)],0)
    return _piece(self,hidden_states)


@triton.jit
def _fused_gateup_quant(X,XS,W,WS,Q,QS, M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,
                        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=(tl.program_id(0)*BM+tl.arange(0,BM)).to(tl.int64)
    n=tl.arange(0,BN);kk=tl.arange(0,BK)
    ag=tl.zeros((BM,BN),tl.float32);au=tl.zeros((BM,BN),tl.float32)
    for k0 in range(tl.cdiv(K,BK)):
        k=k0*BK+kk
        a=tl.load(X+m[:,None]*K+k[None,:],(m<M)[:,None]&(k<K)[None,:],0.0)
        wg=tl.load(W+n[None,:]*K+k[:,None],(n<N)[None,:]&(k<K)[:,None],0.0)
        wu=tl.load(W+(n[None,:]+N)*K+k[:,None],(n<N)[None,:]&(k<K)[:,None],0.0)
        ag=tl.dot(a,wg,ag,max_num_imprecise_acc=BK)
        au=tl.dot(a,wu,au,max_num_imprecise_acc=BK)
    sx=tl.load(XS+m,m<M,0)
    sg=tl.load(WS+n,n<N,0);su=tl.load(WS+n+N,n<N,0)
    g=(ag*sx[:,None]*sg[None,:]).to(tl.bfloat16).to(tl.float32)
    u=(au*sx[:,None]*su[None,:]).to(tl.bfloat16).to(tl.float32)
    silu=(g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    h=(silu*u).to(tl.bfloat16).to(tl.float32)
    h=tl.where(n[None,:]<N,h,0.)
    scale=tl.maximum(tl.max(tl.abs(h),1)/448.,1.e-12)
    tl.store(Q+m[:,None]*N+n[None,:],tl.minimum(tl.maximum(h/scale[:,None],-448.),448.),(m<M)[:,None]&(n<N)[None,:])
    tl.store(QS+m,scale,m<M)


@triton.jit
def _swiglu_quant_tiled(GU,Q,S, M:tl.constexpr,N:tl.constexpr,BT:tl.constexpr,BN:tl.constexpr):
    row=(tl.program_id(0)*BT+tl.arange(0,BT)).to(tl.int64);col=tl.arange(0,BN)
    gate=tl.load(GU+row[:,None]*(2*N)+col[None,:],(row<M)[:,None]&(col<N)[None,:],0.).to(tl.float32)
    up=tl.load(GU+row[:,None]*(2*N)+N+col[None,:],(row<M)[:,None]&(col<N)[None,:],0.).to(tl.float32)
    activated=(gate*tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    h=(activated*up).to(tl.bfloat16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(h),1)/448.,1.e-12)
    tl.store(Q+row[:,None]*N+col[None,:],tl.minimum(tl.maximum(h/scale[:,None],-448.),448.),(row<M)[:,None]&(col<N)[None,:])
    tl.store(S+row,scale,row<M)


@triton.jit
def _gateup_swiglu(X,XS,W,WS,H, M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=(tl.program_id(0)*BM+tl.arange(0,BM)).to(tl.int64)
    n=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    ag=tl.zeros((BM,BN),tl.float32);au=tl.zeros((BM,BN),tl.float32)
    for k0 in range(tl.cdiv(K,BK)):
        k=k0*BK+kk
        a=tl.load(X+m[:,None]*K+k[None,:],(m<M)[:,None]&(k<K)[None,:],0.0)
        wg=tl.load(W+n[None,:]*K+k[:,None],(n<N)[None,:]&(k<K)[:,None],0.0)
        wu=tl.load(W+(n[None,:]+N)*K+k[:,None],(n<N)[None,:]&(k<K)[:,None],0.0)
        ag=tl.dot(a,wg,ag,max_num_imprecise_acc=BK)
        au=tl.dot(a,wu,au,max_num_imprecise_acc=BK)
    sx=tl.load(XS+m,m<M,0.)
    sg=tl.load(WS+n,n<N,0.);su=tl.load(WS+n+N,n<N,0.)
    g=(ag*sx[:,None]*sg[None,:]).to(tl.bfloat16).to(tl.float32)
    u=(au*sx[:,None]*su[None,:]).to(tl.bfloat16).to(tl.float32)
    h=(g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)*u
    tl.store(H+m[:,None]*N+n[None,:],h,(m<M)[:,None]&(n<N)[None,:])


def fused_gateup_quant(q,scale,weight,wscale,width):
    m,k=q.shape
    if os.environ.get('Q38_DEV_FFN')=='quant':
        gu=torch._scaled_mm(q,weight.t(),scale_a=scale[:,None],scale_b=wscale[None,:],out_dtype=torch.bfloat16,use_fast_accum=True)
        h=torch.empty((m,width),device=q.device,dtype=torch.float8_e4m3fn)
        hs=torch.empty((m,),device=q.device,dtype=torch.float32)
        bt=int(os.environ.get('Q38_DEV_FFN_BT','8'))
        _swiglu_quant_tiled[(triton.cdiv(m,bt),)](gu,h,hs,m,width,bt,triton.next_power_of_2(width),num_warps=4)
        return h,hs
    if os.environ.get('Q38_DEV_FFN')=='partial':
        from low_precision import quant_rows
        h=torch.empty((m,width),device=q.device,dtype=torch.bfloat16)
        bm=int(os.environ.get('Q38_DEV_FFN_BM','64'));bn=int(os.environ.get('Q38_DEV_FFN_BN','128'))
        warps=int(os.environ.get('Q38_DEV_FFN_WARPS','4'))
        _gateup_swiglu[(triton.cdiv(m,bm),triton.cdiv(width,bn))](q,scale,weight,wscale,h,m,k,width,bm,bn,64,
            num_warps=warps,num_stages=3,enable_fp_fusion=False)
        return quant_rows(h)
    out=torch.empty((m,width),device=q.device,dtype=torch.float8_e4m3fn)
    s=torch.empty((m,),device=q.device,dtype=torch.float32)
    bm=int(os.environ.get('Q38_DEV_FFN_BM','16'));warps=int(os.environ.get('Q38_DEV_FFN_WARPS','8'))
    _fused_gateup_quant[(triton.cdiv(m,bm),)](q,scale,weight,wscale,out,s,m,k,width,bm,triton.next_power_of_2(width),64,
        num_warps=warps,num_stages=2,enable_fp_fusion=False)
    return out,s


def _ffn_forward(self,x):
    from prequant_norm import quant_rows
    if self.training or torch.is_grad_enabled():raise RuntimeError('C1 optimization is inference-only')
    shape=x.shape;q,scale=quant_rows(x.reshape(-1,shape[-1]))
    hidden,hidden_scale=fused_gateup_quant(q,scale,self._dense_q_gate_up,self._dense_s_gate_up,self._dense_width)
    y=torch._scaled_mm(hidden,self._dense_q_down.t(),scale_a=hidden_scale[:,None],scale_b=self._dense_s_down[None,:],out_dtype=torch.bfloat16,use_fast_accum=True)
    return y.reshape(*shape[:-1],self.down_proj.out_features)


def _fla_kkt(k,b,cu_seqlens=None,chunk_size=64):
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.utils import solve_tril
    assert cu_seqlens is None and chunk_size==64 and k.shape[-1]==64
    a=chunk_scaled_dot_kkt_fwd(k=k,beta=b,g=None,chunk_size=64,output_dtype=torch.float32)
    return solve_tril(a,output_dtype=k.dtype)


def install(model):
    result={}
    if any(os.environ.get(k)=='1' for k in ('Q38_DEV_DIRECT_OUT','Q38_DEV_QKV_NORM')):
        count=0
        for module in model.modules():
            if type(module).__name__.endswith('GatedDeltaNet'):
                assert hasattr(module,'_common_original_gdn')
                module.forward=types.MethodType(_gdn_forward,module);count+=1
        result['gdn']={'modules':count,'direct_out':os.environ.get('Q38_DEV_DIRECT_OUT')=='1','fused_qkv_norm':os.environ.get('Q38_DEV_QKV_NORM')=='1'}
    if os.environ.get('Q38_DEV_FFN') in ('1','partial','quant'):
        count=0
        for module in model.modules():
            if hasattr(module,'_dense_q_gate_up'):
                module.forward=types.MethodType(_ffn_forward,module);count+=1
        result['ffn']={'modules':count,'implementation':'fused_gate_up_swiglu_rowwise_fp8','bm':int(os.environ.get('Q38_DEV_FFN_BM','16'))}
    if os.environ.get('Q38_DEV_KKT')=='fla':
        import flash_qla.ops.gated_delta_rule.chunk as chunk
        chunk.kkt_solve=_fla_kkt
        result['kkt']='FLA_FP32_gram_and_tril_solve_BF16_output'
    elif os.environ.get('Q38_DEV_KKT')=='tilelang':
        import flash_qla.ops.gated_delta_rule.chunk as chunk
        from dev_kkt_tilelang import kkt_solve
        chunk.kkt_solve=kkt_solve
        result['kkt']={'implementation':'local_Tilelang_thread_tuning','threads':int(os.environ.get('Q38_DEV_KKT_THREADS','128'))}
    if os.environ.get('Q38_DEV_CORE_DV'):
        import flash_qla.ops.gated_delta_rule.chunk as chunk
        from dev_qla_fwd import fused_gdr_fwd
        chunk.fused_gdr_fwd=fused_gdr_fwd
        result['qla_core']={'block_DV':int(os.environ['Q38_DEV_CORE_DV'])}
    return result
