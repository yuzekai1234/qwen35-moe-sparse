"""Inference-only Hopper quantization experiments; original parameters remain intact."""
import os
import sys
import types
import hashlib
import torch
import triton
import triton.language as tl
from hopper_inference import _locate_block, GATE_UP_CONFIG, DOWN_CONFIG

NATIVE_FP8_EVIDENCE={}
FP8_GATE_CONFIG=dict(BLOCK_M=64,BLOCK_N=64,BLOCK_K=64,num_warps=4,num_stages=3)
FP8_DOWN_CONFIG=dict(BLOCK_M=64,BLOCK_N=128,BLOCK_K=64,num_warps=4,num_stages=4)


@triton.jit
def _quant_rows(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row*K+k, k<K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0)/448., 1.e-12)
    tl.store(Q+row*K+k, tl.minimum(tl.maximum(x/scale, -448.), 448.), k<K)
    tl.store(S+row, scale)


def quant_rows(x):
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.numel()//x.shape[-1], device=x.device, dtype=torch.float32)
    _quant_rows[(scales.numel(),)](x,q,scales,x.shape[-1],triton.next_power_of_2(x.shape[-1]))
    return q,scales


@triton.jit
def _gate_up_fp8(Cum, Counts, Starts, X, XS, W, WS, H, Perm,
                  K: tl.constexpr, N: tl.constexpr,
                  E: tl.constexpr, E_PAD: tl.constexpr, TOPK: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACCUM_LIMIT: tl.constexpr=32):
    pm,pn=tl.program_id(0),tl.program_id(1)
    e,row0,valid=_locate_block(pm,Cum,Counts,Starts,E,E_PAD,BLOCK_M)
    if valid <= 0:
        return
    mm=tl.arange(0,BLOCK_M);nn=pn*BLOCK_N+tl.arange(0,BLOCK_N);kk=tl.arange(0,BLOCK_K)
    rows=(row0+mm).to(tl.int64)
    token=(tl.load(Perm+rows,mm<valid,other=0)//TOPK).to(tl.int64)
    ag=tl.zeros((BLOCK_M,BLOCK_N),tl.float32);au=tl.zeros((BLOCK_M,BLOCK_N),tl.float32)
    wb=W+e.to(tl.int64)*(2*N*K)
    for k0 in range(tl.cdiv(K,BLOCK_K)):
        k=k0*BLOCK_K+kk
        a=tl.load(X+token[:,None]*K+k[None,:],(mm<valid)[:,None]&(k<K)[None,:],other=0.0)
        g=tl.load(wb+nn[None,:]*K+k[:,None],(nn<N)[None,:]&(k<K)[:,None],other=0.0)
        u=tl.load(wb+(nn+N)[None,:]*K+k[:,None],(nn<N)[None,:]&(k<K)[:,None],other=0.0)
        ag=tl.dot(a,g,ag,max_num_imprecise_acc=ACCUM_LIMIT)
        au=tl.dot(a,u,au,max_num_imprecise_acc=ACCUM_LIMIT)
    sx=tl.load(XS+token,mm<valid,other=0)
    sg=tl.load(WS+e*(2*N)+nn,nn<N,other=0)
    su=tl.load(WS+e*(2*N)+nn+N,nn<N,other=0)
    g=(ag*sx[:,None]*sg[None,:]).to(tl.bfloat16).to(tl.float32)
    u=(au*sx[:,None]*su[None,:]).to(tl.bfloat16).to(tl.float32)
    silu=(g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(H+rows[:,None]*N+nn[None,:],silu*u,(mm<valid)[:,None]&(nn<N)[None,:])


@triton.jit
def _down_fp8(Cum,Counts,Starts,H,HS,W,WS,Y,Perm,RW,
               K:tl.constexpr,N:tl.constexpr,E:tl.constexpr,E_PAD:tl.constexpr,
               BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr,ACCUM_LIMIT:tl.constexpr=32):
    pm,pn=tl.program_id(0),tl.program_id(1)
    e,row0,valid=_locate_block(pm,Cum,Counts,Starts,E,E_PAD,BLOCK_M)
    if valid<=0:
        return
    mm=tl.arange(0,BLOCK_M);nn=pn*BLOCK_N+tl.arange(0,BLOCK_N);kk=tl.arange(0,BLOCK_K)
    rows=(row0+mm).to(tl.int64)
    pair=tl.load(Perm+rows,mm<valid,other=0).to(tl.int64)
    acc=tl.zeros((BLOCK_M,BLOCK_N),tl.float32)
    wb=W+e.to(tl.int64)*(N*K)
    for k0 in range(tl.cdiv(K,BLOCK_K)):
        k=k0*BLOCK_K+kk
        a=tl.load(H+rows[:,None]*K+k[None,:],(mm<valid)[:,None]&(k<K)[None,:],other=0.0)
        w=tl.load(wb+nn[None,:]*K+k[:,None],(nn<N)[None,:]&(k<K)[:,None],other=0.0)
        acc=tl.dot(a,w,acc,max_num_imprecise_acc=ACCUM_LIMIT)
    sh=tl.load(HS+rows,mm<valid,other=0)
    sw=tl.load(WS+e*N+nn,nn<N,other=0)
    weight=tl.load(RW+pair,mm<valid,other=0).to(tl.float32)
    val=(acc*sh[:,None]*sw[None,:]).to(tl.bfloat16).to(tl.float32)*weight[:,None]
    tl.store(Y+pair[:,None]*N+nn[None,:],val,(mm<valid)[:,None]&(nn<N)[None,:])


def fp8_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    if torch.is_grad_enabled():
        raise RuntimeError('Quantized experts require torch.inference_mode()')
    tokens,hidden=hidden_states.shape
    topk=top_k_index.shape[-1];experts=self.num_experts;intermediate=self.intermediate_dim
    ids,perm=torch.sort(top_k_index.reshape(-1),stable=True)
    bounds=torch.searchsorted(ids,torch.arange(experts+1,device=ids.device,dtype=ids.dtype))
    starts=bounds[:-1].int();counts=(bounds[1:]-bounds[:-1]).int();pairs=ids.numel()
    common=dict(E=experts,E_PAD=triton.next_power_of_2(experts))
    def launch(kernel,config,n,*params,**meta):
        block=config['BLOCK_M'];cum=torch.cumsum((counts+block-1)//block,0,dtype=torch.int32)
        grid=(triton.cdiv(pairs,block)+experts,triton.cdiv(n,config['BLOCK_N']))
        limit=min(int(os.environ.get('Q38_FP8_ACCUM_LIMIT','32')),config['BLOCK_K'])
        compiled=kernel[grid](cum,counts,starts,*params,ACCUM_LIMIT=limit,**meta,**config)
        kernel_name=kernel.fn.__name__
        if kernel_name not in NATIVE_FP8_EVIDENCE:
            ptx=compiled.asm['ptx']
            instructions=[line.strip() for line in ptx.splitlines() if 'mma' in line and 'e4m3' in line]
            if not instructions:
                raise RuntimeError('FP8 kernel was lowered without native E4M3 MMA instructions')
            NATIVE_FP8_EVIDENCE[kernel_name]=dict(instruction=instructions[0],
                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                fp32_accumulator=True,max_imprecise_accumulations=limit)
    x,sx=quant_rows(hidden_states)
    h=torch.empty((pairs,intermediate),device=hidden_states.device,dtype=torch.bfloat16)
    launch(_gate_up_fp8,FP8_GATE_CONFIG,intermediate,x,sx,self._q8_gate,self._q8_gate_scale,h,perm,
           K=hidden,N=intermediate,TOPK=topk,**common)
    del x,sx
    qh,sh=quant_rows(h)
    del h
    y=torch.empty((pairs,hidden),device=hidden_states.device,dtype=torch.bfloat16)
    launch(_down_fp8,FP8_DOWN_CONFIG,hidden,qh,sh,self._q8_down,self._q8_down_scale,y,perm,
           top_k_weights.reshape(-1),K=intermediate,N=hidden,**common)
    if os.environ.get('Q38_FUSED_COMBINE')=='1':
        from combine import combine_topk
        return combine_topk(y,tokens,topk,hidden)
    return y.view(tokens,topk,hidden).sum(1)


def install_fp8_experts(model):
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError('This kernel qualification targets Hopper SM90')
    count=0
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__ != 'Qwen3_8MoeExperts':
                continue
            for source,name in (('gate_up_proj','gate'),('down_proj','down')):
                q,scale=quant_rows(getattr(module,source).detach())
                module.register_buffer('_q8_'+name,q,persistent=False)
                module.register_buffer('_q8_'+name+'_scale',scale,persistent=False)
                # Keep original BF16 parameters for checkpoint identity, but do
                # not retain an unused GPU copy beside the quantized cache.
                parameter=getattr(module,source)
                parameter.data=parameter.detach().cpu()
            count+=1
    ALL_EXPERTS_FUNCTIONS.register('q38_hopper_fp8',fp8_experts_forward)
    model.set_experts_implementation('q38_hopper_fp8')
    return count


@triton.jit
def _rms_projection(X,A,Z,M:tl.constexpr,K:tl.constexpr,R:tl.constexpr,
                    BF16:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_K:tl.constexpr,R_PAD:tl.constexpr):
    mm=(tl.program_id(0)*BLOCK_M+tl.arange(0,BLOCK_M)).to(tl.int64)
    rr=tl.arange(0,R_PAD);kk=tl.arange(0,BLOCK_K)
    acc=tl.zeros((BLOCK_M,R_PAD),tl.float32);sum_sq=tl.zeros((BLOCK_M,),tl.float32)
    for k0 in range(tl.cdiv(K,BLOCK_K)):
        k=k0*BLOCK_K+kk
        x=tl.load(X+mm[:,None]*K+k[None,:],(mm<M)[:,None]&(k<K)[None,:],other=0).to(tl.float32)
        a=tl.load(A+rr[None,:]*K+k[:,None],(rr<R)[None,:]&(k<K)[:,None],other=0)
        if BF16:
            acc=tl.dot(x.to(tl.bfloat16),a.to(tl.bfloat16),acc)
        else:
            acc=tl.dot(x,a.to(tl.float32),acc,input_precision='tf32x3')
        sum_sq+=tl.sum(x*x,1)
    inv=tl.rsqrt(sum_sq/K+1.e-6)
    tl.store(Z+mm[:,None]*R+rr[None,:],acc*inv[:,None],(mm<M)[:,None]&(rr<R)[None,:])


def _coefficients(X,lowrank,B,b_pre,b_post,b_res,alphas,n,iters):
    z=torch.tanh(torch.nn.functional.linear(lowrank,B))
    zp,zq,zr=z[...,:n],z[...,n:2*n],z[...,2*n:].reshape(*z.shape[:-1],n,n)
    pre=torch.sigmoid(b_pre+alphas[0]*zp)
    post=2*torch.sigmoid(b_post+alphas[1]*zq)
    logits=b_res+alphas[2]*zr
    res=torch.exp(logits-logits.amax((-2,-1),keepdim=True))
    for _ in range(iters):
        res=res/res.sum(-1,keepdim=True).clamp(min=1.e-9)
        res=res/res.sum(-2,keepdim=True).clamp(min=1.e-9)
    return (pre.unsqueeze(-1)*X).sum(2),pre,post,res


_coefficients_compiled=torch.compile(_coefficients,dynamic=False)


def _mhc_pre(self,X):
    m=X.shape[0]*X.shape[1];k=X.shape[2]*X.shape[3];r=self.A.weight.shape[0]
    lowrank=torch.empty((m,r),device=X.device,dtype=torch.float32)
    _rms_projection[(triton.cdiv(m,64),)](X,self.A.weight.float(),lowrank,m,k,r,
        self._q38_bf16_projection,64,32,max(16,triton.next_power_of_2(r)),num_warps=4,num_stages=3)
    xin,pre,post,res=_coefficients_compiled(X,lowrank.view(*X.shape[:2],-1),self.B.weight.float(),
        self.b_pre.float(),self.b_post.float(),self.b_res.float(),self.alphas.float(),self.n,self.iters)
    return xin,(X,pre,post,res)


def _mix_bf16(X,res,post,y):
    from hopper_inference import _mix_four_branches
    return _mix_four_branches(X,res,post,y).to(torch.bfloat16)


_mix_bf16_compiled=torch.compile(_mix_bf16,dynamic=False)


def _mhc_post(self,X,y,c):
    return _mix_bf16_compiled(c[0],c[3],c[2],y)


def _bf16_stream_trunk(self,ids):
    # The embedding already is BF16. Avoid an otherwise-unused initial FP32
    # four-branch copy remaining live in the caller throughout the first layer.
    # mHC statistics/mixing still upcast to FP32 inside the fused operators.
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('BF16 residual trunk is inference-only')
    length=ids.shape[1]
    if not length:
        raise ValueError('Expected nonempty packed inputs')
    padding=(-length)%64
    if padding:
        ids=torch.nn.functional.pad(ids,(0,padding),value=self.config.pad_token_id or 0)
    embedding=self.embed_tokens(ids)
    positions=torch.arange(ids.shape[1],device=ids.device).view(1,1,-1).expand(3,ids.shape[0],-1)
    cos,sin=self.rotary_emb(embedding,positions)
    if self.config.indexer_head_dim != self.config.head_dim:
        raise ValueError('Indexer/attention rotary dimensions differ')
    residual=embedding.unsqueeze(2).expand(-1,-1,self.config.n_branch,-1).contiguous()
    for index in range(len(self.layers)):
        residual,kl=self._layer(index,residual,ids,cos,sin,embedding.dtype)
        if kl is not None:
            raise RuntimeError('Inference trunk must not compute training KL')
    return residual.sum(2)[:,:length].to(embedding.dtype),[]


def install_mhc(model,projection_bf16=True,stream_bf16=True):
    count=0
    for module in model.modules():
        if type(module).__name__ != 'MHC':
            continue
        module._q38_bf16_projection=projection_bf16
        module.pre=types.MethodType(_mhc_pre,module)
        if stream_bf16:
            module.post=types.MethodType(_mhc_post,module)
        count+=1
    if stream_bf16:
        model.model.trunk=types.MethodType(_bf16_stream_trunk,model.model)
    return dict(modules=count,projection_bf16=projection_bf16,stream_bf16=stream_bf16,
                initial_stream_bf16=stream_bf16)


def install_variant(model,name):
    if model.training:
        raise ValueError('Load an eval model before installing inference quantization')
    result={}
    if name in ('common_memory','general_fp8','general_fp8_memory','general_fp8_gdn','general_dense_fused_gdn','general_dense_fused_norm'):
        original_experts=getattr(model.config,'_experts_implementation',None)
        if name!='common_memory':
            os.environ['Q38_FP8_ACCUM_LIMIT']=str(1<<30)
            os.environ['Q38_FP8_FAST_ACCUM']='1'
            from fp8_linear import install as install_linear
            from mhc_projection import install as install_projection
            result['fp8_linear']=install_linear(model,'general')
            result['mhc_fp8']=install_projection(model,quant_b=True)
        if name in ('common_memory','general_fp8_memory'):
            from common_inference import install_memory_optimizations
            result['common_memory']=install_memory_optimizations(model)
        if name in ('general_fp8_gdn','general_dense_fused_gdn','general_dense_fused_norm'):
            from common_gdn import install as install_common_gdn
            result['common_gdn']=install_common_gdn(model)
        if name in ('general_dense_fused_gdn','general_dense_fused_norm'):
            from dense_ffn import install as install_dense_ffn
            result['dense_ffn']=install_dense_ffn(model)
        if name=='general_dense_fused_norm':
            from common_residual_norm import install as install_residual
            result['residual_norm']=install_residual(model)
        if name=='general_dense_fused_norm':
            if os.environ.get('Q38_CUDNN_ATTENTION')=='1':
                from cudnn_attention import install as install_cudnn
                result['cudnn_attention']=install_cudnn(model)
            elif os.environ.get('Q38_INFRA_ATTENTION')=='1':
                from infra_attention import install as install_attention
                result['infra_attention']=install_attention(model)
        if name=='general_dense_fused_norm' and os.environ.get('Q38_PREQUANT_NORM')=='1':
            from prequant_norm import install as install_prequant
            result['prequant_norm']=install_prequant(model)
        if name=='general_dense_fused_norm' and os.environ.get('Q38_FLASHQLA_BACKEND')=='1':
            from gdn_flashqla_backend import install as install_flashqla
            result['flashqla']=install_flashqla(model)
        if any(key.startswith('Q38_DEV_') and value not in ('','0') for key,value in os.environ.items()):
            from dev_optimizations import install as install_development
            result['development']=install_development(model)
        result['scope']='general_dense_reusable'
        assert getattr(model.config,'_experts_implementation',None)==original_experts
        result['experts_modified']=False;result['router_modified']=False
        result['existing_expert_path']=original_experts
        return result
    if name in ('hybrid_flashinfer_fp8_chunked','hybrid_flashinfer_fp4_chunked'):
        os.environ['Q38_FP8_ACCUM_LIMIT']=str(1<<30)
        os.environ['Q38_FP8_FAST_ACCUM']='1'
        from fp8_linear import install as install_linear
        from mhc_projection import install as install_projection
        from flashinfer_backend import install as install_library
        result['fp8_linear']=install_linear(model,'maximal')
        result['mhc_fp8']=install_projection(model,quant_b=True)
        result['flashinfer']=install_library(model,'fp8' if '_fp8' in name else 'fp4')
        result['accumulation']=dict(triton_accumulation_limit='tile_K',torch_scaled_mm_use_fast_accum=True)
        return result
    fast_aliases={'fp8_broad_fast':'fp8_broad','fp8_stream_fast':'fp8_broad_bf16stream','fp8_maximal_fast':'fp8_maximal'}
    if name in fast_aliases:
        os.environ['Q38_FP8_ACCUM_LIMIT']=str(1<<30)
        os.environ['Q38_FP8_FAST_ACCUM']='1'
        result=install_variant(model,fast_aliases[name])
        result['accumulation']=dict(base_variant=fast_aliases[name],mode='Hopper_fast_accumulation',
            triton_accumulation_limit='tile_K',torch_scaled_mm_use_fast_accum=True)
        return result
    if name in ('mhc_bf16','mhc_bf16_safe','mhc_fp8','mhc_fp8_all','mhc_fp4','mhc_flashinfer_fp8','mhc_flashinfer_fp4'):
        result['mhc']=install_mhc(model)
    elif name == 'mhc_fused_fp32':
        result['mhc']=install_mhc(model,False,False)
    if name in ('fp8_experts','mhc_fp8','mhc_fp8_all','fp8_all','fp8_fused'):
        result['fp8_experts']=install_fp8_experts(model)
    if name in ('fp4_experts','mhc_fp4'):
        from fp4_experts import install_fp4_experts
        result['fp4_experts']=install_fp4_experts(model)
    if name in ('fp8_projections','mhc_fp8_all','fp8_all'):
        from fp8_linear import install
        result['fp8_linear']=install(model)
    if name in ('fp8_fused','fp8_all'):
        os.environ['Q38_FUSED_COMBINE']='1'
        result['fused_combine']=True
    if name in ('flashinfer_fp8','mhc_flashinfer_fp8','flashinfer_fp4','mhc_flashinfer_fp4'):
        from flashinfer_backend import install
        result['flashinfer']=install(model,'fp8' if name.endswith('fp8') else 'fp4')
    if name in ('fp8_mhc','fp8_broad','fp8_broad_bf16stream','fp8_maximal'):
        if name != 'fp8_mhc':
            result['fp8_experts']=install_fp8_experts(model)
            from fp8_linear import install as install_linear
            result['fp8_linear']=install_linear(model,'maximal' if name=='fp8_maximal' else 'projections')
            os.environ['Q38_FUSED_COMBINE']='1'
            result['fused_combine']=True
        from mhc_projection import install as install_projection
        result['mhc_fp8']=install_projection(model,quant_b=name=='fp8_maximal',stream_bf16=name=='fp8_broad_bf16stream')
    if name not in ('mhc_bf16','mhc_bf16_safe','mhc_fused_fp32','fp8_experts','mhc_fp8','mhc_fp8_all',
                    'fp4_experts','mhc_fp4','fp8_projections','flashinfer_fp8','mhc_flashinfer_fp8',
                    'flashinfer_fp4','mhc_flashinfer_fp4','fp8_all','fp8_fused','fp8_mhc','fp8_broad','fp8_broad_bf16stream','fp8_maximal'):
        raise ValueError('Unknown variant '+name)
    return result
