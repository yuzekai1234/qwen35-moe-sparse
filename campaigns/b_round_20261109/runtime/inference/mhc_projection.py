"""Hopper FP8 mHC projections; FP32 stream, Sinkhorn and mixing remain intact."""
import hashlib,os,types
import torch
import triton
import triton.language as tl
from low_precision import quant_rows,_coefficients_compiled,NATIVE_FP8_EVIDENCE

@triton.jit
def _quant_rms(X,Q,S,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);k=tl.arange(0,BLOCK)
    x=tl.load(X+row*K+k,k<K,other=0.).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0)/448.,1.e-12)
    inv=tl.rsqrt(tl.sum(x*x,0)/K+1.e-6)
    tl.store(Q+row*K+k,tl.minimum(tl.maximum(x/scale,-448.),448.),k<K)
    tl.store(S+row,scale*inv)

@triton.jit
def _linear_fp8(X,XS,W,WS,Y,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,ACCUM_LIMIT:tl.constexpr=32):
    m=(tl.program_id(0)*BM+tl.arange(0,BM)).to(tl.int64)
    n=tl.program_id(1)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for k0 in range(tl.cdiv(K,BK)):
        kk=k0*BK+k
        a=tl.load(X+m[:,None]*K+kk[None,:],(m<M)[:,None]&(kk<K)[None,:],other=0.)
        b=tl.load(W+n[None,:]*K+kk[:,None],(n<N)[None,:]&(kk<K)[:,None],other=0.)
        acc=tl.dot(a,b,acc,max_num_imprecise_acc=ACCUM_LIMIT)
    sa=tl.load(XS+m,m<M,other=0.);sb=tl.load(WS+n,n<N,other=0.)
    tl.store(Y+m[:,None]*N+n[None,:],acc*sa[:,None]*sb[None,:],(m<M)[:,None]&(n<N)[None,:])

def linear_q8(q,scale,weight,wscale):
    m,k=q.shape;n=weight.shape[0]
    output=torch.empty((m,n),device=q.device,dtype=torch.float32)
    limit=min(int(os.environ.get('Q38_FP8_ACCUM_LIMIT','32')),64)
    compiled=_linear_fp8[(triton.cdiv(m,64),triton.cdiv(n,128))](q,scale,weight,wscale,output,m,k,n,64,128,64,limit,num_warps=4,num_stages=3)
    if 'mhc_linear_fp8' not in NATIVE_FP8_EVIDENCE:
        ptx=compiled.asm['ptx'];instructions=[l.strip() for l in ptx.splitlines() if 'mma' in l and 'e4m3' in l]
        if not instructions:raise RuntimeError('mHC projection did not lower to native E4M3 MMA')
        NATIVE_FP8_EVIDENCE['mhc_linear_fp8']=dict(instruction=instructions[0],ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),fp32_accumulator=True,max_imprecise_accumulations=limit)
    return output

def _coefficients_z(X,projected,b_pre,b_post,b_res,alphas,n,iters):
    z=torch.tanh(projected)
    zp,zq,zr=z[...,:n],z[...,n:2*n],z[...,2*n:].reshape(*z.shape[:-1],n,n)
    pre=torch.sigmoid(b_pre+alphas[0]*zp);post=2*torch.sigmoid(b_post+alphas[1]*zq)
    logits=b_res+alphas[2]*zr;res=torch.exp(logits-logits.amax((-2,-1),keepdim=True))
    for _ in range(iters):
        res=res/res.sum(-1,keepdim=True).clamp(min=1.e-9)
        res=res/res.sum(-2,keepdim=True).clamp(min=1.e-9)
    return (pre.unsqueeze(-1)*X).sum(2),pre,post,res
_coefficients_z_compiled=torch.compile(_coefficients_z,dynamic=False)

def _pre(self,X):
    if self.training or torch.is_grad_enabled():raise RuntimeError('FP8 mHC is inference only')
    with torch.autocast('cuda',enabled=False):
        X=X if self._q8_stream_bf16 else X.float()
        m=X.shape[0]*X.shape[1];k=X.shape[2]*X.shape[3]
        q=torch.empty((m,k),device=X.device,dtype=torch.float8_e4m3fn)
        scales=torch.empty(m,device=X.device,dtype=torch.float32)
        _quant_rms[(m,)](X,q,scales,k,triton.next_power_of_2(k))
        lowrank=linear_q8(q,scales,self._q8_A,self._q8_A_scale)
        del q,scales
        if self._q8_quant_b:
            q,scales=quant_rows(lowrank)
            projected=linear_q8(q,scales,self._q8_B,self._q8_B_scale).view(*X.shape[:2],-1)
            del q,scales,lowrank
            xin,pre,post,res=_coefficients_z_compiled(X,projected,self.b_pre.float(),self.b_post.float(),self.b_res.float(),self.alphas.float(),self.n,self.iters)
        else:
            xin,pre,post,res=_coefficients_compiled(X,lowrank.view(*X.shape[:2],-1),self.B.weight.float(),self.b_pre.float(),self.b_post.float(),self.b_res.float(),self.alphas.float(),self.n,self.iters)
        return xin,(X,pre,post,res)

def install(model,quant_b=False,stream_bf16=False):
    if stream_bf16:
        from low_precision import install_mhc
        install_mhc(model)
    count=0
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__!='MHC':continue
            for label in (('A','B') if quant_b else ('A',)):
                parameter=getattr(module,label).weight
                q,scale=quant_rows(parameter.detach())
                module.register_buffer('_q8_'+label,q,persistent=False)
                module.register_buffer('_q8_'+label+'_scale',scale,persistent=False)
                parameter.data=parameter.detach().cpu()
            module._q8_quant_b=quant_b;module._q8_stream_bf16=stream_bf16;module.pre=types.MethodType(_pre,module);count+=1
    return dict(modules=count,A='dynamic_E4M3_W8A8_FP32_acc_output',B='dynamic_E4M3_W8A8' if quant_b else 'original_FP32',residual_stream='BF16_storage_FP32_mix' if stream_bf16 else 'original_FP32',coefficients_sinkhorn='original_FP32')
