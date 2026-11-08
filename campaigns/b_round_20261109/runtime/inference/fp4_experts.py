"""E2M1 group-32 packed expert weights, BF16 Tensor Core computation on Hopper.

This is W4A16, not a native FP4 Tensor Core kernel or NVIDIA NVFP4 scale format.
Weights are unpacked inside each GEMM tile, so timing includes their decode cost.
"""
import torch
import triton
import triton.language as tl
from hopper_inference import _locate_block,GATE_UP_CONFIG,DOWN_CONFIG


@torch.no_grad()
def quantize_e2m1(weight,group=32):
    shape=weight.shape
    if shape[-1] % group or group % 2:
        raise ValueError('E2M1 group size must evenly divide the inner dimension')
    x=weight.reshape(-1,group).float()
    scale=(x.abs().amax(-1,keepdim=True)/6.).clamp_min(1.e-12)
    magnitude=(x/scale).abs()
    bounds=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],device=x.device)
    code=torch.bucketize(magnitude,bounds).to(torch.uint8)
    code=code|((x<0).to(torch.uint8)<<3)
    packed=(code[:,0::2]|(code[:,1::2]<<4)).reshape(*shape[:-1],shape[-1]//2)
    return packed,scale.reshape(*shape[:-1],shape[-1]//group)


@triton.jit
def _weight4(W,S,e,n,k,N:tl.constexpr,K:tl.constexpr,GROUP:tl.constexpr):
    base=e.to(tl.int64)*N*(K//2)
    byte=tl.load(W+base+n[None,:]*(K//2)+k[:,None]//2,
                 (n<N)[None,:]&(k<K)[:,None],other=0).to(tl.int32)
    code=(byte>>((k[:,None]%2)*4))&15
    c=code&7
    val=tl.where(c<2,.5*c,(1.+.5*(c&1))*(1<<tl.maximum((c>>1)-1,0)))
    val=tl.where((code&8)!=0,-val,val)
    scale=tl.load(S+e.to(tl.int64)*N*(K//GROUP)+n[None,:]*(K//GROUP)+k[:,None]//GROUP,
                  (n<N)[None,:]&(k<K)[:,None],other=0.)
    return (val*scale).to(tl.bfloat16)


@triton.jit
def _gate4(Cum,Counts,Starts,X,W,WS,H,Perm,
             K:tl.constexpr,N:tl.constexpr,GROUP:tl.constexpr,
             E:tl.constexpr,E_PAD:tl.constexpr,TOPK:tl.constexpr,
             BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr):
    pm,pn=tl.program_id(0),tl.program_id(1)
    e,row0,valid=_locate_block(pm,Cum,Counts,Starts,E,E_PAD,BLOCK_M)
    if valid<=0:
        return
    mm=tl.arange(0,BLOCK_M);nn=pn*BLOCK_N+tl.arange(0,BLOCK_N);kk=tl.arange(0,BLOCK_K)
    rows=(row0+mm).to(tl.int64)
    token=(tl.load(Perm+rows,mm<valid,other=0)//TOPK).to(tl.int64)
    ag=tl.zeros((BLOCK_M,BLOCK_N),tl.float32);au=tl.zeros((BLOCK_M,BLOCK_N),tl.float32)
    for k0 in range(tl.cdiv(K,BLOCK_K)):
        k=k0*BLOCK_K+kk
        a=tl.load(X+token[:,None]*K+k[None,:],(mm<valid)[:,None]&(k<K)[None,:],other=0.)
        # Pass a per-expert base so gate/up remain halves of the same matrix.
        wb=W+e.to(tl.int64)*(2*N)*(K//2);sb=WS+e.to(tl.int64)*(2*N)*(K//GROUP)
        g=_weight4(wb,sb,tl.full((),0,tl.int32),nn,k,2*N,K,GROUP)
        u=_weight4(wb,sb,tl.full((),0,tl.int32),nn+N,k,2*N,K,GROUP)
        ag=tl.dot(a,g,ag);au=tl.dot(a,u,au)
    g=ag.to(tl.bfloat16).to(tl.float32);u=au.to(tl.bfloat16).to(tl.float32)
    silu=(g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(H+rows[:,None]*N+nn[None,:],silu*u,(mm<valid)[:,None]&(nn<N)[None,:])


@triton.jit
def _down4(Cum,Counts,Starts,H,W,WS,Y,Perm,RW,
             K:tl.constexpr,N:tl.constexpr,GROUP:tl.constexpr,E:tl.constexpr,E_PAD:tl.constexpr,
             BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr):
    pm,pn=tl.program_id(0),tl.program_id(1)
    e,row0,valid=_locate_block(pm,Cum,Counts,Starts,E,E_PAD,BLOCK_M)
    if valid<=0:
        return
    mm=tl.arange(0,BLOCK_M);nn=pn*BLOCK_N+tl.arange(0,BLOCK_N);kk=tl.arange(0,BLOCK_K)
    rows=(row0+mm).to(tl.int64)
    pair=tl.load(Perm+rows,mm<valid,other=0).to(tl.int64)
    acc=tl.zeros((BLOCK_M,BLOCK_N),tl.float32)
    for k0 in range(tl.cdiv(K,BLOCK_K)):
        k=k0*BLOCK_K+kk
        a=tl.load(H+rows[:,None]*K+k[None,:],(mm<valid)[:,None]&(k<K)[None,:],other=0.)
        w=_weight4(W,WS,e,nn,k,N,K,GROUP)
        acc=tl.dot(a,w,acc)
    weight=tl.load(RW+pair,mm<valid,other=0).to(tl.float32)
    value=acc.to(tl.bfloat16).to(tl.float32)*weight[:,None]
    tl.store(Y+pair[:,None]*N+nn[None,:],value,(mm<valid)[:,None]&(nn<N)[None,:])


def fp4_experts_forward(self,hidden_states,top_k_index,top_k_weights):
    if torch.is_grad_enabled():
        raise RuntimeError('FP4 packed experts require inference_mode')
    tokens,hidden=hidden_states.shape
    topk=top_k_index.shape[-1];experts=self.num_experts;intermediate=self.intermediate_dim
    ids,perm=torch.sort(top_k_index.reshape(-1),stable=True)
    bounds=torch.searchsorted(ids,torch.arange(experts+1,device=ids.device,dtype=ids.dtype))
    starts=bounds[:-1].int();counts=(bounds[1:]-bounds[:-1]).int();pairs=ids.numel()
    common=dict(E=experts,E_PAD=triton.next_power_of_2(experts),GROUP=32)
    def launch(kernel,config,n,*params,**meta):
        block=config['BLOCK_M'];cum=torch.cumsum((counts+block-1)//block,0,dtype=torch.int32)
        grid=(triton.cdiv(pairs,block)+experts,triton.cdiv(n,config['BLOCK_N']))
        kernel[grid](cum,counts,starts,*params,**meta,**config)
    h=torch.empty((pairs,intermediate),device=hidden_states.device,dtype=torch.bfloat16)
    launch(_gate4,GATE_UP_CONFIG,intermediate,hidden_states,self._q4_gate,self._q4_gate_scale,h,perm,
           K=hidden,N=intermediate,TOPK=topk,**common)
    y=torch.empty((pairs,hidden),device=hidden_states.device,dtype=torch.bfloat16)
    launch(_down4,DOWN_CONFIG,hidden,h,self._q4_down,self._q4_down_scale,y,perm,
           top_k_weights.reshape(-1),K=intermediate,N=hidden,**common)
    return y.view(tokens,topk,hidden).sum(1)


def install_fp4_experts(model):
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    count=0
    for module in model.modules():
        if type(module).__name__ != 'Qwen3_8MoeExperts':
            continue
        for source,name in (('gate_up_proj','gate'),('down_proj','down')):
            q,scale=quantize_e2m1(getattr(module,source).detach())
            module.register_buffer('_q4_'+name,q,persistent=False)
            module.register_buffer('_q4_'+name+'_scale',scale,persistent=False)
            parameter=getattr(module,source)
            parameter.data=parameter.detach().cpu()
        count+=1
    ALL_EXPERTS_FUNCTIONS.register('q38_hopper_fp4_w4a16',fp4_experts_forward)
    model.set_experts_implementation('q38_hopper_fp4_w4a16')
    return dict(modules=count,storage='packed_E2M1',group_size=32,scales='FP32',compute='BF16',
                native_fp4_tensor_cores=False)
