"""Deterministic top-k combine with register FP32 accumulation and BF16 output."""
import torch
import triton
import triton.language as tl


@triton.jit
def _combine(Pairs,Out,TOTAL:tl.constexpr,H:tl.constexpr,TOPK:tl.constexpr,BLOCK:tl.constexpr):
    offset=(tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)).to(tl.int64)
    token=offset//H;column=offset%H
    value=tl.zeros((BLOCK,),tl.float32)
    for k in tl.static_range(TOPK):
        value+=tl.load(Pairs+(token*TOPK+k)*H+column,offset<TOTAL,other=0.).to(tl.float32)
    tl.store(Out+offset,value,offset<TOTAL)


def combine_topk(pairs,tokens,topk,hidden):
    output=torch.empty((tokens,hidden),device=pairs.device,dtype=pairs.dtype)
    _combine[(triton.cdiv(tokens*hidden,1024),)](pairs,output,tokens*hidden,hidden,topk,1024,
        num_warps=4,enable_fp_fusion=False)
    return output
