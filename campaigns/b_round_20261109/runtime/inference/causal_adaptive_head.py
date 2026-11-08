"""Input-only chunked SGD output-head adaptation, reset per sequence/document.

Paper motivation: https://arxiv.org/abs/1904.08378. This is an output-head-only
adaptation on own cosine features, rather than the paper's whole-network update.
The temporary fast weights never modify checkpoint parameters or survive a call.
"""
import torch
from torch.nn import functional as F


class CausalAdaptiveHead:
    def __init__(self,learning_rate,decay=.95,chunk=128,eos_id=53,feature_stride=1):
        if learning_rate<0 or not 0<=decay<=1 or chunk<1 or feature_stride<1:raise ValueError('Invalid adaptation coefficient')
        self.feature_stride=int(feature_stride)
        self.learning_rate=float(learning_rate);self.decay=float(decay);self.chunk=int(chunk);self.eos_id=int(eos_id)

    @torch.inference_mode()
    def __call__(self,ids,hidden,base_logits,inplace=False,output_dtype=torch.bfloat16):
        if ids.ndim!=2 or hidden.shape[:2]!=ids.shape or base_logits.shape[:2]!=ids.shape:raise ValueError('Mismatched adaptation tensors')
        if self.learning_rate==0:return base_logits if base_logits.dtype==output_dtype else base_logits.to(output_dtype)
        batch,length=ids.shape;vocab=base_logits.shape[-1]
        features=F.normalize(hidden[...,::self.feature_stride].float(),dim=-1)
        logits=base_logits if inplace and base_logits.dtype==output_dtype else base_logits.to(output_dtype).clone()
        fast=torch.zeros((batch,vocab,features.shape[-1]),device=hidden.device,dtype=torch.float32)
        for lo in range(0,length,self.chunk):
            hi=min(length,lo+self.chunk)
            if lo:
                previous=max(0,lo-self.chunk)
                prev_ids=ids[:,previous:lo]
                positions=torch.arange(lo-previous,device=ids.device)[None,:]
                last_eos=torch.where(prev_ids==self.eos_id,positions,-1).amax(1)
                reset=(last_eos>=0)|(ids[:,lo]==self.eos_id)
                # Labels are only input tokens up to the current chunk's first position.
                labels=ids[:,previous+1:lo+1]
                valid=(positions>=last_eos[:,None])&(labels!=self.eos_id)&(ids[:,lo,None]!=self.eos_id)
                count=valid.sum(1).clamp_min(1)
                gradient_logits=logits[:,previous:lo].float().softmax(-1)
                gradient_logits.scatter_add_(2,labels[:,:,None],-torch.ones_like(labels,dtype=torch.float32)[:,:,None])
                gradient_logits.mul_(valid[:,:,None]/count[:,None,None])
                gradient=torch.bmm(gradient_logits.transpose(1,2),features[:,previous:lo])
                fast.mul_((~reset)[:,None,None]).mul_(self.decay).add_(gradient,alpha=-self.learning_rate)
            correction=torch.bmm(features[:,lo:hi],fast.transpose(1,2))
            # A new document inside a chunk starts with zero fast weights. Its
            # first update is delayed to the next chunk; no old-document state leaks.
            before_eos=(ids[:,lo:hi]==self.eos_id).cumsum(1)==0
            updated=base_logits[:,lo:hi].float()+correction*before_eos[:,:,None]
            logits[:,lo:hi]=updated.to(output_dtype)
        return logits

    def state_and_compute_bounds(self,batch,sequence,hidden,vocab):
        hidden=(hidden+self.feature_stride-1)//self.feature_stride
        return dict(fast_weight_fp32_bytes=4*batch*vocab*hidden,
            each_gradient_fp32_bytes=4*batch*vocab*hidden,
            normalized_features_fp32_bytes=4*batch*sequence*hidden,
            forward_and_update_gemm_flops_upperbound=4*batch*sequence*hidden*vocab,
            no_persistent_model_update=True,all_vocabulary_output=True)
