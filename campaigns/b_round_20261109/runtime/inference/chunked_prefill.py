"""Preserve the shared release's frame-retention fix while exposing own features."""
import os
import torch


def chunk_size():
    value=int(os.environ.get('Q38_TRUNK_CHUNK','0'))
    if value<0:raise ValueError('Q38_TRUNK_CHUNK must be nonnegative')
    return value


def build(model,compile_function,size):
    lm_head=model.lm_head
    assert lm_head._q8_bias is None and lm_head._q8_weight.shape[0]==lm_head.out_features
    weight=lm_head._q8_weight.t();weight_scale=lm_head._q8_weight_scale[None,:]
    vocab=lm_head.out_features;hidden=weight.shape[0]
    def trunk(ids):
        from prequant_norm import quant_rows
        h,_=model.model.trunk(ids);h=model.model.norm(h)
        return quant_rows(h.reshape(-1,h.shape[-1]))
    compiled=compile_function(trunk)
    def head(heads,batch,length,device):
        # The direct compiled caller never binds logits/features in its frame.
        # Pop all q/scale tensors, preserving the release's documented lifetime fix.
        logits=torch.empty((batch*length,vocab),device=device,dtype=torch.bfloat16)
        features=torch.empty((batch*length,hidden),device=device,dtype=torch.bfloat16)
        row=0
        while heads:
            q,scale=heads.pop(0);end=row+q.shape[0]
            features[row:end].copy_(q.to(torch.bfloat16))
            torch.ops.aten._scaled_mm.out(q,weight,scale[:,None],weight_scale,bias=None,
                out_dtype=torch.bfloat16,use_fast_accum=True,out=logits[row:end])
            row=end
        return logits.view(batch,length,vocab),features.view(batch,length,hidden)
    def forward(ids):
        batch,length=ids.shape
        heads=[compiled(ids[start:start+size]) for start in range(0,batch,size)]
        return head(heads,batch,length,ids.device)
    return forward
