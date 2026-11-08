"""Sparse normalized neural-prior count expert; inputs only, all-vocabulary BF16 logits."""
import json
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl

@triton.jit
def _lookup(X,KEYS,TOTALS,ROWS,N:tl.constexpr,S:tl.constexpr,K:tl.constexpr,EOS:tl.constexpr,
            ORDER:tl.constexpr,SEED:tl.constexpr,PRIME:tl.constexpr,MINIMUM:tl.constexpr,BLOCK:tl.constexpr):
    pos=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);inside=pos<N
    col=pos%S;valid=inside&(col>=ORDER-2);h=tl.full((BLOCK,),SEED&((1<<48)-1),tl.int64)
    for j in range(ORDER-1):
        token=tl.load(X+pos-(ORDER-2)+j,mask=inside&(col>=ORDER-2),other=EOS).to(tl.int64)
        h=((h^(token+1))*PRIME)&((1<<48)-1)
        if j>0:valid=valid&(token!=EOS)
    lo=tl.full((BLOCK,),0,tl.int64);hi=tl.full((BLOCK,),K,tl.int64)
    while tl.sum((lo<hi).to(tl.int32),0)>0:
        mid=(lo+hi)//2;key=tl.load(KEYS+mid,mask=mid<K,other=9223372036854775807)
        move=(lo<hi)&(key<h);hi=tl.where((lo<hi)&(~move),mid,hi);lo=tl.where(move,mid+1,lo)
    found=valid&(lo<K)&(tl.load(KEYS+lo,mask=lo<K,other=-1)==h)
    total=tl.load(TOTALS+lo,mask=found,other=0);found=found&(total>=MINIMUM)
    tl.store(ROWS+pos,tl.where(found,lo,-1),mask=inside)

@triton.jit
def _prior(LOGITS,PROB,ROWS,TOTALS,OFFSET,V:tl.constexpr,TAU:tl.constexpr,MIX:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);ctx=tl.load(ROWS+OFFSET+row)
    if ctx<0:return
    word=tl.arange(0,BLOCK);z=tl.load(LOGITS+(OFFSET+row).to(tl.int64)*V+word,word<V,other=-float('inf')).to(tl.float32)
    exp=tl.exp(z-tl.max(z,0));nn=exp/tl.sum(exp,0)
    total=tl.load(TOTALS+ctx).to(tl.float32);prob=nn*(1.-MIX*total/(total+TAU))
    tl.store(PROB+row.to(tl.int64)*V+word,prob,word<V)

@triton.jit
def _sparse_add(PROB,ROWS,PTR,WORDS,COUNTS,TOTALS,OFFSET,V:tl.constexpr,TAU:tl.constexpr,MIX:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);ctx=tl.load(ROWS+OFFSET+row)
    if ctx<0:return
    start=tl.load(PTR+ctx);end=tl.load(PTR+ctx+1)
    coefficient=MIX/(tl.load(TOTALS+ctx).to(tl.float32)+TAU)
    for at in range(start,end,BLOCK):
        edge=at+tl.arange(0,BLOCK);mask=edge<end
        word=tl.load(WORDS+edge,mask,other=0).to(tl.int64);count=tl.load(COUNTS+edge,mask,other=0).to(tl.float32)
        addr=PROB+row.to(tl.int64)*V+word
        tl.store(addr,tl.load(addr,mask,other=0)+coefficient*count,mask)

@triton.jit
def _finish(LOGITS,PROB,ROWS,OFFSET,V:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);ctx=tl.load(ROWS+OFFSET+row)
    if ctx<0:return
    word=tl.arange(0,BLOCK);p=tl.load(PROB+row.to(tl.int64)*V+word,word<V,other=1.)
    tl.store(LOGITS+(OFFSET+row).to(tl.int64)*V+word,tl.log(p),word<V)

class LongCountHybrid:
    def __init__(self,root,mixing,device,chunk_rows=4096):
        self.root=Path(root);self.meta=json.loads((self.root/'model.json').read_text());self.mixing=dict(mixing)
        self.vocab=self.meta['vocab'];self.eos=self.meta['eos'];self.order=self.meta['order'];self.device=torch.device(device);self.chunk_rows=chunk_rows
        self.lam=float(mixing['lam']);self.tau=float(mixing['tau']);self.minimum=int(mixing['min_context_count'])
        assert 0<=self.lam<=1 and self.tau>0 and self.minimum>=1
        if self.lam==0:return
        def load(name,dtype):
            return torch.from_numpy(np.load(self.root/(name+'.npy'),mmap_mode='r').copy()).to(device=self.device,dtype=dtype)
        self.keys=load('context_keys',torch.int64);self.totals=load('totals',torch.float32)
        self.rowptr=load('rowptr',torch.int64);self.words=load('words',torch.int32);self.values=load('values',torch.float32)
    def contexts(self,inputs):
        x=inputs.to(torch.int64).contiguous();rows=torch.empty(x.numel(),device=x.device,dtype=torch.int64)
        _lookup[(triton.cdiv(x.numel(),128),)](x,self.keys,self.totals,rows,N=x.numel(),S=x.shape[1],K=len(self.keys),EOS=self.eos,ORDER=self.order,SEED=self.meta['hash_seed'],PRIME=self.meta['hash_prime'],MINIMUM=self.minimum,BLOCK=128)
        return rows
    def parameter_upper_bound(self):return self.meta['literal_runtime_scalar_upper_bound']
    def extra_active(self,inputs):
        if self.lam==0:return torch.zeros_like(inputs,dtype=torch.int64)
        rows=self.contexts(inputs);safe=rows.clamp_min(0);edges=self.rowptr[safe+1]-self.rowptr[safe]
        return torch.where(rows>=0,edges+1,0).reshape_as(inputs)
    def __call__(self,inputs,logits):
        assert logits.dtype==torch.bfloat16 and logits.shape==(*inputs.shape,self.vocab) and logits.is_contiguous()
        if self.lam==0:return logits
        flat=logits.reshape(-1,self.vocab);rows=self.contexts(inputs)
        for offset in range(0,len(rows),self.chunk_rows):
            length=min(self.chunk_rows,len(rows)-offset);p=torch.empty((length,self.vocab),device=logits.device,dtype=torch.float32)
            _prior[(length,)](flat,p,rows,self.totals,OFFSET=offset,V=self.vocab,TAU=self.tau,MIX=self.lam,BLOCK=triton.next_power_of_2(self.vocab))
            _sparse_add[(length,)](p,rows,self.rowptr,self.words,self.values,self.totals,OFFSET=offset,V=self.vocab,TAU=self.tau,MIX=self.lam,BLOCK=128)
            _finish[(length,)](flat,p,rows,OFFSET=offset,V=self.vocab,BLOCK=triton.next_power_of_2(self.vocab))
        return logits

def validate_neural_reference(root,config,model_config):
    import hashlib
    root=Path(root);config=Path(config);counts=json.loads((root/'model.json').read_text());c=json.loads(config.read_text())
    summary=json.loads((config.parent/'train/summary.json').read_text());recorded=json.loads((config.parent/'train/run_config.json').read_text())
    assert counts['schema']=='paired_prefix_conditional_branch_v2'
    assert summary['completed_budget'] and summary['tokens_trained']>=counts['targets']
    assert c['train']['seq']==4096 and c['train']['seed']==c['train']['data_seed']==0
    assert recorded['sampling']==counts['sampler'] and recorded['data_identity']==counts['data_identity']
    assert counts['sampler']['sequence_length']==4096 and counts['sampler']['seed']==0 and counts['sampler']['replacement'] is False
    assert counts['targets']==1003945984 and counts['auxiliary_compute_allowance']==counts['targets']*1000000
    proof=json.loads(Path(counts['compute_proof']).read_text());assert proof['targets']==counts['targets'] and proof['auxiliary_allowance']==counts['auxiliary_compute_allowance']
    assert proof['qualification']['passed'] and proof['qualification']['source_sha256']==counts['source_sha256']
    assert counts['source_sha256']==hashlib.sha256((Path(counts['compute_proof']).parent.parent/'paired_count.py').read_bytes()).hexdigest()
    assert model_config['vocab_size']==counts['vocab'] and model_config['eos_token_id']==counts['eos']
    for key in ('hidden_size','num_hidden_layers','num_experts','num_experts_per_tok','moe_intermediate_size','ngram_rows','ngram_max_order','longconv_layers','longconv_channels','longconv_taps'):
        if key in recorded['config']:assert recorded['config'][key]==model_config[key],key
    extra=110000000000000
    combined=summary['train_flops']+counts['auxiliary_compute_allowance']+extra;assert combined<=c['compute']==300000000000000000
    auxiliary=counts['literal_runtime_scalar_upper_bound'];nn_total=c['accounting']['total_params_excluding_embeddings'];nn_active=c['accounting']['active_params_excluding_embeddings'];ratio=nn_active/(nn_total+auxiliary);assert ratio>=.1,ratio
    return dict(neural_config=str(config),same_trained_sample_prefix=True,training_activation_witness='Every count target belongs to identical already executed neural sampler prefix. Independent count updates commute with neural updates; NN forward active parameters therefore lower-bound every count-training target, without additional count-only exposures.',neural_targets=summary['tokens_trained'],count_targets=counts['targets'],auxiliary_compute_allowance=counts['auxiliary_compute_allowance'],extra_calibration_export_allowance=extra,combined_train_compute_upper_bound=combined,auxiliary_literal_scalar_upper_bound=auxiliary,total_nonembedding_parameter_upper_bound=nn_total+auxiliary,train_active_ratio_lower_bound=ratio,counting='Conservatively include all IDs/pointers/counts/totals in model total. Active adds complete selected conditional count row plus total; no targets in forward.')
