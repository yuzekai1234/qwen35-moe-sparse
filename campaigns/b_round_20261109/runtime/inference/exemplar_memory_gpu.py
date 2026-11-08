"""Normalized exemplar experts: only shared primary records plus deterministic sorting indices."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch,triton
import triton.language as tl
from paired_count_gpu import _prior,_finish
from causal_cache_gpu import LocalCausalCache

@triton.jit
def _lower_hash(HASH,PERM,SEARCH,K:tl.constexpr,BLOCK:tl.constexpr):
 lo=tl.full((BLOCK,),0,tl.int64);hi=tl.full((BLOCK,),K,tl.int64)
 while tl.sum((lo<hi).to(tl.int32),0)>0:
  mid=(lo+hi)//2;idx=tl.load(PERM+mid,mid<K,other=0).to(tl.int64);key=tl.load(HASH+idx,mid<K,other=9223372036854775807);move=(lo<hi)&(key<SEARCH);hi=tl.where((lo<hi)&(~move),mid,hi);lo=tl.where(move,mid+1,lo)
 return lo

@triton.jit
def _ranges(X,HASH,PERM,STARTS,ENDS,ROWS,TOTALS,N:tl.constexpr,S:tl.constexpr,K:tl.constexpr,ORDER:tl.constexpr,EOS:tl.constexpr,SEED:tl.constexpr,PRIME:tl.constexpr,CAP:tl.constexpr,BLOCK:tl.constexpr):
 pos=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);inside=pos<N;col=pos%S;valid=inside&(col>=ORDER-2);h=tl.full((BLOCK,),SEED&((1<<48)-1),tl.int64)
 for j in range(ORDER-1):
  token=tl.load(X+pos-(ORDER-2)+j,inside&(col>=ORDER-2),other=EOS).to(tl.int64);h=((h^(token+1))*PRIME)&((1<<48)-1)
  if j>0:valid=valid&(token!=EOS)
 start=_lower_hash(HASH,PERM,h,K,BLOCK);end=_lower_hash(HASH,PERM,h+1,K,BLOCK);valid=valid&(end>start)&(end-start<=CAP)
 tl.store(STARTS+pos,tl.where(valid,start,0),inside);tl.store(ENDS+pos,tl.where(valid,end,0),inside);tl.store(ROWS+pos,tl.where(valid,pos,-1),inside);tl.store(TOTALS+pos,tl.where(valid,end-start,0).to(tl.float32),inside)

@triton.jit
def _add_records(PROB,ROWS,STARTS,ENDS,PERM,WORDS,TOTALS,OFFSET,V:tl.constexpr,TAU:tl.constexpr,MIX:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);query=OFFSET+row;valid=tl.load(ROWS+query)>=0
 if not valid:return
 start=tl.load(STARTS+query);end=tl.load(ENDS+query);coefficient=MIX/(tl.load(TOTALS+query)+TAU)
 for at in range(start,end,BLOCK):
  edge=at+tl.arange(0,BLOCK);mask=edge<end;idx=tl.load(PERM+edge,mask,other=0).to(tl.int64);word=tl.load(WORDS+idx,mask,other=0).to(tl.int64)
  tl.atomic_add(PROB+row.to(tl.int64)*V+word,coefficient+tl.zeros((BLOCK,),tl.float32),mask=mask,sem='relaxed')

class ExemplarHead:
 def __init__(self,spec,device,arrays=None):
  self.spec=spec;root=Path(spec['root']);self.meta=json.loads((root/'model.json').read_text()) if arrays is None else arrays['meta'];self.vocab=self.meta['vocab'];self.eos=self.meta['eos'];self.cap=int(spec['cap']);self.lams=spec['lams'];self.orders=self.meta['orders'];self.lam=1.;self.chunk_rows=1024
  if arrays is None:
   self.words=torch.from_numpy(np.load(root/'primary/words.npy',mmap_mode='r').copy()).to(device);self.hashes={};self.perms={}
   for o in self.orders:
    self.hashes[o]=torch.from_numpy(np.load(root/f'primary/hash{o}.npy',mmap_mode='r').copy()).to(device);self.perms[o]=torch.from_numpy(np.load(root/f'indices/order{o}.npy',mmap_mode='r').copy()).to(device)
  else:
   self.words=torch.as_tensor(arrays['words'],dtype=torch.int16,device=device);self.hashes={o:torch.as_tensor(arrays['hashes'][o],dtype=torch.int64,device=device) for o in self.orders};self.perms={o:torch.as_tensor(arrays['perms'][o],dtype=torch.int32,device=device) for o in self.orders}
  self.local=LocalCausalCache(dict(context=1,tau=1.,lam=self.lams[-1]),self.vocab,self.eos)
 def index(self,inputs,order):
  x=inputs.to(torch.int64).contiguous();n=x.numel();start=torch.empty(n,device=x.device,dtype=torch.int64);end=torch.empty_like(start);rows=torch.empty_like(start);totals=torch.empty(n,device=x.device,dtype=torch.float32)
  _ranges[(triton.cdiv(n,128),)](x,self.hashes[order],self.perms[order],start,end,rows,totals,N=n,S=x.shape[1],K=len(self.words),ORDER=order,EOS=self.eos,SEED=self.meta['hash_seed'],PRIME=self.meta['hash_prime'],CAP=self.cap,BLOCK=128)
  return start,end,rows,totals
 def apply_order(self,inputs,logits,order,lam):
  if lam==0:return logits
  start,end,rows,totals=self.index(inputs,order);flat=logits.reshape(-1,self.vocab)
  for offset in range(0,len(rows),self.chunk_rows):
   length=min(self.chunk_rows,len(rows)-offset);p=torch.empty((length,self.vocab),device=logits.device,dtype=torch.float32)
   _prior[(length,)](flat,p,rows,totals,OFFSET=offset,V=self.vocab,TAU=.1,MIX=lam,BLOCK=triton.next_power_of_2(self.vocab))
   _add_records[(length,)](p,rows,start,end,self.perms[order],self.words,totals,OFFSET=offset,V=self.vocab,TAU=.1,MIX=lam,BLOCK=128)
   _finish[(length,)](flat,p,rows,OFFSET=offset,V=self.vocab,BLOCK=triton.next_power_of_2(self.vocab))
  return logits
 def __call__(self,inputs,logits):
  assert logits.dtype==torch.bfloat16 and logits.shape==(*inputs.shape,self.vocab) and logits.is_contiguous()
  for o,lam in zip(self.orders,self.lams):logits=self.apply_order(inputs,logits,o,lam)
  return self.local(inputs,logits)
 def extra_active(self,inputs):
  # No separately learned count weights: every record has constant unit multiplicity.
  # Count ALL structural records in N, but conservatively omit their activity from Na.
  # This yields a conservative upper bound on normalized EvalTime, never an artificial speed gain.
  return torch.full_like(inputs,2,dtype=torch.int64)

def validate_reference(root,config,model_config):
 root=Path(root);m=json.loads((root/'model.json').read_text());cfg=json.loads(Path(config).read_text());rc=json.loads((Path(config).parent/'train/run_config.json').read_text());summary=json.loads((Path(config).parent/'train/summary.json').read_text());assert summary['completed_budget'] and cfg['train']['seed']==cfg['train']['data_seed']==0 and cfg['train']['seq']==4096
 assert m['schema']=='shared_exemplar_five_fields_v1' and m['sampler']==rc['sampling'] and m['data_identity']==rc['data_identity'];assert m['targets']<=summary['tokens_trained'] and m['logical_scalar_upper_bound']==5*m['targets'];assert m['combined_train_compute_upper_bound']==summary['train_flops']+1450000000000000<=3e17
 q=json.loads((root/'cpu-qualification.json').read_text());assert q['passed'] and q['source_sha256']==m['source_sha256'];assert hashlib.sha256((root.parent/'exemplar_memory_cpu.py').read_bytes()).hexdigest()==q['source_sha256'];assert model_config['vocab_size']==m['vocab'] and model_config['eos_token_id']==m['eos']
 assert {p.name for p in (root/'primary').glob('*.npy')}=={'words.npy',*[f'hash{o}.npy' for o in m['orders']]};assert {p.name for p in (root/'indices').glob('*.npy')}=={f'order{o}.npy' for o in m['orders']}
 n=cfg['accounting']['total_params_excluding_embeddings']+5*m['targets']+256;ratio=cfg['accounting']['active_params_excluding_embeddings']/n;assert ratio>=.1
 return dict(neural_config=str(config),same_trained_sample_prefix=True,training_activation_witness='Every retained exemplar target belongs to actual identical neural-trained seed0 sampler prefix; unit-record insertion commutes with independent NN updates.',combined_train_compute_upper_bound=m['combined_train_compute_upper_bound'],total_nonembedding_parameter_upper_bound=n,train_active_ratio_lower_bound=ratio,count_targets=m['targets'],neural_targets=summary['tokens_trained'],parameter_counting=m['parameter_contract'],evaltime_normalizer='Conservative NN-only active weights plus2 frozen local scalars; structural primary memory fields included in totalN but omitted from Na. Reported EvalTime is therefore an upper bound if structural activity were counted.')
