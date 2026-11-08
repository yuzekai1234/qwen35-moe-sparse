"""Input-only exact longest-context memory over canonical training-token parameters."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl
from paired_count_gpu import _finish
from causal_cache_gpu import LocalCausalCache
from document_cache_gpu import DocumentLocalCache

@triton.jit
def _compare(TX,TH,Q,TP,QP,QL,MASK,PREFIX:tl.constexpr,MAXCTX:tl.constexpr,BLOCK:tl.constexpr):
 result=tl.full((BLOCK,),0,tl.int32);length=tl.load(TH+TP,MASK,other=0).to(tl.int32);active=MASK;j=0
 while (j<MAXCTX+1)&(tl.sum(active.to(tl.int32),0)>0):
  endq=j>=QL;endt=j>=length
  result=tl.where(active&(~endq)&endt,-1,result)
  if not PREFIX:result=tl.where(active&endq&(~endt),1,result)
  active=active&(~endq)&(~endt)
  x=tl.load(TX+TP-j,active,other=0).to(tl.int32);q=tl.load(Q+QP-j,active,other=0).to(tl.int32)
  result=tl.where(active&(x<q),-1,result);result=tl.where(active&(x>q),1,result);active=active&(x==q);j+=1
 return result

@triton.jit
def _search(TX,TH,PERM,Q,QP,QL,MASK,K:tl.constexpr,PREFIX:tl.constexpr,UPPER:tl.constexpr,MAXCTX:tl.constexpr,BLOCK:tl.constexpr):
 lo=tl.full((BLOCK,),0,tl.int64);hi=tl.full((BLOCK,),K,tl.int64)
 while tl.sum(((lo<hi)&MASK).to(tl.int32),0)>0:
  active=(lo<hi)&MASK;mid=(lo+hi)//2;pos=tl.load(PERM+mid,active,other=0).to(tl.int64)
  cmp=_compare(TX,TH,Q,pos,QP,QL,active,PREFIX,MAXCTX,BLOCK)
  move=active&((cmp<0)|((cmp==0)&UPPER));hi=tl.where(active&(~move),mid,hi);lo=tl.where(move,mid+1,lo)
 return lo

@triton.jit
def _common(TX,TH,Q,TP,QP,QL,MASK,MAXCTX:tl.constexpr,BLOCK:tl.constexpr):
 out=tl.full((BLOCK,),0,tl.int32);length=tl.minimum(tl.load(TH+TP,MASK,other=0).to(tl.int32),QL);active=MASK;j=0
 while (j<MAXCTX)&(tl.sum(active.to(tl.int32),0)>0):
  active=active&(j<length);x=tl.load(TX+TP-j,active,other=0).to(tl.int32);q=tl.load(Q+QP-j,active,other=0).to(tl.int32);active=active&(x==q);out=tl.where(active,j+1,out);j+=1
 return out

@triton.jit
def _mix(LENGTH,LAMS):
 group=(LENGTH>=2).to(tl.int32)+(LENGTH>=4).to(tl.int32)+(LENGTH>=6).to(tl.int32)+(LENGTH>=8).to(tl.int32)+(LENGTH>=12).to(tl.int32)+(LENGTH>=20).to(tl.int32)+(LENGTH>=32).to(tl.int32)
 return tl.load(LAMS+group)

@triton.jit
def _ranges(TX,TH,PERM,Q,QH,LAMS,ROWS,TOTALS,STARTS,ENDS,LENGTHS,N:tl.constexpr,K:tl.constexpr,CAP:tl.constexpr,MAXCTX:tl.constexpr,BLOCK:tl.constexpr):
 qp=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);inside=qp<N;ql=tl.load(QH+qp,inside,other=0).to(tl.int32)
 at=_search(TX,TH,PERM,Q,qp,ql,inside,K,False,False,MAXCTX,BLOCK);length=tl.full((BLOCK,),0,tl.int32)
 for delta in tl.static_range(-1,1):
  neighbor=at+delta;valid=inside&(neighbor>=0)&(neighbor<K);tp=tl.load(PERM+neighbor,valid,other=0).to(tl.int64);l=_common(TX,TH,Q,tp,qp,ql,valid,MAXCTX,BLOCK);length=tl.maximum(length,l)
 mask=inside&(length>0);lo=_search(TX,TH,PERM,Q,qp,length,mask,K,True,False,MAXCTX,BLOCK);hi=_search(TX,TH,PERM,Q,qp,length,mask,K,True,True,MAXCTX,BLOCK);n=hi-lo;mix=_mix(length,LAMS);valid=mask&(n>0)&(n<=CAP)&(mix>0)
 tl.store(ROWS+qp,tl.where(valid,qp,-1),inside);tl.store(TOTALS+qp,tl.where(valid,n,0).to(tl.float32),inside);tl.store(STARTS+qp,lo,inside);tl.store(ENDS+qp,hi,inside);tl.store(LENGTHS+qp,length.to(tl.uint8),inside)

@triton.jit
def _prior(LOGITS,PROB,ROWS,TOTALS,LENGTHS,LAMS,OFFSET,V:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);query=OFFSET+row;valid=tl.load(ROWS+query)>=0
 if not valid:return
 word=tl.arange(0,BLOCK);z=tl.load(LOGITS+query.to(tl.int64)*V+word,word<V,other=-float('inf')).to(tl.float32);exp=tl.exp(z-tl.max(z,0));nn=exp/tl.sum(exp,0);n=tl.load(TOTALS+query);mix=_mix(tl.load(LENGTHS+query).to(tl.int32),LAMS);p=nn*(1-mix*n/(n+.1));tl.store(PROB+row.to(tl.int64)*V+word,p,word<V)

@triton.jit
def _add(PROB,ROWS,TOTALS,STARTS,ENDS,LENGTHS,LAMS,PERM,WORDS,OFFSET,V:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);query=OFFSET+row;valid=tl.load(ROWS+query)>=0
 if not valid:return
 start=tl.load(STARTS+query);end=tl.load(ENDS+query);mix=_mix(tl.load(LENGTHS+query).to(tl.int32),LAMS);coefficient=mix/(tl.load(TOTALS+query)+.1)
 for at in range(start,end,BLOCK):
  edge=at+tl.arange(0,BLOCK);mask=edge<end;position=tl.load(PERM+edge,mask,other=0).to(tl.int64);word=tl.load(WORDS+position,mask,other=0).to(tl.int64);tl.atomic_add(PROB+row.to(tl.int64)*V+word,coefficient+tl.zeros((BLOCK,),tl.float32),mask,sem='relaxed')

class CanonicalLMCHead:
 def __init__(self,spec,device,arrays=None):
  self.spec=spec;self.vocab=int(spec.get('vocab',10112));self.eos=int(spec.get('eos',53));self.maxctx=int(spec.get('max_context',64));self.cap=int(spec['cap']);self.lam=1.;self.chunk_rows=1024
  assert spec['length_thresholds']==[2,4,6,8,12,20,32] and len(spec['lmc_lams'])==8 and 0<self.maxctx<=64
  self.lams=torch.tensor(spec['lmc_lams'],device=device,dtype=torch.float32)
  if arrays is None:
   root=Path(spec['root'])
   def load(name):return torch.from_numpy(np.load(root/name,mmap_mode='r').copy()).to(device)
   words=np.load(root/'primary/words.npy',mmap_mode='r');first=np.load(root/'primary/first.npy',mmap_mode='r');cached_x=np.load(root/'cache/inputs.npy',mmap_mode='r');cached_h=np.load(root/'cache/history.npy',mmap_mode='r');block=4096
   # Reconstruct and check every cached input/boundary from canonical model tensors.
   for at in range(0,len(first),1024):
    end=min(len(first),at+1024);y=words[at*block:end*block].reshape(-1,block);x=np.empty_like(y);x[:,0]=first[at:end];x[:,1:]=y[:,:-1];col=np.arange(block,dtype=np.int32)[None,:];last=np.maximum.accumulate(np.where((x==self.eos)|(col==0),col,0),axis=1);h=np.minimum(col-last+1,self.maxctx).astype(np.uint8)
    np.testing.assert_array_equal(cached_x[at*block:end*block],x.ravel());np.testing.assert_array_equal(cached_h[at*block:end*block],h.ravel())
   self.words=load('primary/words.npy');self.inputs=load('cache/inputs.npy');self.history=load('cache/history.npy');self.perm=load('cache/suffix64.npy')
  else:
   self.words=torch.as_tensor(arrays['words'],device=device,dtype=torch.int16);self.inputs=torch.as_tensor(arrays['inputs'],device=device,dtype=torch.int16);self.history=torch.as_tensor(arrays['history'],device=device,dtype=torch.uint8);self.perm=torch.as_tensor(arrays['perm'],device=device,dtype=torch.int32)
  self.locals=[(DocumentLocalCache if m['reset_document'] else LocalCausalCache)(m,self.vocab,self.eos) for m in spec.get('local_models',[])]
 def index(self,inputs):
  x=inputs.to(torch.int64).contiguous();flat=x.reshape(-1);n=len(flat);position=torch.arange(n,device=x.device,dtype=torch.int64);column=position%x.shape[1];boundary=torch.where((flat==self.eos)|(column==0),position,0);last=torch.cummax(boundary,0).values;qh=(position-last+1).clamp_max(self.maxctx).to(torch.uint8)
  rows=torch.empty(n,device=x.device,dtype=torch.int64);totals=torch.empty(n,device=x.device,dtype=torch.float32);start=torch.empty_like(rows);end=torch.empty_like(rows);length=torch.empty(n,device=x.device,dtype=torch.uint8)
  _ranges[(triton.cdiv(n,32),)](self.inputs,self.history,self.perm,x,qh,self.lams,rows,totals,start,end,length,N=n,K=len(self.words),CAP=self.cap,MAXCTX=self.maxctx,BLOCK=32,num_warps=1)
  return rows,totals,start,end,length
 def __call__(self,inputs,logits):
  assert logits.dtype==torch.bfloat16 and logits.is_contiguous() and logits.shape==(*inputs.shape,self.vocab)
  rows,totals,start,end,length=self.index(inputs);flat=logits.reshape(-1,self.vocab)
  for offset in range(0,len(rows),self.chunk_rows):
   n=min(self.chunk_rows,len(rows)-offset);p=torch.empty((n,self.vocab),device=logits.device,dtype=torch.float32)
   _prior[(n,)](flat,p,rows,totals,length,self.lams,offset,V=self.vocab,BLOCK=triton.next_power_of_2(self.vocab))
   _add[(n,)](p,rows,totals,start,end,length,self.lams,self.perm,self.words,offset,V=self.vocab,BLOCK=128)
   _finish[(n,)](flat,p,rows,OFFSET=offset,V=self.vocab,BLOCK=triton.next_power_of_2(self.vocab))
  for head in self.locals:logits=head(inputs,logits)
  return logits
 def extra_active(self,inputs):
  # Omit every statistical field/scalar fromNa: strictly conservative NN-only normalization.
  return torch.zeros_like(inputs,dtype=torch.int64)

def validate_reference(root,config,model_config):
 root=Path(root);config=Path(config);m=json.loads((root/'model.json').read_text());cfg=json.loads(config.read_text());summary=json.loads((config.parent/'train/summary.json').read_text());rc=json.loads((config.parent/'train/run_config.json').read_text());q=json.loads((root/'cpu-qualification.json').read_text());query=json.loads((root/'query-qualification.json').read_text())
 assert m['schema']=='canonical_token_lmc_v1' and q['passed'] and query['passed']
 assert hashlib.sha256((root.parent/'canonical_lmc_cpu.py').read_bytes()).hexdigest()==q['source_sha256']==m['source_sha256']
 assert summary['completed_budget'] and cfg['train']['seed']==cfg['train']['data_seed']==0 and cfg['train']['seq']==4096
 assert m['sampler']==rc['sampling'] and m['data_identity']==rc['data_identity'];assert m['targets']<=summary['tokens_trained'] and m['targets']==m['blocks']*m['block_length']
 assert m['canonical_scalar_count']==m['targets']+m['blocks'];assert m['combined_train_compute_upper_bound']==summary['train_flops']+1450000000000000<=3e17
 assert model_config['vocab_size']==m['vocab'] and model_config['eos_token_id']==m['eos'];assert {p.name for p in (root/'primary').glob('*.npy')}=={'words.npy','first.npy'}
 n=cfg['accounting']['total_params_excluding_embeddings']+m['canonical_scalar_count']+256;ratio=cfg['accounting']['active_params_excluding_embeddings']/n;assert ratio>=.1
 return dict(neural_config=str(config),same_trained_sample_prefix=True,training_activation_witness='Every canonical recordtarget and initialcontext field belongs to the actual identical NN-trained seed0 global shuffled prefix; no extra target exposures.',combined_train_compute_upper_bound=m['combined_train_compute_upper_bound'],total_nonembedding_parameter_upper_bound=n,train_active_ratio_lower_bound=ratio,count_targets=m['targets'],neural_targets=summary['tokens_trained'],parameter_counting=m['parameter_contract'],evaltime_normalizer='Conservative NN-only active weights; all independently counted canonical IDs included in totalN; statistical fields and scalars omitted fromNa. EvalTime is an upperbound under this declared researchcontract.',parameter_organization_requires_organizer_acceptance=True)
