"""Two variable-length prefix experts sharing one immutable canonical memory."""
import copy,hashlib,json
from pathlib import Path
import torch
import triton
import triton.language as tl
import canonical_lmc_gpu as original
from canonical_lmc_gpu import CanonicalLMCHead,_search,_mix,_prior,_add,_finish
from causal_cache_gpu import LocalCausalCache
from document_cache_gpu import DocumentLocalCache

def validate_reference(root,config,model_config):
 q=json.loads((Path(root).parent/'lmc_backoff/gpu-qualification.json').read_text())
 assert q['passed'] and q['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 assert q['base_source_sha256']==hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()
 return original.validate_reference(root,config,model_config)

@triton.jit
def _short_ranges(TX,TH,PERM,Q,LONG_LENGTH,LAMS,ROWS,TOTALS,STARTS,ENDS,LENGTHS,N:tl.constexpr,K:tl.constexpr,CAP:tl.constexpr,MAXCTX:tl.constexpr,BLOCK:tl.constexpr):
 q=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);inside=q<N;length=tl.maximum(tl.load(LONG_LENGTH+q,inside,other=0).to(tl.int32)-1,0);mask=inside&(length>0);lo=_search(TX,TH,PERM,Q,q,length,mask,K,True,False,MAXCTX,BLOCK);hi=_search(TX,TH,PERM,Q,q,length,mask,K,True,True,MAXCTX,BLOCK);n=hi-lo;mix=_mix(length,LAMS);valid=mask&(n>0)&(n<=CAP)&(mix>0)
 tl.store(ROWS+q,tl.where(valid,q,-1),inside);tl.store(TOTALS+q,tl.where(valid,n,0).to(tl.float32),inside);tl.store(STARTS+q,lo,inside);tl.store(ENDS+q,hi,inside);tl.store(LENGTHS+q,length.to(tl.uint8),inside)

def apply(head,inputs,logits,index):
 rows,totals,start,end,length=index;flat=logits.reshape(-1,head.vocab)
 for offset in range(0,len(rows),head.chunk_rows):
  n=min(head.chunk_rows,len(rows)-offset);p=torch.empty((n,head.vocab),device=logits.device,dtype=torch.float32)
  _prior[(n,)](flat,p,rows,totals,length,head.lams,offset,V=head.vocab,BLOCK=triton.next_power_of_2(head.vocab))
  _add[(n,)](p,rows,totals,start,end,length,head.lams,head.perm,head.words,offset,V=head.vocab,BLOCK=128)
  _finish[(n,)](flat,p,rows,OFFSET=offset,V=head.vocab,BLOCK=triton.next_power_of_2(head.vocab))
 return logits

class BackoffLMCHead:
 def __init__(self,spec,device,arrays=None):
  self.spec=spec;self.base=CanonicalLMCHead(dict(spec,local_models=[]),device,arrays=arrays);self.short=copy.copy(self.base);self.short.lams=torch.tensor(spec['short_lams'],device=device,dtype=torch.float32);self.vocab=self.base.vocab;self.eos=self.base.eos;self.lam=1.;self.locals=[(DocumentLocalCache if m['reset_document'] else LocalCausalCache)(m,self.vocab,self.eos) for m in spec.get('local_models',[])]
 def indices(self,inputs):
  long=self.base.index(inputs);n=inputs.numel();rows=torch.empty(n,device=inputs.device,dtype=torch.int64);totals=torch.empty(n,device=inputs.device,dtype=torch.float32);start=torch.empty_like(rows);end=torch.empty_like(rows);length=torch.empty(n,device=inputs.device,dtype=torch.uint8)
  _short_ranges[(triton.cdiv(n,32),)](self.base.inputs,self.base.history,self.base.perm,inputs.to(torch.int64).contiguous(),long[-1],self.short.lams,rows,totals,start,end,length,N=n,K=len(self.base.words),CAP=self.base.cap,MAXCTX=self.base.maxctx,BLOCK=32,num_warps=1)
  return (rows,totals,start,end,length),long
 def __call__(self,inputs,logits):
  assert logits.dtype==torch.bfloat16 and logits.is_contiguous() and logits.shape==(*inputs.shape,self.vocab)
  short,long=self.indices(inputs);logits=apply(self.short,inputs,logits,short);logits=apply(self.base,inputs,logits,long)
  for head in self.locals:logits=head(inputs,logits)
  return logits
 def extra_active(self,inputs):return torch.zeros_like(inputs,dtype=torch.int64)
