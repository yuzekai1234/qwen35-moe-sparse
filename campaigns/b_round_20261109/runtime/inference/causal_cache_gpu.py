"""Causal sequence-local counts using stable context sorting; no stored evaluation data."""
import torch,triton
import triton.language as tl
from paired_count_gpu import _prior,_finish

@triton.jit
def _add_prefix(INPUT,PROB,ROWS,RANK,STARTS,ORDER,OFFSET,V:tl.constexpr,TAU:tl.constexpr,MIX:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);ctx=tl.load(ROWS+OFFSET+row)
 if ctx<0:return
 rank=tl.load(RANK+OFFSET+row);start=tl.load(STARTS+rank)
 coefficient=MIX/((rank-start).to(tl.float32)+TAU)
 for at in range(start,rank,BLOCK):
  index=at+tl.arange(0,BLOCK);mask=index<rank
  past=tl.load(ORDER+index,mask,other=0).to(tl.int64)
  # Stable context order guarantees past < query in the same input sequence.
  word=tl.load(INPUT+past+1,mask,other=0).to(tl.int64)
  tl.atomic_add(PROB+row.to(tl.int64)*V+word,coefficient,mask,sem='relaxed')

class LocalCausalCache:
 def __init__(self,mixing,vocab,eos,chunk_rows=4096):
  self.context=int(mixing['context']);self.lam=float(mixing['lam']);self.tau=float(mixing['tau']);self.vocab=int(vocab);self.eos=int(eos);self.chunk_rows=chunk_rows
  assert self.context>=1 and 0<=self.lam<=1 and self.tau>0
 def index(self,inputs):
  x=inputs.to(torch.int64).contiguous();flat=x.reshape(-1);n=len(flat);seq=x.shape[1];position=torch.arange(n,device=x.device,dtype=torch.int64);column=position%seq
  if self.context==1:keys=flat+(position//seq)*self.vocab;valid=torch.ones(n,device=x.device,dtype=torch.bool)
  else:
   valid=column>=self.context-1;h=torch.full_like(position,1469598103934665603&((1<<48)-1))
   for j in range(self.context):
    token=flat[(position-self.context+1+j).clamp_min(0)];h=((h^(token+1))*1099511628211)&((1<<48)-1)
    if j>0:valid&=token!=self.eos
   keys=torch.where(valid,h+(position//seq)*(1<<48),-1)
  ordered_keys,order=torch.sort(keys,stable=True);rank=torch.empty_like(order);rank.scatter_(0,order,position)
  boundary=torch.ones(n,device=x.device,dtype=torch.bool);boundary[1:]=ordered_keys[1:]!=ordered_keys[:-1]
  starts=torch.cummax(torch.where(boundary,position,0),dim=0).values
  totals=torch.where(valid,rank-starts[rank],0);rows=torch.where(totals>0,position,-1)
  return x,rows,totals.to(torch.float32),rank,starts,order
 def __call__(self,inputs,logits):
  assert logits.dtype==torch.bfloat16 and logits.shape==(*inputs.shape,self.vocab) and logits.is_contiguous()
  if self.lam==0:return logits
  x,rows,totals,rank,starts,order=self.index(inputs);flat=logits.reshape(-1,self.vocab)
  for offset in range(0,len(rows),self.chunk_rows):
   length=min(self.chunk_rows,len(rows)-offset);prob=torch.empty((length,self.vocab),device=logits.device,dtype=torch.float32)
   _prior[(length,)](flat,prob,rows,totals,OFFSET=offset,V=self.vocab,TAU=self.tau,MIX=self.lam,BLOCK=triton.next_power_of_2(self.vocab))
   _add_prefix[(length,)](x,prob,rows,rank,starts,order,OFFSET=offset,V=self.vocab,TAU=self.tau,MIX=self.lam,BLOCK=128)
   _finish[(length,)](flat,prob,rows,OFFSET=offset,V=self.vocab,BLOCK=triton.next_power_of_2(self.vocab))
  return logits
