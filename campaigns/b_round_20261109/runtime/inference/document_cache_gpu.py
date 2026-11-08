"""Exact per-document causal cache using stable lexicographic group/context sorting."""
import copy
import torch
from exemplar_memory_gpu import ExemplarHead
from causal_cache_gpu import LocalCausalCache
from multi_local_runtime import MultiLocalExemplarHead
class DocumentLocalCache(LocalCausalCache):
 def index(self,inputs):
  x=inputs.to(torch.int64).contiguous();flat=x.reshape(-1);n=len(flat);seq=x.shape[1];position=torch.arange(n,device=x.device,dtype=torch.int64);column=position%seq;group=((flat==self.eos)|(column==0)).to(torch.int64).cumsum(0)
  if self.context==1:h=flat;valid=torch.ones(n,device=x.device,dtype=torch.bool)
  else:
   valid=column>=self.context-1;h=torch.full_like(position,1469598103934665603&((1<<48)-1))
   for j in range(self.context):
    token=flat[(position-self.context+1+j).clamp_min(0)];h=((h^(token+1))*1099511628211)&((1<<48)-1)
    if j>0:valid&=token!=self.eos
   h=torch.where(valid,h,-1)
  # Two stable sorts give exact (documentID,contextID,time) order without bitpacking IDs.
  by_hash=torch.argsort(h,stable=True);order=by_hash[torch.argsort(group[by_hash],stable=True)];rank=torch.empty_like(order);rank.scatter_(0,order,position);oh=h[order];og=group[order];boundary=torch.ones(n,device=x.device,dtype=torch.bool);boundary[1:]=(oh[1:]!=oh[:-1])|(og[1:]!=og[:-1]);starts=torch.cummax(torch.where(boundary,position,0),0).values;totals=torch.where(valid,rank-starts[rank],0);rows=torch.where(totals>0,position,-1)
  return x,rows,totals.to(torch.float32),rank,starts,order
class DocumentExemplarHead(MultiLocalExemplarHead):
 def __init__(self,spec,device,base=None):
  self.spec=spec
  if base is None:self.base=ExemplarHead(dict(spec,lams=spec['static_lams']+[0.]),device)
  else:self.base=copy.copy(base);self.base.lams=spec['static_lams']+[0.]
  self.vocab=self.base.vocab;self.eos=self.base.eos;self.lam=1.;self.locals=[(DocumentLocalCache if m['reset_document'] else LocalCausalCache)(m,self.vocab,self.eos) for m in spec['local_models']]
