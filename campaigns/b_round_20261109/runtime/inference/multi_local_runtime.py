"""Qualified exemplar module followed by several separately causal input-only local caches."""
import torch
from exemplar_memory_gpu import ExemplarHead
from causal_cache_gpu import LocalCausalCache
class MultiLocalExemplarHead:
 def __init__(self,spec,device,base=None):
  self.spec=spec;self.base=ExemplarHead(dict(spec,lams=spec['static_lams']+[0.]),device) if base is None else base;self.vocab=self.base.vocab;self.eos=self.base.eos;self.lam=1.;self.locals=[LocalCausalCache(m,self.vocab,self.eos) for m in spec['local_models']]
 def __call__(self,inputs,logits):
  logits=self.base(inputs,logits)
  for head in self.locals:logits=head(inputs,logits)
  return logits
 def extra_active(self,inputs):
  # Count all frozen active local mixture/smoothing scalars; structural records excluded
  # from Na conservatively, exactly as in the qualified base exemplar model.
  return torch.full_like(inputs,2*sum(h.lam!=0 for h in self.locals),dtype=torch.int64)
