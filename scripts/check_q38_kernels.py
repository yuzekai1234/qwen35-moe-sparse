#!/usr/bin/env python3
"""Small CUDA parity checks for the profiling optimizations."""
import sys,os,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from q38.presets import model_config
from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM
from q38.inference import padded_batched_experts_forward
from q38.muon import Muon

def main():
 torch.manual_seed(7)
 cfg=model_config(dict(layers=4,hidden=32,heads=2,kv_heads=1,lin_k_heads=1,lin_v_heads=2,experts=8,topk=3,moe_ffn=8,shared_ffn=8,ngram_rows=0,experts_impl='grouped_mm'),128,128,eos_token_id=0)
 lm=Qwen3_8MoeForCausalLM(cfg).cuda();lm.set_experts_implementation('grouped_mm')
 ref=lm.model.layers[0].mlp.experts;fast=copy.deepcopy(ref)
 ids=torch.rand(53,8,device='cuda').topk(3,dim=-1).indices; rw=torch.softmax(torch.randn(53,3,device='cuda'),-1)
 results={}
 for dtype in [torch.float32,torch.bfloat16]:
  x=torch.randn(53,32,device='cuda',requires_grad=True);xf=x.detach().clone().requires_grad_();r=rw.detach().clone().requires_grad_();rf=r.detach().clone().requires_grad_()
  for m in [ref,fast]:m.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
   y=ref(x,ids,r);yf=padded_batched_experts_forward(fast,xf,ids,rf)
  (y.float().square().sum()).backward();(yf.float().square().sum()).backward()
  errs={'output':float((y-yf).abs().max()),'input_grad':float((x.grad-xf.grad).abs().max()),'routing_grad':float((r.grad-rf.grad).abs().max()),'gate_weight_grad':float((ref.gate_up_proj.grad-fast.gate_up_proj.grad).abs().max()),'down_weight_grad':float((ref.down_proj.grad-fast.down_proj.grad).abs().max())}
  for a,b in [(y,yf),(x.grad,xf.grad),(r.grad,rf.grad),(ref.gate_up_proj.grad,fast.gate_up_proj.grad),(ref.down_proj.grad,fast.down_proj.grad)]:torch.testing.assert_close(a,b,rtol=.025 if dtype==torch.bfloat16 else 1e-4,atol=2e-5 if dtype==torch.bfloat16 else 2e-8)
  results[str(dtype)]=errs
 p=torch.nn.Parameter(torch.randn(8,32,16,device='cuda'));pf=torch.nn.Parameter(p.detach().clone());g=torch.randn_like(p);p.grad=g;pf.grad=g.clone()
 a=Muon([p],lr=.001);b=Muon([pf],lr=.001)
 os.environ['Q38_MUON_BASELINE']='1';a.step();os.environ.pop('Q38_MUON_BASELINE');b.step()
 err=float((p-pf).abs().max());torch.testing.assert_close(p,pf,rtol=1e-4,atol=5e-5);results['muon_single_step_max_abs']=err
 # Force bounded chunks both inside a 3-D expert tensor and across parameters.
 import q38.muon as muon_module
 old_limit=muon_module.MUON_MAX_CHUNK_ELEMENTS;muon_module.MUON_MAX_CHUNK_ELEMENTS=1024
 ps=[torch.nn.Parameter(torch.randn(8,32,16,device='cuda')),torch.nn.Parameter(torch.randn(32,16,device='cuda'))]
 pfs=[torch.nn.Parameter(p.detach().clone()) for p in ps]
 for p,pf in zip(ps,pfs):p.grad=torch.randn_like(p);pf.grad=p.grad.clone()
 ref_opt=Muon(ps,lr=.001);chunk_opt=Muon(pfs,lr=.001)
 os.environ['Q38_MUON_BASELINE']='1';ref_opt.step();os.environ.pop('Q38_MUON_BASELINE');chunk_opt.step()
 chunk_error=max(float((p-pf).abs().max()) for p,pf in zip(ps,pfs))
 for p,pf in zip(ps,pfs):
  torch.testing.assert_close(p,pf,rtol=1e-4,atol=5e-5)
  torch.testing.assert_close(ref_opt.state[p]['buf'],chunk_opt.state[pf]['buf'],rtol=0,atol=0)
 muon_module.MUON_MAX_CHUNK_ELEMENTS=old_limit
 results['muon_forced_chunk_max_abs']=chunk_error
 results['muon_chunk_momentum_buffers_identical']=True
 # Full-query and first-query paths must choose identical causal sparse blocks.
 from q38.models.qwen3_8_moe.modeling_qwen3_8_moe import qsa_attention,select_blocks
 lm.eval();indexer=lm.model.indexers['3'];x=torch.randn(1,256,32,device='cuda')
 pos=torch.arange(256,device='cuda').view(1,1,-1).expand(3,1,-1)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  cos,sin=lm.model.rotary_emb(x,pos)
  full=indexer(x,cos,sin);reduced=indexer(x,cos,sin,block_queries=True)
  torch.testing.assert_close(reduced,full[:,::64],rtol=.01,atol=.005)
  ar=torch.arange(256,device='cuda')
  full_lp=full.masked_fill(ar[None,:]>ar[:,None],-1e9).log_softmax(-1).view(1,256,4,64).logsumexp(-1)
  reduced_lp=reduced.masked_fill(ar[None,:]>ar[::64,None],-1e9).log_softmax(-1).view(1,4,4,64).logsumexp(-1)
  for old,new in zip(select_blocks(full_lp,2),select_blocks(reduced_lp,2)):
   assert torch.equal(old,new),'Sparse block selection changed'
  os.environ['Q38_INDEXER_BASELINE']='1'
  old,_=qsa_attention(lm.model.layers[3].self_attn,x,cos,sin,indexer,2,False,False)
  os.environ.pop('Q38_INDEXER_BASELINE')
  new,_=qsa_attention(lm.model.layers[3].self_attn,x,cos,sin,indexer,2,False,False)
  torch.testing.assert_close(old,new,rtol=0,atol=0)
 results['indexer_first_query_max_abs']=float((reduced-full[:,::64]).abs().max())
 results['qsa_selected_blocks_identical']=True
 results['qsa_output_max_abs']=float((old-new).abs().max())
 from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
 h=torch.randn(64,32,device='cuda',requires_grad=True);w=torch.randn(128,32,device='cuda',requires_grad=True)
 empty_target=torch.full((64,),-100,device='cuda',dtype=torch.long)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  empty_loss=LigerFusedLinearCrossEntropyLoss(reduction='sum')(w,h,empty_target)
 empty_loss.backward()
 assert empty_loss==0 and torch.isfinite(h.grad).all() and torch.isfinite(w.grad).all()
 results['all_ignored_mtp_sum_loss']=float(empty_loss)
 Path('audits/q38_kernel_parity.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
