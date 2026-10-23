#!/usr/bin/env python3
"""Short, reproducible CUDA training/inference benchmark; no corpus or checkpoints needed."""
import argparse, json, time, statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from q38.presets import model_config,load_preset
from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM
from q38.train import LossModel,install_fast_paths
from q38.inference import optimize_for_inference
from q38.muon import HybridOptimizer


def main():
 p=argparse.ArgumentParser();p.add_argument('--preset',default='');p.add_argument('--seq',type=int,default=4096);p.add_argument('--micro-bs',type=int,default=2);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeat',type=int,default=3);p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true');p.add_argument('--hidden',type=int,default=128);p.add_argument('--layers',type=int,default=4);p.add_argument('--grad-ckpt',action='store_true');a=p.parse_args()
 torch.manual_seed(123);torch.cuda.set_device(0)
 shape=load_preset(a.preset)['model'] if a.preset else dict(hidden=a.hidden,layers=a.layers,heads=4,kv_heads=2,lin_k_heads=2,lin_v_heads=6,experts=64,topk=5,moe_ffn=a.hidden//4,shared_ffn=a.hidden//4,ngram_rows=4096,experts_impl='grouped_mm')
 cfg=model_config(shape,10112,a.seq,eos_token_id=53)
 m=Qwen3_8MoeForCausalLM(cfg).cuda().train();m.model.gradient_checkpointing=a.grad_ckpt;fast=install_fast_paths(m);loss_model=LossModel(m,cfg.num_experts,cfg.num_experts_per_tok);opt=HybridOptimizer(m,lr=1e-3,weight_decay=.1)
 ids=torch.randint(0,10112,(a.micro_bs,a.seq),device='cuda');labels=torch.randint(0,10112,ids.shape,device='cuda')
 timings=[];warmup_timings=[]
 def sync_time():torch.cuda.synchronize();return time.perf_counter()
 def train_step():
  t0=sync_time()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   ce,aux,mtp,kl=loss_model(ids,labels);loss=ce+.001*aux+.3*mtp+.05*kl
  assert not loss_model._router, "Router hooks retained a training graph"
  t1=sync_time();loss.backward();t2=sync_time();opt.step();opt.zero_grad();t3=sync_time()
  return dict(forward_ms=(t1-t0)*1000,backward_ms=(t2-t1)*1000,optimizer_ms=(t3-t2)*1000,total_ms=(t3-t0)*1000,loss=float(loss.detach()))
 start=time.perf_counter()
 for i in range(a.warmup):
  r=train_step();warmup_timings.append(r);print('train warmup',i,r,flush=True)
 for i in range(a.repeat):
  r=train_step();timings.append(r);print('train measure',i,r,flush=True)
 train_peak=torch.cuda.max_memory_allocated()
 prof_table=''
 if a.profile:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:train_step()
  prof_table=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=20);print(prof_table,flush=True)
  Path(a.output).with_suffix('.train_profile.txt').write_text('\n'.join(line.rstrip() for line in prof_table.splitlines())+'\n')
 del loss_model,opt;m.zero_grad(set_to_none=True);m.to(torch.bfloat16).eval();infer_fast=optimize_for_inference(m)
 inf=[]
 with torch.inference_mode():
  for i in range(a.warmup+a.repeat):
   t0=sync_time();out=m(input_ids=ids,use_cache=False).logits;t1=sync_time();dt=(t1-t0)*1000
   assert torch.isfinite(out).all();del out
   if i>=a.warmup:inf.append(dt)
   print('infer',i,dt,flush=True)
 train={k:statistics.median([r[k] for r in timings]) for k in timings[0]};train['tokens_per_second']=a.micro_bs*a.seq/(train['total_ms']/1000)
 result=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,shape=shape,seq=a.seq,micro_bs=a.micro_bs,grad_ckpt=a.grad_ckpt,warmup=a.warmup,repeat=a.repeat,training=train,inference=dict(median_ms=statistics.median(inf),tokens_per_second=a.micro_bs*a.seq/(statistics.median(inf)/1000)),training_peak_bytes=train_peak,optimizer_step_every_microbatch=True,includes_data_loading=False,includes_gradient_clipping=False,fast_paths=fast,inference_fast_paths=infer_fast,wall_seconds=time.perf_counter()-start,warmup_training=warmup_timings,raw_training=timings,raw_inference_ms=inf)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
