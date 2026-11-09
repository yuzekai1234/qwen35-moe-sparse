"""Offline native checkpoint expansion, preserving learned tensors and optimizer state."""
import argparse
import copy
import json
import os
from pathlib import Path
import types
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO
from common import HERE, read, write, source_hash
sys.path.insert(0,str(HERE/'runtime'))


def partition(groups, names, world):
    # Use the installed Torch implementation, not an independently guessed
    # ordering. Local state IDs follow each rank's flattened parameter groups.
    fake=types.SimpleNamespace(param_groups=groups,world_size=world,_partition_parameters_cache=[])
    fake._get_min_index=types.MethodType(ZRO._get_min_index,fake)
    fake._partition_param_group=types.MethodType(ZRO._partition_param_group,fake)
    return [[[names[id(p)] for p in g['params']] for g in rank]
            for rank in ZRO._partition_parameters(fake)]


def layouts(model, optimizer, world):
    names={id(p):n for n,p in model.named_parameters()}
    return {family:partition(getattr(optimizer,family).param_groups,names,world)
            if getattr(optimizer,family) is not None else None for family in ('adam','muon')}


def unpack(states, layout, shapes):
    all_states={};hyper=None;seen=set()
    for rank,(state,groups) in enumerate(zip(states,layout)):
        if len(state['param_groups'])!=len(groups):raise ValueError('Optimizer group count mismatch')
        current=[]
        for group,names in zip(state['param_groups'],groups):
            current.append({k:v for k,v in group.items() if k!='params'})
            if len(group['params'])!=len(names):raise ValueError('Optimizer local ordering mismatch')
            for pid,name in zip(group['params'],names):
                if name in seen:raise ValueError('Duplicate optimizer parameter '+name)
                seen.add(name)
                value=state['state'].get(pid)
                if value is not None:
                    for key,tensor in value.items():
                        if torch.is_tensor(tensor) and tensor.ndim and tuple(tensor.shape)!=shapes[name]:
                            raise ValueError('Optimizer tensor shape mismatch '+name+'/'+key)
                all_states[name]=value
        if hyper is None:hyper=current
        elif current!=hyper:raise ValueError('Optimizer hyperparameters differ across ranks')
    return all_states,hyper


def pack(global_states,hyper,groups):
    result={'state':{},'param_groups':[]};index=0
    for template,names in zip(hyper,groups):
        ids=[]
        for name in names:
            ids.append(index)
            if global_states[name] is not None:result['state'][index]=global_states[name]
            index+=1
        result['param_groups'].append(copy.deepcopy(template)|{'params':ids})
    return result


def equal_state(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def save_tensor_state(path,value):
    temporary=Path(str(path)+'.tmp');torch.save(value,temporary);os.replace(temporary,path)


def expand(source,target,config_path,world):
    source,target,config_path=map(Path,(source,target,config_path))
    target.mkdir(parents=True,exist_ok=False)
    cfg=read(config_path)
    oldcfg=read(source/'run_config.json')
    from q38 import train
    from q38.presets import model_config
    from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM
    from q38.muon import HybridOptimizer
    a=train.parse_args(['--config',str(config_path),'--dry-run'])
    _,hf,fc,_,_,_,_,plan=train.prepare_inputs(a,world)
    for key in train.RESUME_KEYS:
        if oldcfg.get(key,train.NEW_RESUME_DEFAULTS.get(key))!=getattr(a,key):
            raise ValueError('Resume setting changed: '+key)
    if oldcfg['sampling']!=plan['sampling'] or oldcfg['data_identity']!=plan['data_identity']:
        raise ValueError('Corpus or global sampling stream changed')
    ck=torch.load(source/'ckpt.pt',map_location='cpu',weights_only=False,mmap=True)
    oldworld=ck['world'];factor,rem=divmod(world,oldworld)
    if rem or factor<1 or ck['opt'] is not None:raise ValueError('Only integral expansion of native sharded checkpoints is supported')
    if ck['tok_seen']!=(ck['step']+1)*a.batch_tokens:raise ValueError('Step/token mismatch')
    if oldcfg['flops_per_token']!=fc['train_flops_per_token']:raise ValueError('Compute convention changed')
    with torch.device('meta'):model=Qwen3_8MoeForCausalLM(hf)
    for name,p in model.named_parameters():
        if '.longconv.' in name and name.endswith('.taps'):p.requires_grad_(False)
    opt=HybridOptimizer(model,lr=a.lr,weight_decay=a.wd,muon=bool(a.muon),shard=False)
    shapes={n:tuple(p.shape) for n,p in model.named_parameters()}
    oldlayout,newlayout=layouts(model,opt,oldworld),layouts(model,opt,world)
    oldstates=[torch.load(source/f'opt-rank{rank}.pt',map_location='cpu',weights_only=False,mmap=True)
               for rank in range(oldworld)]
    families={}
    for family in ('adam','muon'):
        families[family]=unpack([s[family] for s in oldstates],oldlayout[family],shapes)
    for rank in range(world):
        state={family:pack(*families[family],newlayout[family][rank]) for family in ('adam','muon')}
        save_tensor_state(target/f'opt-rank{rank}.pt',state)
    # Read back every assigned optimizer tensor by its global parameter name.
    for family in ('adam','muon'):
        back=[torch.load(target/f'opt-rank{rank}.pt',map_location='cpu',weights_only=False,mmap=True)[family]
              for rank in range(world)]
        recovered,h=unpack(back,newlayout[family],shapes)
        if h!=families[family][1] or not equal_state(recovered,families[family][0]):
            raise RuntimeError('Optimizer roundtrip differs: '+family)
    for rank in range(oldworld):
        lc=torch.load(source/f'longconv-rank{rank}.pt',map_location='cpu',weights_only=False,mmap=True)
        if lc['world']!=oldworld or lc['rank']!=rank:raise ValueError('LongConv shard identity mismatch')
        for piece in range(factor):
            destination=rank*factor+piece
            state=dict(world=world,rank=destination)
            for key in ('master','buffers'):
                state[key]=[]
                for tensor in lc[key]:
                    if tensor.shape[0]%factor:raise ValueError('Lag shards cannot split evenly')
                    state[key].append(tensor.chunk(factor,dim=0)[piece].clone())
            path=target/f'longconv-rank{destination}.pt';save_tensor_state(path,state)
            check=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            if not equal_state(state,check):raise RuntimeError('LongConv roundtrip differs')
    original_world=ck['world'];ck['world']=world
    # Keep the old RNG records as evidence. The native loader uses its existing
    # world-change fallback when len(rank_rng) != the new world. Sampling is
    # stateless and verified above; bitwise CUDA/reduction equivalence is not claimed.
    save_tensor_state(target/'ckpt.pt',ck)
    restored=torch.load(target/'ckpt.pt',map_location='cpu',weights_only=False,mmap=True)
    if not equal_state(ck['model'],restored['model']):raise RuntimeError('Learned model tensor roundtrip differs')
    for name in ('run_config.json','training_metrics.csv'):
        p=source/name
        if p.exists():
            import shutil
            shutil.copy2(p,target/name)
    proof=dict(passed=True,source=str(source),target=str(target),source_world=original_world,target_world=world,
               step=ck['step'],tokens=ck['tok_seen'],model_tensors_exact=True,
               optimizer_tensors_exact=True,longconv_masters_and_momenta_exact=True,
               global_sampling_unchanged=True,source_hash=source_hash(),new_NN_forwards=0,
               rng_policy='Native different-world fallback; old per-rank RNG states retained. Not bitwise trajectory equivalence.')
    write(target/'RESHARD_PROOF.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--world',type=int,required=True)
    a=p.parse_args();print(json.dumps(expand(a.source,a.target,a.config,a.world),indent=2))
