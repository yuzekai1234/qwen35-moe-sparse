"""Two-rank CPU proof: persistent bucket accumulation equals full-batch gradients."""
import contextlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from q38.ddp_memory import accumulate_without_sync, clear_gradients_to_none


def main():
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.manual_seed(9)
    initial=torch.nn.Sequential(torch.nn.Linear(4,6),torch.nn.Tanh(),torch.nn.Linear(6,2)).double()
    import copy
    model=DDP(copy.deepcopy(initial),gradient_as_bucket_view=True,find_unused_parameters=True)
    reference=copy.deepcopy(initial)
    optimizer=torch.optim.SGD(model.parameters(),lr=.03)
    ref_optimizer=torch.optim.SGD(reference.parameters(),lr=.03)
    proofs=[]
    # A new construction/resume boundary, then a steady accumulated step.
    for step in (7,8):
        xs,ys=[],[]
        for r in range(world):
            g=torch.Generator().manual_seed(1000+100*step+r)
            xs.append(torch.randn(3,2,4,generator=g,dtype=torch.float64))
            ys.append(torch.randn(3,2,2,generator=g,dtype=torch.float64))
        for micro in range(3):
            sync=model.no_sync() if accumulate_without_sync(True,step,7,micro,3) else contextlib.nullcontext()
            with sync:
                loss=torch.nn.functional.mse_loss(model(xs[rank][micro]),ys[rank][micro])/3
                loss.backward()
        x=torch.cat([a.reshape(-1,4) for a in xs]);y=torch.cat([a.reshape(-1,2) for a in ys])
        torch.nn.functional.mse_loss(reference(x),y).backward()
        error=max(float((a.grad-b.grad).abs().max()) for a,b in zip(model.module.parameters(),reference.parameters()))
        views=all(p.grad._base is not None for p in model.module.parameters())
        assert error<1e-12,(step,error)
        assert views,'DDP gradients are not bucket views'
        optimizer.step();ref_optimizer.step()
        pointers=[p.grad.data_ptr() for p in model.module.parameters()]
        optimizer.zero_grad(set_to_none=clear_gradients_to_none(True));ref_optimizer.zero_grad()
        assert pointers==[p.grad.data_ptr() for p in model.module.parameters()]
        assert all(torch.count_nonzero(p.grad)==0 for p in model.module.parameters())
        proofs.append(dict(step=step,max_gradient_error=error,all_bucket_views=True,views_preserved_after_zero=True))
    if rank==0:
        print(json.dumps(dict(passed=True,world=world,device='cpu',cuda_initialized=torch.cuda.is_initialized(),steps=proofs)),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
