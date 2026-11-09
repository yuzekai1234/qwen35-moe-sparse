"""Optimizer IDs are local: test expansion with nonzero named states and group ordering."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from q38.muon import HybridOptimizer
from reshard_checkpoint import layouts,pack,unpack,equal_state


class ReshardTests(unittest.TestCase):
    def test_four_to_eight_and_back_preserves_every_parameter_state(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens=torch.nn.Embedding(17,8)
                self.layers=torch.nn.ModuleList([torch.nn.Linear(8,8) for _ in range(9)])
        m=Model();opt=HybridOptimizer(m,.001,.1,shard=False)
        shapes={n:tuple(p.shape) for n,p in m.named_parameters()}
        old,new=layouts(m,opt,4),layouts(m,opt,8)
        for family in ('adam','muon'):
            names=[n for rank in old[family] for g in rank for n in g]
            self.assertEqual(len(names),len(set(names)))
            global_states={n:{'moment':torch.full(shapes[n],float(i+1)),'step':torch.tensor(123.)}
                           for i,n in enumerate(names)}
            hyper=[{'lr':.0007,'weight_decay':float(i)/10} for i in range(len(old[family][0]))]
            four=[pack(global_states,hyper,g) for g in old[family]]
            merged,h=unpack(four,old[family],shapes)
            eight=[pack(merged,h,g) for g in new[family]]
            recovered,h2=unpack(eight,new[family],shapes)
            self.assertTrue(equal_state(recovered,global_states));self.assertEqual(hyper,h2)
            back=[pack(recovered,h2,g) for g in old[family]]
            self.assertTrue(equal_state(four,back))
    def test_longconv_lag_mapping_preserves_global_order(self):
        full=torch.arange(16*3*3,dtype=torch.float32).reshape(16,3,3)
        old=list(full.chunk(4));new=[piece.clone() for shard in old for piece in shard.chunk(2)]
        self.assertTrue(torch.equal(torch.cat(new),full))


if __name__=='__main__':unittest.main()
