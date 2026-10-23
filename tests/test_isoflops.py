"""CPU-only checks for stream fairness, resume, validation and experiment plans."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from q38.presets import ROOT, load_training_config
from q38.sampling import ShuffledBlockSampler, sequential_windows
from q38.train import evaluate, lr_schedule, main, micro_bs_candidates, parse_args, prepare_inputs


class Sampling(unittest.TestCase):
    def test_global_stream_is_independent_of_rank_and_micro_batch(self):
        sampler = ShuffledBlockSampler(4097, 8, 7)
        expected = sampler.offsets(0, 32)
        for world, micro_bs in [(1,1),(1,4),(1,8),(2,1),(2,2),(4,1),(8,1)]:
            actual = []
            for step in range(4):
                for micro in range(64//(world*micro_bs*8)):
                    for rank in range(world):
                        actual.extend(sampler.micro_offsets(step,micro,rank,world,micro_bs,64))
            np.testing.assert_array_equal(actual,expected)
        targets = np.concatenate([np.arange(i+1,i+9) for i in expected])
        self.assertEqual(len(np.unique(targets)),len(targets))

    def test_resume_and_no_epoch_wrapping(self):
        a=ShuffledBlockSampler(4097,8,42)
        b=ShuffledBlockSampler(4097,8,42)
        self.assertEqual(a.sha256,b.sha256)
        np.testing.assert_array_equal(a.offsets(80,8),b.micro_offsets(10,0,0,1,8,64))
        self.assertFalse(np.array_equal(a.order,ShuffledBlockSampler(4097,8,43).order))
        with self.assertRaises(ValueError):a.offsets(a.blocks-1,2)
        with self.assertRaises(ValueError):a.micro_offsets(0,0,0,3,1,64)

    def test_fixed_evaluation_scores_tail_and_restores_mode(self):
        self.assertEqual(sequential_windows(12,4),[(0,4),(4,4),(8,3)])
        class Uniform(torch.nn.Module):
            def __init__(self):
                super().__init__();self.seen=[]
            def forward(self,input_ids,use_cache=False):
                self.seen.extend(input_ids[0].tolist())
                return SimpleNamespace(logits=torch.zeros((*input_ids.shape,32)))
        model=Uniform().train()
        result=evaluate(model,np.arange(12,dtype='<u4'),4,torch.device('cpu'),'sequential')
        self.assertAlmostEqual(result,math.log(32),places=6)
        self.assertEqual(model.seen,list(range(11)))
        self.assertTrue(model.training)

    def test_cosine_reaches_registered_final_learning_rate(self):
        a=SimpleNamespace(lr=0.004,warmup=11,min_lr_ratio=0.1)
        schedule=lr_schedule(a,223)
        self.assertAlmostEqual(schedule(10),0.004)
        self.assertAlmostEqual(schedule(222),0.0004)


class Configs(unittest.TestCase):
    def test_archived_q35_configs_match_backbone_budget(self):
        from q38.flops import _count_backbone
        from q38.presets import model_config, budget_tokens
        for path in (ROOT/'configs/q35_isoflops').glob('*/*/config.json'):
            _, p = load_training_config(path)
            fc = _count_backbone(model_config(p['model'],10112,8192).to_dict(),8192)
            self.assertEqual(fc['train_flops_per_token'],p['accounting']['train_flops_per_token'])
            self.assertEqual(budget_tokens(fc['train_flops_per_token'],p['compute'],p['train']['batch_tokens']),p['accounting']['train_tokens'])

if __name__ == '__main__': unittest.main()
