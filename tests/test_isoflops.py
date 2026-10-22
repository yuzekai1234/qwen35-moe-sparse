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

from q35.presets import ROOT, load_training_config
from q35.sampling import ShuffledBlockSampler, sequential_windows
from q35.train import evaluate, lr_schedule, main, micro_bs_candidates, parse_args, prepare_inputs


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
    @classmethod
    def setUpClass(cls):
        cls.base=ROOT/'configs/isoflops'

    def test_every_configuration_matches_budget_and_directory(self):
        configs=sorted(self.base.glob('*/*/*.json'))
        self.assertEqual(len(configs),28)
        for path in configs:
            with self.subTest(path=str(path)):
                config_path,p=load_training_config(path.parent if path.name=='config.json' else path)
                a=parse_args(['--config',str(config_path)])
                _,_,fc,_,dev,final,sampler,plan=prepare_inputs(a)
                self.assertEqual(plan['steps'],p['accounting']['steps'])
                self.assertEqual(plan['train_tokens'],p['accounting']['train_tokens'])
                self.assertEqual(plan['train_flops_per_token'],p['accounting']['train_flops_per_token'])
                self.assertLessEqual(plan['actual_train_flops'],p['compute'])
                self.assertGreater(plan['actual_train_flops']+fc['train_flops_per_token']*a.batch_tokens,p['compute'])
                self.assertEqual(Path(plan['run_dir']),ROOT/p['train']['run_dir'])
                self.assertEqual((plan['micro_bs'],plan['accumulation']),(8,1))
                self.assertFalse(plan['sampling']['replacement'])
                self.assertGreaterEqual(sampler.blocks*a.seq,plan['train_tokens'])
                self.assertGreaterEqual(len(dev)-1,2_000_000)
                self.assertGreaterEqual(len(final)-1,16_000_000)
                self.assertEqual(a.final_eval_offset,len(dev))

    def test_cli_override_and_dry_run_have_no_training_side_effects(self):
        config=self.base/'3e15/M1'
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'must-not-be-created'
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main(['--config',str(config),'--dry-run','--run_dir',str(output),'--lr','0.002'])
            plan=json.loads(stream.getvalue())
            self.assertEqual(plan['lr'],0.002)
            self.assertFalse(output.exists())
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parse_args(['--config',str(config),'--preset','q35_c1_s1'])

    def test_micro_batch_follows_world_size(self):
        config=self.base/'3e15/M1'
        for world,expected in [(1,[8,4,2,1]),(2,[4,2,1]),(4,[2,1]),(8,[1])]:
            plan=prepare_inputs(parse_args(['--config',str(config)]),world)[-1]
            self.assertEqual((plan['micro_bs'],plan['accumulation']),(expected[0],1))
            plan=prepare_inputs(parse_args(['--config',str(config),'--auto_micro_bs','1']),world)[-1]
            self.assertEqual((plan['micro_bs'],plan['micro_bs_candidates']),('auto',expected))
        plan=prepare_inputs(parse_args(['--config',str(config),'--micro_bs','2']),2)[-1]
        self.assertEqual((plan['micro_bs'],plan['accumulation']),(2,2))
        plan=prepare_inputs(parse_args(['--config',str(config),'--micro_bs','3','--auto_micro_bs','1']),1)[-1]
        self.assertEqual(plan['micro_bs_candidates'],[2,1])
        with self.assertRaises(ValueError):
            prepare_inputs(parse_args(['--config',str(config),'--micro_bs','3']),1)
        with self.assertRaises(ValueError):
            prepare_inputs(parse_args(['--config',str(config)]),3)
        self.assertEqual(micro_bs_candidates(12,12),[12,6,4,3,2,1])

    def test_validation_partitions_have_disjoint_document_ranges(self):
        val=json.loads((self.base/'validation.json').read_text())
        self.assertEqual(val['dev']['offset'],0)
        self.assertEqual(val['dev']['end'],val['curve']['offset'])
        self.assertEqual(val['curve']['end'],val['report']['offset'])
        self.assertEqual(sum(val[k]['documents'] for k in ('dev','curve','report')),val['document_count'])


if __name__=='__main__':
    unittest.main()
