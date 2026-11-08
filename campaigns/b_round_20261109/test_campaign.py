"""Regression checks for budget integrity, epoch boundaries, claims and dynamic scoring."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from common import HERE, Lease, claim, record, show_status
from prepare import build, counter, hf_config
from score import fit_performance, quantile, rerank

spec = importlib.util.spec_from_file_location('isolated_sampling', HERE/'runtime/q38/sampling.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


class CampaignTests(unittest.TestCase):
    def test_first_epoch_matches_frozen_sampler(self):
        s = sampling.ShuffledBlockSampler(101, 4, 7, epochs=3)
        expected = np.random.default_rng(7).permutation(25)*4
        np.testing.assert_array_equal(s.offsets(0, 25), expected)
        self.assertEqual(sampling.ShuffledBlockSampler(101, 4, 7).metadata()['version'], sampling.SAMPLER_VERSION)

    def test_epoch_boundary_and_resume_are_deterministic(self):
        s = sampling.ShuffledBlockSampler(101, 4, 7, epochs=2)
        expected = np.concatenate([np.random.default_rng(7).permutation(25)[23:],
                                   np.random.default_rng(8).permutation(25)[:3]])*4
        np.testing.assert_array_equal(s.offsets(23, 5), expected)
        np.testing.assert_array_equal(s.offsets(23, 5), sampling.ShuffledBlockSampler(101, 4, 7, 2).offsets(23, 5))
        self.assertEqual(len(set(s.offsets(25, 25))), 25)
        with self.assertRaises(ValueError):
            s.offsets(49, 2)

    def test_rank_partition_does_not_change_global_stream(self):
        s = sampling.ShuffledBlockSampler(1001, 4, 0, 2)
        world, batch, micro = 2, 32, 2
        pieces = [s.micro_offsets(30, m, r, world, micro, batch)
                  for m in range(2) for r in range(world)]
        np.testing.assert_array_equal(np.concatenate(pieces), s.offsets(240, 8))

    def test_budgets_and_shapes_reconcile(self):
        for level in ('c1','c2','c3','c4'):
            c, report = build(level, Path('/nonexistent-test-workspace'), 'test')
            actual = counter()(hf_config(c['model'], c['train']['seq']), c['train']['seq'])
            self.assertEqual(actual['train_flops_per_token'], c['accounting']['train_flops_per_token'])
            self.assertLess(c['accounting']['actual_train_flops'], c['compute'])
            self.assertEqual(c['train']['tokens'] % c['train']['batch_tokens'], 0)
            self.assertGreaterEqual(report['activation_ratio'], .1)
        c1, _ = build('c1', Path('/nonexistent-test-workspace'), 'test')
        self.assertEqual(c1['accounting']['actual_train_flops'], 298484080142450688)
        self.assertEqual(c1['accounting']['total_params_all'], 3369671121)
        c4, _ = build('c4', Path('/nonexistent-test-workspace'), 'test')
        self.assertEqual(c4['train']['data_epochs'], 2)

    def test_variants_recompute_budget(self):
        c, _ = build('c1', Path('/nonexistent-test-workspace'), 'variant', taps=4096)
        baseline, _ = build('c1', Path('/nonexistent-test-workspace'), 'base')
        self.assertNotEqual(c['train']['tokens'], baseline['train']['tokens'])
        self.assertNotEqual(c['accounting']['train_flops_per_token'], baseline['accounting']['train_flops_per_token'])

    def test_no_duplicate_or_foreign_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            claim(temp, 'c4', 'alice')
            with self.assertRaises(RuntimeError):
                claim(temp, 'c4', 'bob', resume=True)
            with self.assertRaises(RuntimeError):
                record(temp, 'c4', 'bob', 'complete', {})
            record(temp, 'c4', 'alice', 'failed', {'error':'fixture'})
            claim(temp, 'c4', 'alice', resume=True)
            record(temp, 'c4', 'alice', 'complete', {})
            with self.assertRaises(RuntimeError):
                claim(temp, 'c4', 'alice', resume=True)

    def test_gpu_lock_and_physical_busy_check(self):
        inv = {0:dict(uuid='GPU-test', total_mib=90000, used_mib=0, utilization=0)}
        with tempfile.TemporaryDirectory() as temp, patch('common.inventory', return_value=inv), \
             patch('common.subprocess.check_output', return_value=''):
            one = Lease(temp,[0],'alice','one').acquire()
            try:
                with self.assertRaises(BlockingIOError):
                    Lease(temp,[0],'bob','two').acquire()
            finally:
                one.release()
            with patch('common.subprocess.check_output', return_value='GPU-test, 123\n'):
                with self.assertRaises(RuntimeError):
                    Lease(temp,[0],'bob','three').acquire()
            inv[0]['used_mib'] = 2000
            with self.assertRaises(RuntimeError):
                Lease(temp,[0],'alice','four').acquire()

    def test_dynamic_weight_and_rank_are_recomputed(self):
        board = [dict(team=t,performance=p,evaltime=e) for t,p,e in
                 [('SQ2026AAA012562',1.,100.),('a',.9,1.),('b',1.1,2.),('c',1.2,3.),('d',1.3,4.)]]
        a = rerank(board, 1., 10.)
        b = rerank(board, 1., 1.)
        self.assertNotEqual(a['lambda_'], b['lambda_'])
        self.assertEqual(b['target']['score'], 1.)
        self.assertEqual(board[0]['evaltime'], 100.)
        self.assertEqual(quantile([1,2,3,4,5], .25), 2)

    def test_power_law_integral(self):
        points = [dict(train_flops=c, result=dict(bpb=.7+.5*(c/3e17)**(-.2)))
                  for c in (3e17,1e18,3e18,1e19)]
        fit = fit_performance(points)
        expected = .7+.5*(1-(1e19/3e17)**(-.2))/(.2*math.log(1e19/3e17))
        self.assertAlmostEqual(fit['performance'], expected, places=7)
        self.assertAlmostEqual(fit['log_space_r2'], 1., places=10)


if __name__ == '__main__':
    unittest.main()
