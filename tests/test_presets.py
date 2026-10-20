import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from tokenizers import Tokenizer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from count_flops_q35 import count
from launch import command, NAMES

class Baselines(unittest.TestCase):
    def test_archived_parameter_and_flop_counts(self):
        for name in NAMES:
            with self.subTest(name=name):
                run = json.loads((ROOT / 'run_configs' / name / 'run_config.json').read_text())
                c = run['config']
                actual = count(c, run['seq'])
                self.assertEqual(actual['train_flops_per_token'], run['flops_per_token'])
                self.assertEqual(actual['total_nonvocab_params'] + actual['embed_params'], run['n_params'])
                self.assertEqual(actual['active_nonvocab_params'], run['n_nonvocab_active'])
                self.assertEqual(c['vocab_size'], 153600)
                self.assertTrue(c['tie_word_embeddings'])
                self.assertEqual(c['num_experts'], 64)
                self.assertEqual(c['num_experts_per_tok'], 4)
                self.assertEqual(c['layer_types'], ['full_attention' if i % 4 == 3 else 'linear_attention' for i in range(run['layers'])])
                self.assertEqual(run['batch_tokens'] % (run['micro_bs'] * run['seq'] * 2), 0)

    def test_launch_uses_local_paths_and_preserves_overrides(self):
        overrides = ['--max_steps', '2', '--run_dir', '/tmp/test run']
        cmd = command('q35_c2_m', overrides, 2)
        self.assertIn('--nproc_per_node=2', cmd)
        self.assertIn(str(ROOT / 'tokenizer/gzy_10k'), cmd)
        self.assertIn(str(ROOT / 'data/corpus_gzy10k/train.bin'), cmd)
        self.assertIn(str(ROOT / 'runs/q35_c2_m_gzy10k'), cmd)
        self.assertEqual(cmd[-len(overrides):], overrides)
        self.assertFalse(any('/local/hzh-agent/' in value for value in cmd))

    def test_small_vocab_budgets_and_tokenizer_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            for name in NAMES:
                for tokenizer in ('gzy_10k', 'exaone_154k'):
                    with self.subTest(name=name, tokenizer=tokenizer):
                        tok_path = ROOT / 'tokenizer' / tokenizer
                        overrides = ['--tokenizer', str(tok_path), '--seq', '4096']
                        cmd = command(name, overrides, 1)
                        # CLI overrides follow the defaults; argparse uses the last value.
                        args = dict(zip(cmd[2::2], cmd[3::2]))
                        saved = json.loads((ROOT / 'run_configs' / name / 'run_config.json').read_text())
                        cfg = dict(saved['config'], vocab_size=Tokenizer.from_file(
                            str(tok_path / 'tokenizer.json')).get_vocab_size())
                        flops = count(cfg, 4096)['train_flops_per_token']
                        budget = 3e17 if name.startswith('q35_c1_') else 1e18
                        tokens = int(args['--tokens'])
                        self.assertLessEqual(tokens * flops, budget)
                        self.assertGreater((tokens + int(args['--batch_tokens'])) * flops, budget)

if __name__ == '__main__':
    unittest.main()
