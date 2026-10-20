import json
from pathlib import Path
import sys
import unittest
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
        self.assertIn(str(ROOT / 'tokenizer/exaone_154k'), cmd)
        self.assertEqual(cmd[-len(overrides):], overrides)
        self.assertFalse(any('/local/hzh-agent/' in value for value in cmd))

if __name__ == '__main__':
    unittest.main()
