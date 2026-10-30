"""Combined-model presets share one graph and do not claim a completed N/D fit."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from q38.flops import count
from q38.presets import ROOT, dense_shape, model_config
from scripts.make_baseline_config import FAMILY, make_config, make_training_config
from scripts.make_dense_scaling_configs import (
    DEFAULT_OUTPUT, DEFAULT_RUN_ROOT, PROVISIONAL_HIDDEN, SOURCE,
    build_plan, load_scale_prior, write_plan,
)
from scripts.make_q38_configs import BUDGETS, batch_for, shape_for
from scripts.make_q38_isoflops_configs import serialized


class DenseScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(SOURCE.read_bytes())
        cls.files, cls.selection = build_plan()

    def test_legacy_fit_is_authenticated_and_only_a_scale_prior(self):
        digest = hashlib.sha256(serialized(self.record['isoflops'])).hexdigest()
        prior = load_scale_prior()
        self.assertEqual(digest, self.record['prediction_files_sha256']['fit.json'])
        self.assertEqual(prior['fit_sha256'], digest)
        self.assertEqual(prior['sha256'], hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self.assertEqual(self.selection['family'], FAMILY)
        self.assertIsNone(self.selection['current_architecture_fit'])
        self.assertNotIn('Na_log10_coefficients', self.selection)
        self.assertNotIn('D_log10_coefficients', self.selection)
        with patch('scripts.fit_q38_isoflops.fit_curves', side_effect=AssertionError('No refit')):
            self.assertEqual(build_plan(), (self.files, self.selection))

    def test_changed_legacy_fit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'results.json'
            changed = copy.deepcopy(self.record)
            changed['isoflops']['Na_log10_coefficients'][0] += .01
            path.write_bytes(serialized(changed))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_scale_prior(path)

    def test_c1_matches_the_validated_combined_graph_and_canonical_recipe(self):
        canonical = make_config()
        self.assertEqual(canonical, json.loads((ROOT/'configs/baseline.json').read_text()))
        canonical.pop('baseline')
        canonical['train']['run_dir'] = str(DEFAULT_RUN_ROOT/'c1')
        self.assertEqual(self.files[DEFAULT_OUTPUT/'c1.json'], canonical)
        self.assertEqual(canonical['model']['ngram_rows'], 61440)
        self.assertEqual(canonical['accounting']['active_params_excluding_embeddings'], 33776160)
        self.assertEqual(canonical['accounting']['train_flops_per_token'], 366904320)
        self.assertEqual(canonical['train']['tokens'], 813432832)

    def test_all_tiers_share_graph_data_precision_and_maximal_whole_batch_budget(self):
        base = make_config()['train']
        for tier, compute in BUDGETS.items():
            with self.subTest(tier=tier):
                cfg = self.files[DEFAULT_OUTPUT/f'{tier}.json']
                train, accounting = cfg['train'], cfg['accounting']
                self.assertEqual(cfg['compute'], compute)
                self.assertEqual(cfg['model']['hidden'], PROVISIONAL_HIDDEN[tier])
                self.assertEqual(train['batch_tokens'], batch_for(compute))
                self.assertEqual(train['lr'], round(.3118*compute**-.125, 6))
                for key, value in base.items():
                    if key not in {'run_dir', 'tokens', 'warmup', 'batch_tokens', 'lr'}:
                        self.assertEqual(train[key], value, key)
                resolved = model_config(cfg['model'], 10112, train['seq'])
                self.assertFalse(resolved.mhc_enabled)
                self.assertEqual(resolved.qsa_mode, 'full')
                self.assertEqual(resolved.num_experts, 0)
                self.assertTrue(resolved.mtp)
                self.assertEqual(cfg['model']['ngram_rows'],
                    dense_shape(shape_for(PROVISIONAL_HIDDEN[tier]), 10112, 4096)['ngram_rows'])
                self.assertIn('ling-no-mtp-c1-global-pruned10k-scale-isoflops-v1', train['data'])
                self.assertEqual((train['ffn_precision'], train['projection_precision']), ('bf16', 'bf16'))
                self.assertEqual((train['mtp_coef'], train['kl_coef'], train['dense_warmup']), (.3, 0., 0))
                fc = count(resolved.to_dict(), train['seq'])
                for key, value in fc.items():
                    self.assertEqual(accounting[key], value, key)
                f, d, batch = fc['train_flops_per_token'], train['tokens'], train['batch_tokens']
                budget = compute*995//1000
                self.assertEqual(d % batch, 0)
                self.assertEqual(accounting['steps']*batch, d)
                self.assertEqual(accounting['actual_train_flops'], f*d)
                self.assertLessEqual(f*d, budget)
                self.assertGreater(f*(d+batch), budget)
                self.assertEqual(train['warmup'], max(10, round(accounting['steps']*.01)))
                choice = self.selection['configurations'][tier]
                self.assertIsNone(choice['quality_prediction'])
                self.assertEqual(choice['validation_status'],
                    'combined_c1_validated' if tier == 'c1' else 'untrained_provisional')

    def test_invalid_shape_or_budget_is_rejected(self):
        for hidden, compute in [(63, 3*10**17), (1664, 3*10**17), (512, 0), (512, 3e17)]:
            with self.subTest(hidden=hidden, compute=compute), self.assertRaises(ValueError):
                make_training_config(hidden, compute, '/unused')

    def test_generated_files_bind_config_hashes_and_are_reproducible(self):
        for tier in BUDGETS:
            path = DEFAULT_OUTPUT/f'{tier}.json'
            self.assertEqual(path.read_bytes(), serialized(self.files[path]))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             self.selection['configurations'][tier]['config_sha256'])
        self.assertEqual((DEFAULT_OUTPUT/'selection.json').read_bytes(), serialized(self.selection))
        self.assertEqual(write_plan(check=True), self.selection)

    def test_check_does_not_write_and_missing_check_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'configs'
            with self.assertRaises(FileNotFoundError):
                write_plan(output, check=True)
            self.assertFalse(output.exists())
            expected = write_plan(output)
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            with patch.object(Path, 'write_bytes', side_effect=AssertionError('check wrote a file')):
                self.assertEqual(write_plan(output, check=True), expected)
            self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}, before)

    def test_all_destinations_are_checked_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'configs'
            output.mkdir()
            (output/'c4.json').write_text('{}\n')
            with self.assertRaises(FileExistsError):
                write_plan(output)
            self.assertEqual((output/'c4.json').read_text(), '{}\n')
            self.assertEqual(list(output.iterdir()), [output/'c4.json'])


if __name__ == '__main__':
    unittest.main()
