"""Recorded current-graph fit, canonical presets and fail-closed generation."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from q38.flops import count
from q38.presets import ROOT, dense_shape, model_config
from scripts.make_baseline_config import FAMILY, make_config, make_training_config
from scripts.make_dense_scaling_configs import (
    DEFAULT_FIT_SOURCE, DEFAULT_OUTPUT, DEFAULT_RUN_ROOT,
    build_plan, load_current_fit, write_plan,
)
from scripts.make_q38_configs import BUDGETS, batch_for, shape_for
from scripts.make_q38_isoflops_configs import serialized
from scripts.fit_q38_isoflops import fit_curves


class DenseScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = DEFAULT_FIT_SOURCE.read_bytes()
        cls.record = json.loads(cls.raw)
        cls.files, cls.selection = build_plan()

    def test_default_raw_fit_retains_its_original_identity(self):
        self.assertEqual(DEFAULT_FIT_SOURCE,
                         ROOT/'docs/experiments/q38_combined_isoflops_results.json')
        digest = hashlib.sha256(self.raw).hexdigest()
        self.assertEqual(digest, '238e4669fbf4d7b9d7b8e4929af3fa9b67d051f867b400b960f31da1f5a05621')
        self.assertEqual(self.record['source_commit'], '5ee96657df7d923c63b6dcddb38a53dd99a69a17')
        fit, source = load_current_fit(DEFAULT_FIT_SOURCE)
        self.assertEqual(fit, self.record)
        self.assertEqual(DEFAULT_FIT_SOURCE.read_bytes(), self.raw)
        self.assertEqual(source['path'], 'docs/experiments/q38_combined_isoflops_results.json')
        self.assertEqual(source['sha256'], digest)
        self.assertEqual(source['training_source_commit'], self.record['source_commit'])
        self.assertEqual(source['manifest_sha256'], self.record['manifest_sha256'])
        self.assertEqual(source['fitting_code_sha256'], self.record['fitting_code_sha256'])
        self.assertEqual(self.selection['current_architecture_fit'], source)
        self.assertIsNone(self.selection['scale_prior'])
        for key in ('Na_log10_coefficients', 'D_log10_coefficients'):
            self.assertEqual(self.selection[key], self.record[key])

    def test_c1_matches_baseline_except_four_known_paths_and_keeps_its_anchor(self):
        canonical = make_config()
        self.assertEqual(canonical, json.loads((ROOT/'configs/baseline.json').read_text()))
        c1 = self.files[DEFAULT_OUTPUT/'c1.json']
        changed = {key for key in canonical['train']
                   if canonical['train'][key] != c1['train'][key]}
        self.assertEqual(changed, {'run_dir', 'data', 'val_data', 'tokenizer'})
        expected = copy.deepcopy(canonical)
        expected.pop('baseline')
        expected['train'].update(run_dir=str(DEFAULT_RUN_ROOT/'c1'),
            data=self.record['data_identity']['train']['path'],
            val_data=self.record['data_identity']['validation']['path'],
            tokenizer=self.record['tokenizer'])
        self.assertEqual(c1, expected)
        self.assertEqual(c1['model']['ngram_rows'], 61440)
        self.assertEqual(c1['accounting']['active_params_excluding_embeddings'], 33776160)
        self.assertEqual(c1['accounting']['train_flops_per_token'], 366904320)
        self.assertEqual(c1['train']['tokens'], 813432832)
        choice = self.selection['configurations']['c1']
        for key, value in self.record['c1_prediction'].items():
            self.assertEqual(choice[key], value, key)
        self.assertEqual(choice['selected_hidden'], 512)
        self.assertEqual(choice['selection_rule'], 'keep validated h512')
        self.assertEqual(choice['unconstrained_grid_optimum']['hidden'], 576)
        self.assertFalse(choice['c1_free_choice_matches_validated'])

    def test_all_tiers_share_verified_data_precision_and_maximal_whole_batch_budget(self):
        base = make_config()['train']
        expected_shapes = {'c1': (512, 10, 768), 'c2': (704, 14, 1056),
                           'c3': (896, 18, 1344), 'c4': (1152, 22, 1728)}
        for tier, compute in BUDGETS.items():
            with self.subTest(tier=tier):
                cfg = self.files[DEFAULT_OUTPUT/f'{tier}.json']
                train, accounting = cfg['train'], cfg['accounting']
                hidden = cfg['model']['hidden']
                self.assertEqual(cfg['compute'], compute)
                self.assertEqual((hidden, cfg['model']['layers'], cfg['model']['dense_ffn']),
                                 expected_shapes[tier])
                self.assertEqual(train['run_dir'], str(DEFAULT_RUN_ROOT/tier))
                self.assertEqual(train['data'], self.record['data_identity']['train']['path'])
                self.assertEqual(train['val_data'], self.record['data_identity']['validation']['path'])
                self.assertEqual(train['tokenizer'], self.record['tokenizer'])
                self.assertEqual(train['batch_tokens'], batch_for(compute))
                self.assertEqual(train['lr'], round(.3118*compute**-.125, 6))
                for key, value in base.items():
                    if key not in {'run_dir', 'tokens', 'warmup', 'batch_tokens', 'lr',
                                   'data', 'val_data', 'tokenizer'}:
                        self.assertEqual(train[key], value, key)
                resolved = model_config(cfg['model'], 10112, train['seq'])
                self.assertFalse(resolved.mhc_enabled)
                self.assertEqual(resolved.qsa_mode, 'full')
                self.assertEqual(resolved.num_experts, 0)
                self.assertTrue(resolved.mtp)
                self.assertEqual(cfg['model']['ngram_rows'],
                    dense_shape(shape_for(hidden), 10112, 4096)['ngram_rows'])
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
                    'combined_c1_validated' if tier == 'c1' else 'untrained_fitted_extrapolation')
                self.assertLessEqual(d, choice['unique_training_tokens'])

    def test_c2_c4_minimize_the_recorded_joint_score_on_the_feasible_grid(self):
        unique = (self.record['data_identity']['train']['bytes']//4-1)//4096*4096
        for tier in ('c2', 'c3', 'c4'):
            with self.subTest(tier=tier):
                compute = BUDGETS[tier]
                choice = self.selection['configurations'][tier]
                ncoef, dcoef = self.record['Na_log10_coefficients'], self.record['D_log10_coefficients']
                target_n = 10**(ncoef[0]*math.log10(compute)+ncoef[1])
                target_d = 10**(dcoef[0]*math.log10(compute)+dcoef[1])
                self.assertAlmostEqual(choice['predicted_Na'], target_n, delta=target_n*1e-12)
                self.assertAlmostEqual(choice['predicted_D'], target_d, delta=target_d*1e-12)
                self.assertEqual(choice['unique_training_tokens'], unique)
                scores = []
                for hidden in range(64, 1601, 64):
                    cfg = make_training_config(hidden, compute, DEFAULT_RUN_ROOT/tier,
                        self.record['data_root'], self.record['tokenizer'])
                    n, d = cfg['accounting']['active_params_excluding_embeddings'], cfg['train']['tokens']
                    score = math.log(n/choice['predicted_Na'])**2+math.log(d/choice['predicted_D'])**2
                    scores.append((score, hidden, n, d))
                feasible = [row for row in scores if row[3] <= unique]
                selected = min(feasible)
                self.assertEqual((choice['selection_score'], choice['selected_hidden'],
                                  choice['selected_Na'], choice['selected_D']), selected)
                free = choice['unconstrained_grid_optimum']
                self.assertEqual((free['selection_score'], free['hidden'], free['Na'], free['D']), min(scores))
                self.assertEqual(free['fits_unique_pool'], free['D'] <= unique)

    def test_invalid_shape_or_budget_is_rejected(self):
        for hidden, compute in [(63, 3*10**17), (1664, 3*10**17), (512., 3*10**17), (512, 0), (512, 3e17)]:
            with self.subTest(hidden=hidden, compute=compute), self.assertRaises(ValueError):
                make_training_config(hidden, compute, '/unused')

    def test_default_five_files_bind_config_hashes_and_are_reproducible(self):
        files, selection = build_plan()
        self.assertEqual(files, self.files)
        self.assertEqual(selection, self.selection)
        for tier in BUDGETS:
            path = DEFAULT_OUTPUT/f'{tier}.json'
            self.assertEqual(path.read_bytes(), serialized(files[path]))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             selection['configurations'][tier]['config_sha256'])
        self.assertEqual((DEFAULT_OUTPUT/'selection.json').read_bytes(), serialized(selection))
        self.assertEqual(write_plan(check=True), selection)
        self.assertEqual(DEFAULT_FIT_SOURCE.read_bytes(), self.raw)

    def test_check_does_not_write_or_create_missing_directories(self):
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

    def test_missing_none_legacy_or_corrupt_fit_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, 'fit source is required'):
            build_plan(fit_source=None)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'fit.json'
            with self.assertRaises(FileNotFoundError):
                build_plan(fit_source=source)
            source.write_text('{not-json')
            with self.assertRaises(json.JSONDecodeError):
                build_plan(fit_source=source)
        with self.assertRaisesRegex(ValueError, 'family'):
            build_plan(fit_source=ROOT/'docs/experiments/q38_dense_results.json')
        original_read = Path.read_bytes

        def missing_default(path):
            if path == DEFAULT_FIT_SOURCE:
                raise FileNotFoundError(path)
            return original_read(path)

        with patch.object(Path, 'read_bytes', missing_default):
            with self.assertRaises(FileNotFoundError):
                build_plan()

    def test_explicit_fit_and_verified_data_paths_match_the_default(self):
        self.assertEqual(build_plan(data_root=self.record['data_root'], tokenizer=self.record['tokenizer']),
                         (self.files, self.selection))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'fit.json'
            source.write_bytes(self.raw)
            files, selection = build_plan(fit_source=source)
            for tier in BUDGETS:
                self.assertEqual(files[DEFAULT_OUTPUT/f'{tier}.json'], self.files[DEFAULT_OUTPUT/f'{tier}.json'])
            self.assertEqual(selection['current_architecture_fit']['sha256'], hashlib.sha256(self.raw).hexdigest())
            with self.assertRaisesRegex(ValueError, 'snapshot canonical path'):
                build_plan(fit_source=source, data_root='/data/q38/different-unhashed-pool')
            with self.assertRaisesRegex(ValueError, 'Tokenizer location'):
                build_plan(fit_source=source, tokenizer='/data/q38/different-tokenizer')

    def test_raw_fit_rechecks_point_acceptance_identity_and_derived_numerics(self):
        changed_cases = []
        changed = copy.deepcopy(self.record)
        for curve in changed['budget_fits']:
            for point in curve['points']:
                point['loss'] = point['Na']
        changed_cases.append(changed)
        changed = copy.deepcopy(self.record)
        changed['budget_fits'][0]['points'][0]['loss'] = float('nan')
        changed_cases.append(changed)
        changed = copy.deepcopy(self.record)
        changed['budget_fits'][0]['points'].pop()
        changed_cases.append(changed)
        for field in ('Na_log10_coefficients', 'D_log10_coefficients'):
            changed = copy.deepcopy(self.record)
            changed[field][0] += .01
            changed_cases.append(changed)
        changed = copy.deepcopy(self.record)
        changed['budget_fits'][0]['Na_opt'] *= 1.01
        changed_cases.append(changed)
        changed = copy.deepcopy(self.record)
        changed['leave_one_budget_out'][0]['c1_D_prediction'] *= 1.01
        changed_cases.append(changed)
        changed = copy.deepcopy(self.record)
        changed['validation']['offset'] += 1
        changed_cases.append(changed)
        changed = copy.deepcopy(self.record)
        changed['data_identity']['train']['sha256'] = '0'*64
        changed_cases.append(changed)
        # Recomputing all derived curves cannot legitimize wrong Na/D/FD inputs.
        for field in ('Na', 'D', 'compute'):
            changed = copy.deepcopy(self.record)
            points = [p for curve in changed['budget_fits'] for p in curve['points']]
            for point in points:
                point[field] *= 2
            changed.update(fit_curves(points))
            changed_cases.append(changed)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'fit.json'
            for i, changed in enumerate(changed_cases):
                source.write_bytes(serialized(changed))
                with self.subTest(case=i), self.assertRaises(ValueError):
                    build_plan(fit_source=source)

    def test_float64_tolerance_does_not_replace_original_fit_values_or_bytes(self):
        changed = copy.deepcopy(self.record)
        changed['Na_log10_coefficients'][0] += 1e-12
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'fit.json'
            raw = serialized(changed)
            source.write_bytes(raw)
            loaded, identity = load_current_fit(source)
            self.assertEqual(loaded['Na_log10_coefficients'], changed['Na_log10_coefficients'])
            self.assertEqual(identity['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(source.read_bytes(), raw)

    def test_generation_only_writes_json_without_launching_or_creating_c1_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, run_root = Path(tmp)/'configs', Path(tmp)/'runs'
            raw_before = (DEFAULT_FIT_SOURCE.read_bytes(), DEFAULT_FIT_SOURCE.stat().st_mtime_ns)
            with patch('subprocess.Popen', side_effect=AssertionError('generation launched a process')):
                selection = write_plan(output, run_root)
            self.assertFalse(run_root.exists())
            self.assertEqual(sorted(p.name for p in output.iterdir()),
                             ['c1.json', 'c2.json', 'c3.json', 'c4.json', 'selection.json'])
            self.assertEqual((DEFAULT_FIT_SOURCE.read_bytes(), DEFAULT_FIT_SOURCE.stat().st_mtime_ns), raw_before)
            self.assertEqual(selection['configurations']['c1']['selected_hidden'], 512)


if __name__ == '__main__':
    unittest.main()
