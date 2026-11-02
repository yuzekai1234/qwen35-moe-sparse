"""Complete current-graph sweep, strict provenance, and constrained N/D fitting."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from q38.flops import count
from q38.presets import dense_shape, model_config
from scripts.fit_q38_isoflops import fit_curves, training_recipe, load_rows
from scripts import fit_q38_isoflops as fitter
from scripts.make_dense_scaling_configs import load_current_fit
from scripts.make_baseline_config import make_training_config
from scripts.make_q38_configs import shape_for
from scripts.make_q38_isoflops_configs import build_plan, write_plan, BUDGETS
from scripts.q38_isoflops_contract import FAMILY, GRID
from tests.q38_isoflops_fixtures import data_identity, runtime_data_identity, synthetic_fit


class Plans(unittest.TestCase):
    def test_seventeen_points_and_no_c1(self):
        configs, manifest = build_plan('/unused/configs', '/unused/runs', 'source',
            data_identity=data_identity(), data_root='/data/q38/fixture', tokenizer='/data/q38/fixture/tokenizer')
        self.assertEqual(len(configs), 17)
        self.assertEqual(manifest['family'], FAMILY)
        self.assertEqual([sum(j['compute'] == c for j in manifest['jobs']) for c in BUDGETS.values()], [4,4,4,5])
        self.assertNotIn(3*10**17, [j['compute'] for j in manifest['jobs']])
        self.assertEqual(manifest['validation']['offset'], 2002536)
        self.assertEqual(manifest['validation']['targets'], 16001875)
        for label, widths in GRID.items():
            self.assertEqual([j['hidden'] for j in manifest['jobs'] if j['budget'] == label], list(widths))
        for job in manifest['jobs']:
            cfg = configs[Path(job['config'])]
            self.assertEqual(cfg['experiment']['family'], FAMILY)
            self.assertEqual(cfg['model']['mhc_enabled'], 0)
            self.assertEqual(cfg['model']['qsa_mode'], 'full')
            self.assertEqual(cfg['model']['experts'], 0)
            self.assertEqual(cfg['model']['ngram_rows'], dense_shape(shape_for(job['hidden']),10112,4096)['ngram_rows'])
            self.assertEqual(cfg['train']['data'], '/data/q38/fixture/train.bin')
            self.assertEqual(cfg['train']['tokenizer'], '/data/q38/fixture/tokenizer')
            self.assertEqual(cfg['train']['max_steps'], 0)
            self.assertEqual((cfg['train']['ffn_precision'],cfg['train']['projection_precision']), ('bf16','bf16'))
            fc = count(model_config(cfg['model'],10112,4096).to_dict(),4096)
            self.assertEqual(job['Na'], fc['active_params_excluding_embeddings'])
            self.assertEqual(job['expected_tokens'], cfg['train']['tokens'])
            self.assertEqual(job['expected_steps']*cfg['train']['batch_tokens'], job['expected_tokens'])
            self.assertLessEqual(job['expected_tokens']*job['train_flops_per_token'], job['compute']*995//1000)

    def test_explicit_snapshot_is_required_and_invalid_hash_is_not_defaulted(self):
        with self.assertRaises(TypeError):
            build_plan('/unused/configs','/unused/runs')
        snapshot = data_identity()
        del snapshot['validation']['sha256']
        with self.assertRaisesRegex(ValueError,'validation'):
            build_plan('/unused/configs','/unused/runs',data_identity=snapshot)

    def test_unhashed_dataset_location_is_rejected_before_any_files_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'cfg'
            with self.assertRaisesRegex(ValueError,'snapshot canonical path'):
                write_plan(out,Path(tmp)/'runs','source',data_identity=data_identity(),
                    data_root='/data/q38/different-unhashed-pool',tokenizer='/data/q38/fixture/tokenizer')
            self.assertFalse(out.exists())

    def test_existing_different_configuration_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,runs = Path(tmp)/'cfg',Path(tmp)/'runs'
            paths = dict(data_root='/data/q38/fixture',tokenizer='/data/q38/fixture/tokenizer')
            manifest = write_plan(out,runs,'source',data_identity=data_identity(),**paths)
            self.assertEqual(write_plan(out,runs,'source',check=True,data_identity=data_identity(),**paths),manifest)
            first = Path(manifest['jobs'][0]['config'])
            first.write_text('{}')
            with self.assertRaises(FileExistsError):
                write_plan(out,runs,'source',data_identity=data_identity(),**paths)
            self.assertEqual(first.read_text(),'{}')


class Fitting(unittest.TestCase):
    def test_convex_curves_keep_c1_and_report_the_free_grid_choice(self):
        fit,_ = synthetic_fit()
        self.assertEqual(len(fit['budget_fits']),4)
        self.assertEqual(len(fit['leave_one_budget_out']),4)
        for curve in fit['budget_fits']:
            self.assertLess(curve['residual_rmse'],1e-10)
            self.assertAlmostEqual(curve['fitted_minimum_ce'],2.)
        cfg,pred = training_recipe(fit,'c1',Path('/unused/c1'))
        self.assertEqual(cfg['model']['hidden'],512)
        self.assertEqual(cfg['train']['tokens'],813432832)
        self.assertEqual(pred['c1_free_choice_matches_validated'], pred['unconstrained_grid_optimum']['hidden']==512)
        self.assertIsNone(pred['quality_prediction'])
        self.assertEqual(cfg['train']['data'],fit['data_identity']['train']['path'])
        with self.assertRaisesRegex(ValueError,'snapshot canonical path'):
            training_recipe(fit,'c1','/unused/c1',data_root='/data/q38/different-unhashed-pool')

    def test_boundary_missing_duplicate_legacy_or_c1_input_is_rejected(self):
        _,rows = synthetic_fit()
        changed = copy.deepcopy(rows)
        for row in changed:
            row['loss'] = row['Na']
        with self.assertRaisesRegex(ValueError,'boundary'):
            fit_curves(changed)
        for changed in [rows[:-1], rows+[dict(rows[-1])], rows+[dict(rows[-1],target_compute=3*10**17)]]:
            with self.assertRaisesRegex(ValueError,'17-point'):
                fit_curves(changed)
        changed = copy.deepcopy(rows)
        changed[0]['family'] = 'dense'
        with self.assertRaisesRegex(ValueError,'legacy'):
            fit_curves(changed)

    def test_unique_pool_constraint_is_enforced_and_unconstrained_choice_is_reported(self):
        fit,_ = synthetic_fit()
        fit['Na_log10_coefficients'] = [0.,3.]
        fit['D_log10_coefficients'] = [0.,12.]
        cfg,pred = training_recipe(fit,'c4','/unused/c4')
        self.assertFalse(pred['unconstrained_grid_optimum']['fits_unique_pool'])
        self.assertLessEqual(cfg['train']['tokens'],pred['unique_training_tokens'])
        c1,pred1 = training_recipe(fit,'c1','/unused/c1')
        self.assertEqual(c1['model']['hidden'],512)
        self.assertFalse(pred1['c1_free_choice_matches_validated'])
        fit['family'] = 'dense'
        with self.assertRaisesRegex(ValueError,'current-architecture'):
            training_recipe(fit,'c2','/unused/c2')

    def test_loader_binds_whole_file_attestation_without_meta_declared_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_plan(root/'cfg',root/'runs','source',data_identity=data_identity(),
                data_root='/data/q38/fixture',tokenizer='/data/q38/fixture/tokenizer')
            _,synthetic_rows = synthetic_fit()
            losses = {row['id']:row['loss'] for row in synthetic_rows}
            for job in manifest['jobs']:
                cfg = json.loads(Path(job['config']).read_text())
                run = Path(job['run_dir']); run.mkdir(parents=True)
                total = job['expected_tokens']*job['train_flops_per_token']
                summary = dict(completed_budget=True,tokens_trained=job['expected_tokens'],train_flops=total)
                final = dict(summary,loss=losses[job['id']],metric='pure_cross_entropy',offset=manifest['validation']['offset'],
                    scored_tokens=manifest['validation']['targets'],tokenizer_sha256=manifest['validation']['tokenizer_sha256'])
                rc = dict(cfg['train'],**cfg['model'],compute=job['compute'],steps=job['expected_steps'],
                    config_sha256=job['expected_config_sha256'],model_code=dict(files=manifest['model_files_sha256']),
                    tokenizer_sha256=manifest['validation']['tokenizer_sha256'],
                    data_identity=runtime_data_identity(manifest['data_identity']))
                for name,value in [('summary',summary),('final_eval',final),('run_config',rc),('flops',cfg['accounting'])]:
                    (run/(name+'.json')).write_text(json.dumps(value))
            path = root/'cfg/manifest.json'
            _,rows = load_rows(path)
            self.assertEqual(len(rows),17)
            c1 = root/'already-validated-c1'; c1.mkdir()
            prediction = root/'prediction'
            argv = ['fit','--manifest',str(path),'--output',str(prediction),'--c1-run-dir',str(c1)]
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                fitter.main()
            raw,_ = load_current_fit(prediction/'fit.json')
            self.assertEqual(raw['family'],FAMILY)
            self.assertEqual(json.loads((prediction/'c1-config.json').read_text())['model']['hidden'],512)
            self.assertEqual(list(c1.iterdir()),[])
            record_path = Path(manifest['jobs'][0]['run_dir'])/'run_config.json'
            original = json.loads(record_path.read_text())
            for stream,key,value in [('train','mtime_ns',1),('train','path','/wrong/train.bin'),
                                     ('validation','bytes',4),('validation','mtime_ns',1)]:
                changed = copy.deepcopy(original)
                changed['data_identity'][stream][key] = value
                record_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    load_rows(path)
            changed = copy.deepcopy(original)
            del changed['data_identity']['validation']
            record_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                load_rows(path)
            record_path.write_text(json.dumps(original))
            missing = dict(manifest,jobs=manifest['jobs'][:-1])
            path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError,'17-point'):
                load_rows(path)


if __name__ == '__main__':
    unittest.main()
