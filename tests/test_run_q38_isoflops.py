"""Controller provenance and child-process safety; all GPU calls are mocked."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import run_q38_isoflops as runner_module
from scripts.q38_isoflops_contract import BUDGETS,FAMILY,GRID,VALIDATION_OFFSET,VALIDATION_TARGETS
from tests.q38_isoflops_fixtures import data_identity,runtime_data_identity


def make_manifest(root):
    snapshot = data_identity()
    jobs=[]
    for label,widths in GRID.items():
        compute=BUDGETS[label]
        for hidden in widths:
            i=len(jobs)+1;run=root/f'run{i}';path=root/f'config{i}.json'
            f=compute//10000;batch=4096;d=compute*995//1000//(f*batch)*batch
            config=dict(compute=compute,model=dict(hidden=hidden,experts=0,topk=0,mhc_enabled=0,qsa_mode='full',mtp=1),
                train=dict(run_dir=str(run),tokens=d,batch_tokens=batch,seq=4096,final_eval_offset=VALIDATION_OFFSET,
                    data=snapshot['train']['path'],val_data=snapshot['validation']['path'],
                    tokenizer='/data/q38/fixture/tokenizer',
                    final_eval_tokens=VALIDATION_TARGETS,ffn_precision='bf16',projection_precision='bf16',
                    ffn_calibration='',precision_transition=0,mtp_coef=.3,kl_coef=0.,dense_warmup=0),
                accounting=dict(train_flops_per_token=f,active_params_excluding_embeddings=11,accounting_method='fixture'))
            path.write_text(json.dumps(config))
            jobs.append(dict(id=f'job{i}',budget=label,hidden=hidden,config=str(path),run_dir=str(run),compute=compute,
                expected_tokens=d,expected_steps=d//batch,train_flops_per_token=f,Na=11,
                final_eval_offset=VALIDATION_OFFSET,final_eval_targets=VALIDATION_TARGETS,
                tokenizer_sha256=snapshot['tokenizer_sha256']))
    manifest=root/'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1,family=FAMILY,fit_budgets=BUDGETS,jobs=jobs,source_commit='fixture',
        model_files_sha256={'fixture':'code'},data_identity=snapshot,
        validation=dict(offset=VALIDATION_OFFSET,targets=VALIDATION_TARGETS,
            tokenizer_sha256=snapshot['tokenizer_sha256'],validation_sha256=snapshot['validation']['sha256']))))
    return manifest,Path(jobs[0]['config']),Path(jobs[0]['run_dir'])


class RunnerTests(unittest.TestCase):
    def test_controller_import_does_not_import_torch(self):
        proc=subprocess.run([os.sys.executable,'-c',
            "import scripts.run_q38_isoflops,sys; assert 'torch' not in sys.modules"],
            cwd=runner_module.ROOT,check=True,capture_output=True,text=True,timeout=30)
        self.assertEqual(proc.returncode,0)

    def test_dry_run_preserves_python_symlink_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_,_=make_manifest(root)
            venv=root/'venv';venv.mkdir();python=venv/'python';python.symlink_to(os.path.realpath(os.sys.executable))
            output=root/'control'
            argv=['runner','--manifest',str(manifest),'--gpus','0','--python',str(python),'--output',str(output)]
            with mock.patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()) as stdout, \
                    mock.patch.object(runner_module,'check_snapshot_files') as check_files:
                self.assertEqual(runner_module.main(),0)
            check_files.assert_not_called()
            self.assertIn(str(python),stdout.getvalue());self.assertFalse(output.exists())

    def test_manifest_requires_all_17_points_and_current_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,_=make_manifest(root);original=json.loads(path.read_text())
            for update in [dict(jobs=original['jobs'][:-1]),dict(jobs=original['jobs']+[original['jobs'][-1]]),dict(family='dense')]:
                path.write_text(json.dumps(dict(original,**update)))
                with self.assertRaises(ValueError):runner_module.load_manifest(path)

    def test_manifest_rejects_unhashed_config_paths_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config,run=make_manifest(root)
            data=json.loads(config.read_text());data['train']['data']='/data/q38/different-unhashed-pool/train.bin'
            config.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'snapshot canonical path'):
                runner_module.load_manifest(path)
            self.assertFalse(run.exists())

    def test_config_changed_after_plan_is_rejected_without_starting_a_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config,run=make_manifest(root);job=runner_module.load_manifest(path)[0]
            data=json.loads(config.read_text());data['train']['data']='/data/q38/different-unhashed-pool/train.bin'
            config.write_text(json.dumps(data));output=root/'control';output.mkdir()
            runner=runner_module.Runner([job],[0],Path(os.sys.executable),output,1,2,1,1)
            with mock.patch.object(runner_module,'gpu_idle',return_value=(True,{'memory_used_mib':0})), \
                    mock.patch.object(runner_module.subprocess,'Popen') as popen:
                self.assertFalse(runner.run())
            popen.assert_not_called()
            self.assertFalse(run.exists());self.assertEqual(runner.states['job1']['status'],'failed')

    def test_snapshot_stat_check_checks_only_the_two_known_files(self):
        snapshot=data_identity();seen=[]
        def stat(path):
            seen.append(str(path))
            item=next(value for value in (snapshot['train'],snapshot['validation']) if value['path']==str(path))
            return SimpleNamespace(st_size=item['bytes'],st_mtime_ns=item['mtime_ns'])
        with mock.patch.object(Path,'stat',stat):
            runner_module.check_snapshot_files(snapshot)
        self.assertEqual(seen,[snapshot['train']['path'],snapshot['validation']['path']])
        with mock.patch.object(Path,'stat',return_value=SimpleNamespace(st_size=4,st_mtime_ns=1)), \
                self.assertRaisesRegex(ValueError,'snapshot no longer matches'):
            runner_module.check_snapshot_files(snapshot)
        with mock.patch.object(Path,'stat',side_effect=FileNotFoundError('fixture missing')), \
                self.assertRaises(FileNotFoundError):
            runner_module.check_snapshot_files(snapshot)

    def test_execute_rejects_changed_snapshot_before_creating_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,_=make_manifest(root);output=root/'control'
            argv=['runner','--manifest',str(path),'--gpus','0','--python',os.sys.executable,'--output',str(output),'--execute']
            with mock.patch('sys.argv',argv),mock.patch.object(runner_module,'git_source',return_value=('fixture',[])), \
                    mock.patch.object(runner_module,'check_snapshot_files',side_effect=ValueError('snapshot changed')), \
                    mock.patch.object(runner_module.subprocess,'Popen') as popen, \
                    contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'snapshot changed'):
                runner_module.main()
            self.assertFalse(output.exists());popen.assert_not_called()

    def test_changed_snapshot_before_popen_cannot_start_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,_=make_manifest(root);job=runner_module.load_manifest(path)[0]
            output=root/'control';output.mkdir();runner=runner_module.Runner([job],[0],Path(os.sys.executable),output,1,2,1,1)
            with mock.patch.object(runner_module,'gpu_idle',return_value=(True,{'memory_used_mib':0})), \
                    mock.patch.object(runner_module,'git_source',return_value=('fixture',[])), \
                    mock.patch.object(runner_module,'check_snapshot_files',side_effect=ValueError('snapshot changed')), \
                    mock.patch.object(runner_module.subprocess,'Popen') as popen:
                self.assertFalse(runner.run())
            popen.assert_not_called();self.assertEqual(runner.states['job1']['status'],'failed')

    def test_manifest_rejects_mismatched_expected_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,_=make_manifest(root);data=json.loads(path.read_text())
            data['jobs'][0]['expected_steps']+=1;path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'不一致'):runner_module.load_manifest(path)

    def test_manifest_inherits_source_pin_and_main_rejects_wrong_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,_=make_manifest(root);data=json.loads(path.read_text());data['source_commit']='expected-commit'
            path.write_text(json.dumps(data));job=runner_module.load_manifest(path)[0]
            self.assertEqual(job['source_commit'],'expected-commit');output=root/'control'
            argv=['runner','--manifest',str(path),'--gpus','0','--python',os.sys.executable,'--output',str(output),'--execute']
            with mock.patch('sys.argv',argv),mock.patch.object(runner_module,'git_source',return_value=('wrong',[])), \
                    contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'源码版本不符合 manifest'):
                runner_module.main()
            self.assertFalse(output.exists())

    def test_success_guard_requires_snapshot_and_scientific_fields_without_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,run=make_manifest(root);job=runner_module.load_manifest(path)[0];run.mkdir()
            cfg=job['config_data'];total=job['expected_tokens']*job['train_flops_per_token']
            summary=dict(completed_budget=True,tokens_trained=job['expected_tokens'],train_flops=total)
            ev=dict(summary,metric='pure_cross_entropy',offset=VALIDATION_OFFSET,scored_tokens=VALIDATION_TARGETS,
                loss=1.,tokenizer_sha256=job['tokenizer_sha256'])
            rc=dict(cfg['train'],**cfg['model'],compute=job['compute'],steps=job['expected_steps'],
                flops_per_token=job['train_flops_per_token'],flops_accounting='fixture',
                config_sha256=job['config_sha256'],tokenizer_sha256=job['tokenizer_sha256'],
                data_identity=runtime_data_identity(job['data_identity']),model_code=dict(files=job['model_files_sha256']))
            for name,value in [('summary',summary),('final_eval',ev),('run_config',rc),('flops',cfg['accounting'])]:
                (run/(name+'.json')).write_text(json.dumps(value))
            runner=runner_module.Runner([job],[0],Path(os.sys.executable),root/'control',1,2,1,1)
            self.assertTrue(runner.valid_success(job,0))
            for stream,key,value in [('train','path','/wrong'),('train','mtime_ns',1),('validation','bytes',4)]:
                changed=copy.deepcopy(rc);changed['data_identity'][stream][key]=value
                (run/'run_config.json').write_text(json.dumps(changed));self.assertFalse(runner.valid_success(job,0))
            changed=copy.deepcopy(rc);del changed['data_identity']['validation']
            (run/'run_config.json').write_text(json.dumps(changed));self.assertFalse(runner.valid_success(job,0))
            changed=copy.deepcopy(rc);changed['mtp_coef']=.9
            (run/'run_config.json').write_text(json.dumps(changed));self.assertFalse(runner.valid_success(job,0))
            (run/'run_config.json').write_text(json.dumps(rc));job['validation_sha256']='wrong'
            self.assertFalse(runner.valid_success(job,0))

    def test_manifest_rejects_configuration_validation_partition_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config,_=make_manifest(root);data=json.loads(config.read_text())
            data['train']['final_eval_offset']+=1;config.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'partition'):runner_module.load_manifest(path)

    def test_timeout_terminates_only_child_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,run=make_manifest(root);job=runner_module.load_manifest(path)[0]
            python=root/'fake-python';python.write_text('#!/bin/sh\nsleep 30\n');python.chmod(0o755)
            output=root/'control';output.mkdir();runner=runner_module.Runner([job],[0],python,output,1,4,.2,.2)
            with mock.patch.object(runner_module,'gpu_idle',return_value=(True,{'memory_used_mib':0})), \
                    mock.patch.object(runner_module,'git_source',return_value=('fixture',[])), \
                    mock.patch.object(runner_module,'check_snapshot_files'):
                self.assertFalse(runner.run())
            self.assertEqual(runner.states['job1']['status'],'timed_out')
            self.assertEqual(runner.states['job1']['exitcode'],-15);self.assertTrue((run/'train.log').is_file())

    def test_completion_notification_covers_worker_exit_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,_=make_manifest(root);job=runner_module.load_manifest(path)[0]
            output=root/'control';output.mkdir();runner=runner_module.Runner([job],[0],Path(os.sys.executable),output,1,2,1,1)
            def delayed_worker(_gpu):
                runner.update('job1',status='complete');__import__('threading').Event().wait(.15)
            runner.worker=delayed_worker;started=__import__('time').monotonic()
            self.assertFalse(runner.run());self.assertLess(__import__('time').monotonic()-started,.5)
            self.assertEqual(runner.completed_workers,1)

    def test_busy_gpu_leaves_jobs_for_idle_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,_=make_manifest(root);jobs=runner_module.load_manifest(path)[:2]
            python=root/'fake-python';python.write_text('#!/bin/sh\nexit 0\n');python.chmod(0o755)
            output=root/'control';output.mkdir();runner=runner_module.Runner(jobs,[0,1],python,output,1,3,1,1)
            def card_state(gpu):return (gpu==1,{'memory_used_mib':0 if gpu==1 else 3})
            with mock.patch.object(runner_module,'gpu_idle',side_effect=card_state), \
                    mock.patch.object(runner_module,'git_source',return_value=('fixture',[])), \
                    mock.patch.object(runner_module,'check_snapshot_files'):
                self.assertFalse(runner.run())
            self.assertEqual(runner.gpu_states[0]['status'],'busy')
            self.assertTrue(all(state.get('gpu_id')==1 for state in runner.states.values()))
            self.assertTrue(all(state['status']=='incomplete' for state in runner.states.values()))


if __name__=='__main__':unittest.main()
