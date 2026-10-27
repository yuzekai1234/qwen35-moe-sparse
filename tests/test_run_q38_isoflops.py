import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_q38_isoflops", ROOT / "scripts/run_q38_isoflops.py")
runner_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner_module)


def make_manifest(root):
    run_dir = root / "run"
    config = {
        "compute": 1000,
        "model": {},
        "train": {"run_dir": str(run_dir), "tokens": 10, "batch_tokens": 2, "final_eval_tokens": 5,
                  "final_eval_offset": 3},
        "accounting": {"train_flops_per_token": 7, "active_params_excluding_embeddings": 11,
                       "accounting_method": "fixture"},
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "jobs": [{
        "id": "job1", "config": str(config_path), "run_dir": str(run_dir), "compute": 1000,
        "expected_tokens": 10, "expected_steps": 5, "train_flops_per_token": 7, "Na": 11,
        "final_eval_offset": 3, "final_eval_targets": 5, "tokenizer_sha256": "toksha",
    }], "validation": {"offset": 3, "targets": 5, "tokenizer_sha256": "toksha",
                       "validation_sha256": "validsha"}}))
    return manifest_path, config_path, run_dir


class RunnerTests(unittest.TestCase):
    def test_dry_run_preserves_python_symlink_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = make_manifest(root)
            venv = root / "venv"
            venv.mkdir()
            python_link = venv / "python"
            python_link.symlink_to(os.path.realpath(os.sys.executable))
            output = root / "control"
            argv = ["runner", "--manifest", str(manifest), "--gpus", "0", "--python", str(python_link),
                    "--output", str(output)]
            with mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(runner_module.main(), 0)
            self.assertIn(str(python_link), stdout.getvalue())
            self.assertFalse(output.exists())

    def test_manifest_rejects_mismatched_expected_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = make_manifest(root)
            data = json.loads(manifest.read_text())
            data["jobs"][0]["expected_steps"] = 4
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "不一致"):
                runner_module.load_manifest(manifest)

    def test_manifest_inherits_top_level_source_pin_and_main_rejects_wrong_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = make_manifest(root)
            data = json.loads(manifest.read_text())
            data["source_commit"] = "expected-commit"
            manifest.write_text(json.dumps(data))
            job = runner_module.load_manifest(manifest)[0]
            self.assertEqual(job["source_commit"], "expected-commit")
            output = root / "control"
            python = Path(os.sys.executable)
            argv = ["runner", "--manifest", str(manifest), "--gpus", "0", "--python", str(python),
                    "--output", str(output), "--execute"]
            with mock.patch("sys.argv", argv), mock.patch.object(runner_module, "git_source", return_value=("wrong-commit", [])), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "源码版本不符合 manifest"):
                runner_module.main()
            self.assertFalse(output.exists())

    def test_success_guard_checks_manifest_validation_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, run_dir = make_manifest(root)
            job = runner_module.load_manifest(manifest)[0]
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(json.dumps({"completed_budget": True, "tokens_trained": 10, "train_flops": 70}))
            (run_dir / "final_eval.json").write_text(json.dumps({"completed_budget": True, "tokens_trained": 10,
                "train_flops": 70, "metric": "pure_cross_entropy", "offset": 3, "scored_tokens": 5,
                "loss": 1.0, "tokenizer_sha256": "toksha"}))
            (run_dir / "run_config.json").write_text(json.dumps({"run_dir": str(run_dir), "compute": 1000,
                "tokens": 10, "batch_tokens": 2, "steps": 5, "seq": None, "final_eval_tokens": 5,
                "final_eval_offset": 3, "flops_per_token": 7, "flops_accounting": "fixture",
                "config_sha256": job["config_sha256"],
                "tokenizer_sha256": "toksha", "config": {},
                "data_identity": {"validation": {"declared_sha256": "validsha"}}}))
            (run_dir / "flops.json").write_text(json.dumps({"active_params_excluding_embeddings": 11,
                "train_flops_per_token": 7, "accounting_method": "fixture"}))
            runner = runner_module.Runner([job], [0], Path(os.sys.executable), root / "control",
                                          job_timeout=1, queue_timeout=2, heartbeat=1, grace=1)
            self.assertTrue(runner.valid_success(job, 0))
            job["validation_sha256"] = "wrong"
            self.assertFalse(runner.valid_success(job, 0))

    def test_manifest_rejects_configuration_validation_partition_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, config_path, _ = make_manifest(root)
            data = json.loads(manifest.read_text())
            data["validation"] = dict(offset=3, targets=5)
            data["jobs"][0].update(final_eval_offset=3, final_eval_targets=5)
            manifest.write_text(json.dumps(data))
            config = json.loads(config_path.read_text())
            config["train"]["final_eval_offset"] = 4
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "partition"):
                runner_module.load_manifest(manifest)

    def test_timeout_terminates_only_child_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, run_dir = make_manifest(root)
            job = runner_module.load_manifest(manifest)[0]
            fake_python = root / "fake-python"
            fake_python.write_text("#!/bin/sh\nsleep 30\n")
            fake_python.chmod(0o755)
            output = root / "control"
            output.mkdir()
            runner = runner_module.Runner([job], [0], fake_python, output,
                                          job_timeout=1, queue_timeout=4, heartbeat=0.2, grace=0.2)
            runner.started = __import__("time").monotonic()
            runner.started_utc = runner_module.utc()
            with mock.patch.object(runner_module, "gpu_idle", return_value=(True, {"memory_used_mib": 0})), \
                    mock.patch.object(runner_module, "git_source", return_value=("fixture", [])):
                self.assertFalse(runner.run())
            self.assertEqual(runner.states["job1"]["status"], "timed_out")
            self.assertEqual(runner.states["job1"]["exitcode"], -15)
            self.assertTrue((run_dir / "train.log").is_file())

    def test_completion_notification_covers_worker_exit_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = make_manifest(root)
            job = runner_module.load_manifest(manifest)[0]
            output = root / "control"
            output.mkdir()
            runner = runner_module.Runner([job], [0], Path(os.sys.executable), output,
                                          job_timeout=1, queue_timeout=2, heartbeat=1, grace=1)
            release = __import__("threading").Event()

            def delayed_worker(_gpu):
                runner.update("job1", status="complete")
                release.wait(timeout=0.15)

            runner.worker = delayed_worker
            started = __import__("time").monotonic()
            self.assertFalse(runner.run())
            self.assertLess(__import__("time").monotonic() - started, 0.5)
            self.assertEqual(runner.completed_workers, 1)

    def test_busy_gpu_leaves_jobs_for_idle_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = make_manifest(root)
            data = json.loads(manifest.read_text())
            second = dict(data["jobs"][0], id="job2", run_dir=str(root / "run2"))
            config_path = Path(data["jobs"][0]["config"])
            config = json.loads(config_path.read_text())
            config["train"]["run_dir"] = second["run_dir"]
            config_path2 = root / "config2.json"
            config_path2.write_text(json.dumps(config))
            second["config"] = str(config_path2)
            data["jobs"].append(second)
            manifest.write_text(json.dumps(data))
            jobs = runner_module.load_manifest(manifest)
            fake_python = root / "fake-python"
            fake_python.write_text("#!/bin/sh\nexit 0\n")
            fake_python.chmod(0o755)
            output = root / "control"
            output.mkdir()
            runner = runner_module.Runner(jobs, [0, 1], fake_python, output,
                                          job_timeout=1, queue_timeout=3, heartbeat=1, grace=1)
            runner.started = __import__("time").monotonic()

            def card_state(gpu):
                return (False, {"memory_used_mib": 3}) if gpu == 0 else (True, {"memory_used_mib": 0})

            with mock.patch.object(runner_module, "gpu_idle", side_effect=card_state), \
                    mock.patch.object(runner_module, "git_source", return_value=("fixture", [])):
                self.assertFalse(runner.run())
            self.assertEqual(runner.gpu_states[0]["status"], "busy")
            self.assertTrue(all(state.get("gpu_id") == 1 for state in runner.states.values()))
            self.assertTrue(all(state["status"] == "incomplete" for state in runner.states.values()))


if __name__ == "__main__":
    unittest.main()
