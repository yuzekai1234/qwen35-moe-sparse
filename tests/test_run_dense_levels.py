"""CPU-only contracts for the current dense sequential campaign."""
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import run_dense_levels as runner


def make_plan(root):
    with mock.patch.object(runner.subprocess, "run"):
        return runner.make_plan(root / "campaign", root / "A.jsonl", [str(i) for i in range(8)], "commit")


def training_outputs(job, plan):
    config = job["runtime_config"]
    train, accounting = config["train"], config["accounting"]
    root = Path(train["run_dir"])
    (root / "hf").mkdir(parents=True)
    total_flops = train["tokens"] * accounting["train_flops_per_token"]
    rc = dict(config["model"], **train, compute=config["compute"], steps=job["expected_steps"],
              accum=train["batch_tokens"] // (8 * 4096 * train["micro_bs"]),
              config_sha256=runner.sha256(job["runtime_path"]),
              data_identity=plan["data_identity"], tokenizer_sha256=runner.TOKENIZER_SHA256)
    summary = dict(tokens_trained=train["tokens"], train_flops=total_flops, world=8,
                   completed_budget=True, final_eval_loss=2.8, elapsed_s=100,
                   flops_accounting=accounting["accounting_method"])
    final = dict(tokens_trained=train["tokens"], train_flops=total_flops, completed_budget=True,
                 loss=2.8, metric="pure_cross_entropy", offset=runner.VALIDATION_OFFSET,
                 scored_tokens=runner.VALIDATION_TARGETS, tokenizer_sha256=runner.TOKENIZER_SHA256,
                 flops_accounting=accounting["accounting_method"])
    flops = dict(train_flops=total_flops, train_tokens=train["tokens"],
                 train_flops_per_token=accounting["train_flops_per_token"])
    for name, data in (("run_config", rc), ("summary", summary), ("final_eval", final), ("flops", flops)):
        runner.atomic_json(root / (name + ".json"), data)
    (root / "training_metrics.csv").write_text(
        f"step,tokens,loss,grad_norm\n0,{train['batch_tokens']},9,1\n"
        f"{job['expected_steps'] - 1},{train['tokens']},3,0.1\n")
    (root / "ckpt.pt").write_bytes(b"fixture")
    return root


def evaluation_output(job):
    active = job["runtime_config"]["accounting"]["active_params_excluding_embeddings"]
    nll = math.log(2) * runner.EVALUATION_BYTES * 1.1
    return dict(model_path=str(Path(job["runtime_config"]["train"]["run_dir"]) / "hf"),
                bpb=1.1, total_nll_nats=nll, forward_seconds=100,
                evaltime=100 / ((active / 1e9) * (runner.EVALUATION_TOKENS / 1e6)),
                active_params_excluding_embeddings=active, gpu_count=1, batch_size=16,
                sequence_length=4096, scored_tokens=runner.EVALUATION_TOKENS,
                text_bytes=runner.EVALUATION_BYTES, timing_enabled=True)


class DenseLevels(unittest.TestCase):
    def test_import_has_no_torch_or_cuda_side_effect(self):
        subprocess.run([sys.executable, "-c", "import scripts.run_dense_levels,sys; assert 'torch' not in sys.modules"],
                       cwd=runner.ROOT, check=True, timeout=20)

    def test_plan_uses_only_current_presets_and_eight_gpu_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_plan(root)
            self.assertFalse((root / "campaign").exists())
            self.assertEqual([j["level"] for j in plan["jobs"]], list(runner.LEVELS))
            self.assertEqual([j["expected_steps"] for j in plan["jobs"]], [6206, 4514, 7231, 6438])
            for job in plan["jobs"]:
                canonical = runner.read_json(job["canonical_config"])
                actual = copy.deepcopy(job["runtime_config"])
                actual["train"].update({key: canonical["train"][key] for key in ("run_dir", "resume", "max_steps")})
                self.assertEqual(actual, canonical)
                self.assertIn("configs/dense_isoflops/", job["canonical_config"])
                self.assertIn("--nproc_per_node=8", job["train_command"])
                self.assertIn("q38.evaluate_single_gpu", job["evaluation_command"])

    def test_dry_run_does_not_check_remote_data_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["--run-root", str(root / "campaign"), "--evaluation-jsonl", str(root / "missing.jsonl"),
                    "--expected-commit", "commit"]
            with mock.patch.object(runner.subprocess, "run"), mock.patch.object(runner, "check_inputs") as check, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(args), 0)
            check.assert_not_called()
            self.assertFalse((root / "campaign").exists())

    def test_rejects_old_family_and_duplicate_devices_and_existing_root(self):
        for value in ("0,1,2,3", "0,1,2,3,4,5,6,6", "0,1,2,3,4,5,6,x"):
            with self.assertRaises(ValueError):
                runner.parse_gpus(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_read = runner.read_json
            for change in ({"mhc_enabled": 1}, {"qsa_mode": "learned"}, {"experts": 64}):
                def altered(path):
                    value = real_read(path)
                    if Path(path).name == "c1.json":
                        value["model"].update(change)
                    return value
                with mock.patch.object(runner, "read_json", side_effect=altered), self.assertRaises(ValueError):
                    make_plan(root)
            (root / "campaign").mkdir()
            with self.assertRaisesRegex(ValueError, "new absolute"):
                make_plan(root)

    def test_child_environment_is_single_gpu_and_offline_without_proxy_or_ddp(self):
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "secret", "RANK": "7", "WORLD_SIZE": "8"}):
            env = runner.child_environment(["4"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4")
        self.assertNotIn("RANK", env)
        self.assertNotIn("WORLD_SIZE", env)
        self.assertFalse(any("proxy" in key.lower() for key in env))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_incomplete_or_wrong_family_training_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = make_plan(Path(directory))
            job = plan["jobs"][0]
            Path(job["runtime_path"]).parent.mkdir(parents=True)
            runner.atomic_json(job["runtime_path"], job["runtime_config"])
            root = training_outputs(job, plan)
            with mock.patch.object(runner, "validate_hf", return_value={}):
                self.assertEqual(runner.validate_training(job, plan)["steps"], 6206)
                for name, key, value in (("summary", "completed_budget", False),
                                         ("summary", "tokens_trained", 100),
                                         ("final_eval", "scored_tokens", 100),
                                         ("run_config", "mhc_enabled", 1),
                                         ("run_config", "qsa_mode", "learned")):
                    path = root / (name + ".json")
                    original = path.read_bytes()
                    data = runner.read_json(path)
                    data[key] = value
                    runner.atomic_json(path, data)
                    with self.assertRaises(ValueError, msg=key):
                        runner.validate_training(job, plan)
                    path.write_bytes(original)

    def test_hf_wrong_family_wrong_layer_types_and_non_bf16_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = make_plan(Path(directory))
            cfg = plan["jobs"][0]["runtime_config"]
            model = cfg["model"]
            root = Path(directory) / "hf"
            root.mkdir()
            hf = dict(num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0,
                      shared_expert_intermediate_size=0, mhc_enabled=False, qsa_mode="full", mtp=True,
                      hidden_size=model["hidden"], num_hidden_layers=model["layers"],
                      intermediate_size=model["dense_ffn"], ngram_rows=model["ngram_rows"],
                      layer_types=cfg["accounting"]["layer_types"])
            (root / "tokenizer.json").write_bytes(b"fixture")
            weight = root / "model.safetensors"
            def write_weight(dtype):
                header = json.dumps({"weight": dict(dtype=dtype, shape=[1], data_offsets=[0, 2])}).encode()
                weight.write_bytes(len(header).to_bytes(8, "little") + header + b"00")
            write_weight("BF16")
            real_sha = runner.sha256
            with mock.patch.object(runner, "sha256", side_effect=lambda p: runner.TOKENIZER_SHA256
                                   if Path(p).name == "tokenizer.json" else real_sha(p)):
                runner.atomic_json(root / "config.json", hf)
                self.assertIn("model.safetensors", runner.validate_hf(root, cfg))
                for change in ({"mhc_enabled": True}, {"qsa_mode": "learned"},
                               {"layer_types": ["full_attention"] * model["layers"]}):
                    runner.atomic_json(root / "config.json", dict(hf, **change))
                    with self.assertRaises(ValueError):
                        runner.validate_hf(root, cfg)
                runner.atomic_json(root / "config.json", hf)
                write_weight("F8_E4M3")
                with self.assertRaisesRegex(ValueError, "BF16"):
                    runner.validate_hf(root, cfg)

    def test_evaluation_requires_full_a_single_gpu_and_correct_metric_formulas(self):
        with tempfile.TemporaryDirectory() as directory:
            job = make_plan(Path(directory))["jobs"][0]
            good = evaluation_output(job)
            runner.validate_evaluation(job, good)
            for key, value in (("gpu_count", 8), ("scored_tokens", 100), ("text_bytes", 100),
                               ("evaltime", 10), ("bpb", float("nan")), ("model_path", "other")):
                with self.assertRaises(ValueError, msg=key):
                    runner.validate_evaluation(job, dict(good, **{key: value}))

    def campaign_fixture(self, root, fail=None):
        plan = make_plan(root)
        calls = []
        def run(command, devices, log, timeout, heartbeat):
            stage, level = log.stem, log.parent.name
            calls.append((level, stage, devices))
            log.write_text("fixture\n")
            if (level, stage) == fail:
                raise RuntimeError("fixture stage failure")
            job = next(j for j in plan["jobs"] if j["level"] == level)
            if stage == "training":
                training_outputs(job, plan)
            else:
                runner.atomic_json(log.parent / "evaluation.json", evaluation_output(job))
        with mock.patch.object(runner, "check_inputs"), mock.patch.object(runner, "environment_record", return_value={}), \
                mock.patch.object(runner, "wait_for_gpus"), mock.patch.object(runner, "run_command", side_effect=run), \
                mock.patch.object(runner, "validate_hf", return_value={}):
            campaign = runner.Campaign(plan, 10, 10, 10)
            try:
                campaign.execute()
            except RuntimeError:
                if fail is None:
                    raise
        return plan, calls

    def test_sequential_train_evaluate_then_next_level_with_completion_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, calls = self.campaign_fixture(Path(directory))
            self.assertEqual([(level, stage) for level, stage, _ in calls],
                             [(level, stage) for level in runner.LEVELS for stage in ("training", "evaluation")])
            self.assertTrue(all(len(devices) == (8 if stage == "training" else 1) for _, stage, devices in calls))
            final = runner.read_json(Path(plan["run_root"]) / "completion.json")
            self.assertEqual(final["status"], "completed")
            self.assertTrue(all(j["status"] == "completed" for j in final["jobs"]))

    def test_fail_stop_does_not_start_next_level_or_report_completion(self):
        for stage in ("training", "evaluation"):
            with tempfile.TemporaryDirectory() as directory:
                plan, calls = self.campaign_fixture(Path(directory), fail=("c1", stage))
                final = runner.read_json(Path(plan["run_root"]) / "completion.json")
                self.assertEqual(final["status"], "failed")
                self.assertTrue(all(level == "c1" for level, _, _ in calls))
                self.assertEqual(final["jobs"][1]["status"], "pending")
                with self.assertRaises(FileExistsError):
                    runner.Campaign(plan, 10, 10, 10).execute()

    @unittest.skipUnless(hasattr(os, "pidfd_open"), "Requires Linux pidfds")
    def test_timeout_stops_only_owned_child_not_other_process(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(TimeoutError):
                runner.run_command([sys.executable, "-c", "import time; time.sleep(60)"], [],
                                   Path(directory) / "stage.log", 0.1, lambda pid: None)
            self.assertIsNone(other.poll())
        finally:
            other.terminate()
            other.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
