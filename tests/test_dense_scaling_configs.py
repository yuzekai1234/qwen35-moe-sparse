"""Frozen dense N/D extrapolation and configuration export, without training."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from q38.flops import count
from q38.presets import ROOT, model_config
from scripts.fit_q38_isoflops import training_recipe
from scripts.make_dense_configs import make_config
from scripts.make_dense_scaling_configs import (
    DEFAULT_OUTPUT, DEFAULT_RUN_ROOT, SOURCE, build_plan, load_fit, write_plan,
)
from scripts.make_q38_configs import BUDGETS, batch_for, shape_for
from scripts.make_q38_isoflops_configs import serialized


class DenseScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(SOURCE.read_bytes())
        cls.fit, cls.source = load_fit()
        cls.files, cls.selection = build_plan()

    def test_recorded_fit_identity_and_no_refit_or_c1_input(self):
        digest = hashlib.sha256(serialized(self.fit)).hexdigest()
        self.assertEqual(digest, self.record["prediction_files_sha256"]["fit.json"])
        self.assertEqual(self.source["fit_sha256"], digest)
        self.assertEqual(self.source["sha256"], hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self.assertEqual([len(f["points"]) for f in self.fit["budget_fits"]], [4, 4, 4, 5])
        self.assertNotIn(BUDGETS["c1"], {
            p["target_compute"] for f in self.fit["budget_fits"] for p in f["points"]})
        with patch("scripts.fit_q38_isoflops.fit_curves", side_effect=AssertionError("No refit")):
            self.assertEqual(build_plan(), (self.files, self.selection))

    def test_changed_fit_and_held_out_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"results.json"
            changed = copy.deepcopy(self.record)
            changed["isoflops"]["Na_log10_coefficients"][0] += .01
            path.write_bytes(serialized(changed))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_fit(path)
            changed = copy.deepcopy(self.record)
            changed["isoflops"]["budget_fits"][0]["points"][0]["target_compute"] = BUDGETS["c1"]
            changed["prediction_files_sha256"]["fit.json"] = hashlib.sha256(
                serialized(changed["isoflops"])).hexdigest()
            path.write_bytes(serialized(changed))
            with self.assertRaisesRegex(ValueError, "without C1"):
                load_fit(path)

    def test_c1_matches_the_actual_held_out_configuration_except_run_dir(self):
        golden = copy.deepcopy(self.record["c1_dense"]["config"])
        self.assertEqual(hashlib.sha256(serialized(golden)).hexdigest(),
                         self.record["prediction_files_sha256"]["c1-config.json"])
        golden["train"]["run_dir"] = str(DEFAULT_RUN_ROOT/"c1")
        self.assertEqual(self.files[DEFAULT_OUTPUT/"c1.json"], golden)
        prediction = self.selection["configurations"]["c1"]
        for key, value in self.fit["c1_prediction"].items():
            if key != "limitation":
                self.assertEqual(prediction[key], value, key)

    def test_all_tiers_preserve_training_rules_and_maximal_whole_batch_budget(self):
        for tier, compute in BUDGETS.items():
            with self.subTest(tier=tier):
                cfg = self.files[DEFAULT_OUTPUT/f"{tier}.json"]
                template = json.loads((ROOT/f"configs/q38_{tier}.json").read_bytes())
                train, accounting = cfg["train"], cfg["accounting"]
                self.assertEqual(cfg["compute"], compute)
                self.assertEqual(train["batch_tokens"], batch_for(compute))
                self.assertEqual(train["lr"], round(.3118*compute**-.125, 6))
                self.assertEqual(train["seq"], 4096)
                for key, value in template["train"].items():
                    if key not in {"run_dir", "tokens", "warmup"}:
                        self.assertEqual(train[key], value, key)
                resolved = model_config(cfg["model"], 10112, train["seq"])
                self.assertTrue(resolved.mhc_enabled)
                self.assertEqual(resolved.qsa_mode, "learned")
                self.assertEqual(resolved.num_experts, 0)
                self.assertTrue(resolved.mtp)
                self.assertIn("isoflops-gzy10k-16b-v1", train["data"])
                fc = count(resolved.to_dict(), train["seq"])
                for key, value in fc.items():
                    self.assertEqual(accounting[key], value, key)
                f, d, batch = fc["train_flops_per_token"], train["tokens"], train["batch_tokens"]
                budget = int(compute*(1-accounting["compute_reserve_fraction"]))
                self.assertEqual(accounting["compute_reserve_fraction"], .005)
                self.assertEqual(d % batch, 0)
                self.assertEqual(accounting["steps"]*batch, d)
                self.assertEqual(accounting["actual_train_flops"], f*d)
                self.assertLessEqual(f*d, budget)
                self.assertGreater(f*(d+batch), budget)
                self.assertEqual(train["warmup"], min(accounting["steps"],
                    max(train["warmup_min"], round(accounting["steps"]*train["warmup_ratio"]))))
                self.assertEqual(cfg["model"]["ngram_rows"],
                                 cfg["comparison"]["ngram_rows_fixed_from_reference"])

    def test_every_selection_minimizes_the_original_grid_score(self):
        for tier in BUDGETS:
            with self.subTest(tier=tier):
                prediction = self.selection["configurations"][tier]
                template = json.loads((ROOT/f"configs/q38_{tier}.json").read_bytes())
                scores = []
                for hidden in range(64, 1601, 64):
                    reference = copy.deepcopy(template)
                    reference["model"] = shape_for(hidden)
                    cfg = make_config(reference, DEFAULT_RUN_ROOT/tier)
                    n, d = cfg["accounting"]["active_params_excluding_embeddings"], cfg["train"]["tokens"]
                    score = math.log(n/prediction["predicted_Na"])**2 + math.log(d/prediction["predicted_D"])**2
                    scores.append((score, hidden, n, d))
                best = min(scores)
                self.assertEqual((prediction["selection_score"], prediction["selected_hidden"],
                                  prediction["selected_Na"], prediction["selected_D"]), best)
                self.assertIsNone(prediction["quality_prediction"])
                self.assertEqual(prediction["validation_status"],
                    "held_out_c1_validated" if tier == "c1" else "untrained_extrapolation")

    def test_unknown_tiers_are_rejected(self):
        for tier in ("c0", "C1", "3e17", "baseline"):
            with self.subTest(tier=tier), self.assertRaisesRegex(ValueError, "tier"):
                training_recipe(self.fit, tier, Path("/unused"))

    def test_generated_files_bind_config_hashes_and_are_reproducible(self):
        for tier in BUDGETS:
            path = DEFAULT_OUTPUT/f"{tier}.json"
            self.assertEqual(path.read_bytes(), serialized(self.files[path]))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             self.selection["configurations"][tier]["config_sha256"])
        self.assertEqual((DEFAULT_OUTPUT/"selection.json").read_bytes(), serialized(self.selection))
        self.assertEqual(write_plan(check=True), self.selection)

    def test_check_does_not_write_and_missing_check_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"configs"
            with self.assertRaises(FileNotFoundError):
                write_plan(output, check=True)
            self.assertFalse(output.exists())
            expected = write_plan(output)
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            with patch.object(Path, "write_bytes", side_effect=AssertionError("check wrote a file")):
                self.assertEqual(write_plan(output, check=True), expected)
            self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}, before)

    def test_all_destinations_are_checked_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"configs"
            output.mkdir()
            (output/"c4.json").write_text("{}\n")
            with self.assertRaises(FileExistsError):
                write_plan(output)
            self.assertEqual((output/"c4.json").read_text(), "{}\n")
            self.assertEqual(list(output.iterdir()), [output/"c4.json"])


if __name__ == "__main__":
    unittest.main()
