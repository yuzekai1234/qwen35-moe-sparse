"""Four-budget Q38 dense plan and fit-input isolation, without training data."""
from pathlib import Path
import json
import tempfile
import unittest

from q38.flops import count
from q38.presets import model_config
from scripts.make_q38_isoflops_configs import build_plan, write_plan, BUDGETS
from scripts.fit_q38_isoflops import fit_curves, c1_recipe, load_rows
import math
import numpy as np


class Plans(unittest.TestCase):
    def test_seventeen_points_and_no_c1(self):
        configs, m = build_plan("/unused/configs", "/unused/runs", "source")
        self.assertEqual(len(configs), 17)
        self.assertEqual([sum(j["compute"] == c for j in m["jobs"]) for c in BUDGETS.values()], [4,4,4,5])
        self.assertNotIn(3*10**17, [j["compute"] for j in m["jobs"]])
        self.assertEqual(m["validation"]["offset"], 2002536)
        self.assertEqual(m["validation"]["targets"], 16001875)
        for job in m["jobs"]:
            p = configs[Path(job["config"])]
            self.assertEqual(p["model"]["experts"], 0)
            self.assertEqual(p["train"]["max_steps"], 0)
            fc = count(model_config(p["model"], 10112, 4096).to_dict(), 4096)
            self.assertEqual(job["Na"], fc["active_params_excluding_embeddings"])
            self.assertEqual(job["expected_tokens"], p["train"]["tokens"])
            self.assertEqual(job["expected_steps"]*p["train"]["batch_tokens"], job["expected_tokens"])
            self.assertLessEqual(job["expected_tokens"]*job["train_flops_per_token"], job["compute"]*.995)
            self.assertEqual(p["model"]["ngram_rows"], p["comparison"]["ngram_rows_fixed_from_reference"])

    def test_existing_different_configuration_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, runs = Path(tmp)/"cfg", Path(tmp)/"runs"
            m = write_plan(out, runs, "source")
            self.assertEqual(write_plan(out, runs, "source", check=True), m)
            first = Path(m["jobs"][0]["config"])
            first.write_text("{}")
            with self.assertRaises(FileExistsError):
                write_plan(out, runs, "source")
            self.assertEqual(first.read_text(), "{}")


class Fitting(unittest.TestCase):
    def rows(self):
        _, manifest = build_plan("/unused/configs", "/unused/runs")
        rows = []
        for budget, compute in BUDGETS.items():
            jobs = [j for j in manifest["jobs"] if j["compute"] == compute]
            center = float(np.median(np.log10([j["Na"] for j in jobs])))
            for j in jobs:
                rows.append(dict(id=j["id"], budget=budget, target_compute=compute,
                    compute=j["expected_tokens"]*j["train_flops_per_token"], Na=j["Na"],
                    D=j["expected_tokens"], loss=2+.4*(math.log10(j["Na"])-center)**2))
        return rows

    def test_convex_curves_and_held_out_c1_recipe(self):
        fit = fit_curves(self.rows())
        self.assertEqual(len(fit["budget_fits"]), 4)
        self.assertEqual(len(fit["leave_one_budget_out"]), 4)
        for f in fit["budget_fits"]:
            self.assertLess(f["residual_rmse"], 1e-10)
            self.assertAlmostEqual(f["fitted_minimum_ce"], 2)
        cfg, prediction = c1_recipe(fit, Path("/unused/c1"))
        self.assertEqual(cfg["compute"], 3*10**17)
        self.assertEqual(cfg["model"]["experts"], 0)
        self.assertIsNone(prediction["quality_prediction"])

    def test_boundary_minimum_or_c1_input_is_rejected(self):
        rows = self.rows()
        for row in rows:
            row["loss"] = row["Na"]
        with self.assertRaisesRegex(ValueError, "boundary"):
            fit_curves(rows)
        rows = self.rows()
        rows.append(dict(rows[-1], target_compute=3*10**17))
        with self.assertRaisesRegex(ValueError, "without C1"):
            fit_curves(rows)

    def test_loader_requires_completed_identical_curve_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = write_plan(root/"cfg", root/"runs", "source")
            for job in m["jobs"]:
                p = json.loads(Path(job["config"]).read_text())
                run = Path(job["run_dir"]); run.mkdir(parents=True)
                total = job["expected_tokens"]*job["train_flops_per_token"]
                summary = dict(completed_budget=True, tokens_trained=job["expected_tokens"], train_flops=total)
                final = dict(summary, loss=2., metric="pure_cross_entropy", offset=m["validation"]["offset"],
                    scored_tokens=m["validation"]["targets"], tokenizer_sha256=m["validation"]["tokenizer_sha256"])
                rc = dict(p["train"], **p["model"], compute=job["compute"], steps=job["expected_steps"],
                    config_sha256=job["expected_config_sha256"], model_code=dict(files=m["model_files_sha256"]),
                    data_identity=dict(validation=dict(declared_sha256=m["validation"]["validation_sha256"])))
                for name, value in [("summary",summary),("final_eval",final),("run_config",rc),("flops",p["accounting"])]:
                    (run/(name+".json")).write_text(json.dumps(value))
            path = root/"cfg/manifest.json"
            _, rows = load_rows(path)
            self.assertEqual(len(rows), 17)
            first = Path(m["jobs"][0]["run_dir"])/"summary.json"
            bad = json.loads(first.read_text()); bad["completed_budget"] = False
            first.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"Incomplete"):
                load_rows(path)


if __name__ == "__main__":
    unittest.main()
