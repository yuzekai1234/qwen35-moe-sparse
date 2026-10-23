"""Q38 presets and whole-batch compute budgets."""
import unittest
from q38.presets import preset_names, load_preset, model_config, preset_budget

class Presets(unittest.TestCase):
    def test_q38_budgets(self):
        for name in preset_names():
            p=load_preset(name)
            cfg=model_config(p["model"],10112,4096)
            self.assertEqual((cfg.num_experts,cfg.num_experts_per_tok),(64,5))
            self.assertGreater(cfg.ngram_rows,0)
            self.assertTrue(cfg.mtp)
            tokens,fc=preset_budget(name,10112,4096)
            self.assertEqual(tokens,p["train"]["tokens"])
            self.assertEqual(fc["train_flops_per_token"],p["accounting"]["train_flops_per_token"])
            self.assertEqual(fc["accounting_method"],p["accounting"]["accounting_method"])
            target = p["compute"] * (1 - p["accounting"]["compute_reserve_fraction"])
            self.assertLessEqual(tokens*fc["train_flops_per_token"],target)
            self.assertGreater((tokens+p["train"]["batch_tokens"])*fc["train_flops_per_token"],target)
            self.assertGreater(fc["active_nonvocab_params"]/fc["total_nonvocab_params"],.10)
