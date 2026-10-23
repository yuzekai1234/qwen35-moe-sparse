"""Q38 presets and whole-batch compute budgets."""
import unittest
from q38.presets import preset_names, load_preset, model_config, preset_budget

class Presets(unittest.TestCase):
    def test_q38_budgets(self):
        for name in preset_names():
            p=load_preset(name)
            cfg=model_config(p["model"],10112,2048)
            self.assertEqual((cfg.num_experts,cfg.num_experts_per_tok),(256,5))
            self.assertGreater(cfg.ngram_rows,0)
            self.assertTrue(cfg.mtp)
            tokens,fc=preset_budget(name,10112,2048)
            self.assertLessEqual(tokens*fc["train_flops_per_token"],p["compute"])
            self.assertGreater((tokens+524288)*fc["train_flops_per_token"],p["compute"])
