"""One small Q38 forward/backward and portable HF export smoke check."""
import tempfile
import unittest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from q38.flops import count
from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM
from q38.presets import model_config
from q38.train import LossModel
from q38.muon import HybridOptimizer

class CE(torch.nn.Module):
    def forward(self, weight, hidden, labels):
        return F.cross_entropy(F.linear(hidden, weight), labels)

class Model(unittest.TestCase):
    def test_q38_smoke(self):
        torch.manual_seed(0)
        cfg = model_config(dict(layers=4, hidden=32, heads=2, kv_heads=1, lin_k_heads=1,
            lin_v_heads=2, experts=8, topk=2, moe_ffn=8, shared_ffn=8, ngram_rows=64, qsa_blocks=1),
            128, 128, eos_token_id=0)
        model = Qwen3_8MoeForCausalLM(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()), count(cfg.to_dict(),128)["total_params_all"])
        ids = torch.randint(1,128,(1,128))
        wrapped = LossModel(model,8,2,ce=CE())
        ce, aux, mtp, kl = wrapped(ids,ids.roll(-1,1))
        loss = ce + .001*aux + .3*mtp + .05*kl
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        opt=HybridOptimizer(model,lr=.001,weight_decay=.1)
        opt.step(); opt.zero_grad()
        self.assertIsNotNone(opt.state_dict()["muon"])
        model.eval()
        with torch.no_grad():
            expected = model(input_ids=ids,use_cache=False).logits
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded=AutoModelForCausalLM.from_pretrained(tmp,trust_remote_code=True,local_files_only=True).eval()
            with torch.no_grad():
                actual=loaded(input_ids=ids,use_cache=False).logits
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertEqual(loaded.config.model_type,"qwen38_moe_sparse")

if __name__ == "__main__": unittest.main()
