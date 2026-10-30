"""Combined no-mHC/full-attention baseline: real modules, causal math and export."""
import json
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM
from q38.flops import count, validate_resume_accounting
from q38.models.qwen3_8_moe import Qwen3_8MoeForCausalLM, Qwen3_8MoeTextConfig
from q38.models.qwen3_8_moe import modeling_qwen3_8_moe as M
from q38.presets import ROOT, dense_shape, model_config
from q38.train import LossModel, NEW_RESUME_DEFAULTS, RESUME_KEYS
from scripts.make_baseline_config import make_config
from tests.test_dense import CE, tiny_shape


class Baseline(unittest.TestCase):
    def small_config(self, **changes):
        shape = dict(dense_shape(tiny_shape(), 128, 128), mhc_enabled=0, qsa_mode='full')
        shape.update(changes)
        return model_config(shape, 128, 128, eos_token_id=0)

    def test_combined_budget_and_registered_parameter_count(self):
        config = make_config()
        self.assertEqual(config, json.loads((ROOT/'configs/baseline.json').read_text()))
        cfg = model_config(config['model'], 10112, 4096)
        fc = count(cfg.to_dict(), 4096)
        with torch.device('meta'):
            model = Qwen3_8MoeForCausalLM(cfg)
        self.assertFalse(model.model.mhc)
        self.assertFalse(model.model.indexers)
        self.assertFalse(any(isinstance(m, (M.MHC, M.Indexer)) for m in model.modules()))
        actual = sum(p.numel() for p in model.parameters())
        embeddings = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Embedding))
        self.assertEqual(actual, fc['total_params_all'])
        self.assertEqual(actual-embeddings-sum(p.numel() for p in model.mtp.parameters()),
                         fc['active_params_excluding_embeddings'])
        self.assertEqual(fc['active_params_excluding_embeddings'], 33776160)
        self.assertEqual(fc['train_flops_per_token'], 366904320)
        for name in ['mhc_projections','mhc_mix','indexer_projections','indexer_score']:
            self.assertEqual(fc['breakdown'][name], 0)
        self.assertEqual(fc['distill_target_flops'], 0)
        self.assertEqual(fc['breakdown']['attn_score'], 2*2*4096*20*64)
        tokens, batch = config['train']['tokens'], config['train']['batch_tokens']
        budget = config['compute']*995//1000
        self.assertLessEqual(tokens*fc['train_flops_per_token'], budget)
        self.assertGreater((tokens+batch)*fc['train_flops_per_token'], budget)

    def test_training_is_causal_has_mtp_gradients_and_never_calls_indexer_kl(self):
        torch.manual_seed(4)
        model = Qwen3_8MoeForCausalLM(self.small_config())
        ids = torch.randint(1,128,(1,91))
        with patch.object(M, 'block_mass', side_effect=AssertionError('KL must be absent')), \
             patch.object(M.Indexer, 'forward', side_effect=AssertionError('Indexer must be absent')):
            ce, aux, mtp, kl = LossModel(model,0,0,ce=CE())(ids, ids.roll(-1,1))
            self.assertEqual((aux.item(),kl.item()), (0.,0.))
            (ce+.3*mtp).backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        model.eval()
        changed = ids.clone(); changed[:,60:] = (changed[:,60:]+17)%127+1
        with torch.no_grad():
            before = model(input_ids=ids).logits
            after = model(input_ids=changed).logits
        self.assertEqual(before.shape,(1,91,128))
        torch.testing.assert_close(before[:,:60],after[:,:60],rtol=0,atol=0)

    def test_full_attention_matches_explicit_gqa_with_gate(self):
        cfg = self.small_config(mtp=0)
        attn = M.Qwen3_8MoeAttention(cfg, 3)
        x = torch.randn(1,19,32)
        cos,sin = torch.ones(1,19,16),torch.zeros(1,19,16)
        actual,kl = M.qsa_attention(attn,x,cos,sin,None,1,True,False)
        q,g = attn.q_proj(x).view(1,19,2,128).chunk(2,-1)
        q = attn.q_norm(q).transpose(1,2)
        k = attn.k_norm(attn.k_proj(x).view(1,19,1,64)).transpose(1,2).repeat_interleave(2,1)
        v = attn.v_proj(x).view(1,19,1,64).transpose(1,2).repeat_interleave(2,1)
        scores = (q@k.transpose(-1,-2))*attn.scaling
        scores = scores.masked_fill(~torch.ones(19,19,dtype=torch.bool).tril(),float('-inf'))
        y = (scores.softmax(-1)@v).transpose(1,2).reshape(1,19,128)
        expected = attn.o_proj(y*torch.sigmoid(g.reshape(1,19,128)))
        self.assertIsNone(kl)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_ngram_short_inputs_preserve_the_token_axis(self):
        ngram = M.NGramEmbedding(32,128,64,eos_id=0)
        for length in (1,2):
            ids = torch.arange(1,length+1).unsqueeze(0)
            h = torch.randn(1,length,32)
            for indices in ngram.indices(ids):
                self.assertEqual(indices.shape,ids.shape)
            self.assertEqual(ngram.delta(h,ids).shape,h.shape)

    def test_portable_export_preserves_combined_architecture_and_logits(self):
        model = Qwen3_8MoeForCausalLM(self.small_config()).eval()
        ids = torch.randint(1,128,(1,65))
        with torch.no_grad():
            expected = model(input_ids=ids).logits
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            loaded = AutoModelForCausalLM.from_pretrained(d,trust_remote_code=True,local_files_only=True).eval()
            self.assertFalse(loaded.config.mhc_enabled)
            self.assertEqual(loaded.config.qsa_mode,'full')
            self.assertFalse(loaded.model.mhc)
            self.assertFalse(loaded.model.indexers)
            with torch.no_grad():
                actual = loaded(input_ids=ids).logits
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_legacy_defaults_and_resume_accounting_are_explicit(self):
        shape = dense_shape(tiny_shape(),128,128)
        old = model_config(shape,128,128)
        self.assertTrue(old.mhc_enabled)
        self.assertEqual(old.qsa_mode,'learned')
        raw = old.to_dict()
        raw.pop('mhc_enabled');raw.pop('qsa_mode')
        loaded = Qwen3_8MoeTextConfig.from_dict(raw)
        self.assertTrue(loaded.mhc_enabled)
        self.assertEqual(loaded.qsa_mode,'learned')
        before = count(old.to_dict(),128)
        for options in [dict(mhc_enabled=0),dict(qsa_mode='full'),dict(mhc_enabled=0,qsa_mode='full')]:
            new = count(model_config(dict(shape,**options),128,128).to_dict(),128)
            with self.assertRaisesRegex(ValueError,'FLOPs accounting changed'):
                validate_resume_accounting(dict(flops_per_token=before['train_flops_per_token'],
                    flops_accounting=before['accounting_method']),new)
        for field,default in [('mhc_enabled',1),('qsa_mode','learned')]:
            self.assertIn(field,RESUME_KEYS)
            self.assertEqual(NEW_RESUME_DEFAULTS[field],default)


if __name__ == '__main__':
    unittest.main()
