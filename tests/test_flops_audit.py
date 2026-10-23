"""Independent operation-level regressions; no GPU or downloaded source needed."""
import copy
import unittest

from q38.flops import _count_backbone as count, count as count_q38, validate_resume_accounting
from scripts.audit_flops import fla_chunk_gemm_ledger, reference_forward_terms, legacy_train_flops_per_token


def config():
    return dict(hidden_size=128,num_hidden_layers=4,vocab_size=10112,num_experts=64,
                num_experts_per_tok=4,moe_intermediate_size=64,shared_expert_intermediate_size=64,
                num_attention_heads=4,num_key_value_heads=1,head_dim=64,
                linear_num_key_heads=2,linear_num_value_heads=4,
                linear_key_head_dim=64,linear_value_head_dim=64,linear_conv_kernel_dim=4,
                layer_types=['linear_attention']*3+['full_attention'])


class AccountingAudit(unittest.TestCase):
    def test_causal_core_matches_two_triangular_attention_matmuls(self):
        c=config()
        # Two causal GEMMs, each FMA=2, causal factor 1/2. NOT full square work.
        expected=2*8192*4*64
        terms=reference_forward_terms(c,8192)
        self.assertEqual(terms['full_causal_core'],expected)
        self.assertEqual(count(c,8192)['breakdown']['attn_score'],expected)
        self.assertEqual(reference_forward_terms(c,4096)['full_causal_core'],expected//2)

    def test_output_gate_is_included_as_a_separate_projection(self):
        c=config();h=c['hidden_size'];q=4*64;kv=64
        expected=2*h*(q+kv+kv+q)+2*h*q  # Q, K, V, gate, O
        self.assertEqual(reference_forward_terms(c,8192)['full_projections_including_output_gate'],expected)

    def test_reference_projection_and_core_are_distinct(self):
        c=config();t=reference_forward_terms(c,8192)
        self.assertEqual(t['gdn_recurrent_core_reference'],3*8*4*64**2)
        self.assertEqual(count(c,8192)['breakdown']['gdn_core'],3*4*8*64**2)
        # GVA changes core work with value heads, not key-head count.
        d=copy.deepcopy(c);d['linear_num_value_heads']=8
        self.assertEqual(reference_forward_terms(d,8192)['gdn_recurrent_core_reference'],2*t['gdn_recurrent_core_reference'])

    def test_calibrated_total_matches_reference_plus_actual_router_and_gate(self):
        for dk,dv in [(64,64),(64,128),(128,64)]:
            c=config();c['linear_key_head_dim']=dk;c['linear_value_head_dim']=dv
            for seq in [128,8192]:
                with self.subTest(dk=dk,dv=dv,seq=seq):
                    result=count(c,seq)
                    reference=3*sum(reference_forward_terms(c,seq).values())
                    extra=6*4*128*(64+1)
                    self.assertEqual(result['megatron_reference_train_flops_per_token'],reference)
                    self.assertEqual(result['extra_router_shared_gate_train_flops_per_token'],extra)
                    self.assertEqual(result['train_flops_per_token'],reference+extra)
                    self.assertEqual(result['fwd_flops_per_token'],sum(result['breakdown'].values()))
        self.assertEqual(legacy_train_flops_per_token(config(),8192),28968720)
        self.assertEqual(count(config(),8192)['train_flops_per_token'],27780096)

    def test_fla_inverse_has_non_gemm_forward_and_full_gemm_backward(self):
        l=fla_chunk_gemm_ledger(64,64)
        self.assertFalse(l['complete_operator_count'])
        self.assertEqual(l['forward_gemms']['kkt_ten_lower_blocks'],10*2*16*16*64)
        self.assertEqual(l['forward_gemms']['solve_sixteen_block_merge_gemms'],16*2*16**3)
        self.assertEqual(l['backward_gemms_excluding_activation_recompute']['inverse_gradient_two_full_gemms'],4*64**3)
        self.assertEqual(l['forward_gemms_per_token'],64512)
        self.assertEqual(l['train_gemms_per_token_excluding_activation_recompute'],211968)
        self.assertNotEqual(l['train_gemms_per_token_excluding_activation_recompute'],3*l['forward_gemms_per_token'])
        with self.assertRaises(ValueError):fla_chunk_gemm_ledger(64,64,32)

    def test_resume_rejects_old_accounting_even_with_fixed_token_budget(self):
        current=count(config(),8192)
        previous=dict(tokens=1_000_000,flops_per_token=legacy_train_flops_per_token(config(),8192))
        with self.assertRaisesRegex(ValueError,'FLOPs accounting changed'):
            validate_resume_accounting(previous,current)
        previous['flops_per_token']=current['train_flops_per_token']
        validate_resume_accounting(previous,current)
        previous['flops_accounting']='different_convention'
        with self.assertRaisesRegex(ValueError,'FLOPs accounting changed'):
            validate_resume_accounting(previous,current)


class Q38IndexerAccounting(unittest.TestCase):
    def test_detached_projection_accounting_matches_autograd_gemms(self):
        import torch
        from torch.profiler import ProfilerActivity, profile
        from q38.models.qwen3_8_moe.modeling_qwen3_8_moe import Indexer

        c = config()
        seq = 64
        indexer = Indexer(c['hidden_size'], 4, 64)
        x = torch.randn(1, seq, c['hidden_size'], requires_grad=True)
        cos, sin = torch.ones(1, seq, 16), torch.zeros(1, seq, 16)
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as trace:
            indexer(x.detach(), cos, sin).square().mean().backward()
        # Linear projections use mm; indexer score/aggregation use bmm.
        projection_train = sum(e.flops for e in trace.key_averages() if e.key == 'aten::mm') // seq
        self.assertIsNone(x.grad)
        for projection in (indexer.wq, indexer.wk, indexer.ww):
            self.assertIsNotNone(projection.weight.grad)
        for mtp in (False, True):
            with self.subTest(mtp=mtp):
                result = count_q38(dict(c, mtp=mtp), seq)
                projection_fwd = result['breakdown']['indexer_projections']
                self.assertEqual(projection_train, 2 * projection_fwd)
                self.assertEqual(result['train_flops_per_token'],
                                 3 * (result['fwd_flops_per_token'] - projection_fwd)
                                 + projection_train + result['distill_target_flops'])
                # The core continues to use the pinned Megatron convention.
                self.assertEqual(result['breakdown']['gdn_core'], count(c, seq)['breakdown']['gdn_core'])

    def test_resume_rejects_previous_q38_accounting(self):
        current = count_q38(config(), 4096)
        previous = dict(flops_accounting='q38_sparse_mhc_ngram_mtp_v1',
                        flops_per_token=current['train_flops_per_token']
                        + current['breakdown']['indexer_projections'])
        with self.assertRaisesRegex(ValueError, 'FLOPs accounting changed'):
            validate_resume_accounting(previous, current)
        validate_resume_accounting(dict(flops_accounting=current['accounting_method'],
                                       flops_per_token=current['train_flops_per_token']), current)


if __name__=='__main__':
    unittest.main()
