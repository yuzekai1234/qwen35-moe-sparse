#!/usr/bin/env python3
"""Compare q35 accounting with the EXACT Megatron revision linked in the contest PDF.

Read-only with respect to training configurations and checkpoints. Usage:
  .venv_q35/bin/python scripts/audit_flops.py \
    --reference-source /tmp/qwen35-megatron-b92c3ed-training.py \
    --output audits/flops_audit.json

Download REFERENCE_URL first if the source is not available. Its SHA256 is checked
before extracting/executing ONLY num_floating_point_operations via AST. The three
injected predicates select its standard Transformer + GDN branch; no Megatron
package, GPU, or distributed initialization is needed.

The FLA ledger is a separate, explicitly PARTIAL logical GEMM ledger, NOT an
official FLOPs convention or a claim of accuracy within 1%. Pointwise operations,
reductions, normalization, activation functions and the vector part of the solve
are not included. Tensor-core padding and repeated tile-local work are not counted.
Activation recomputations are listed separately, following the contest exclusion.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q35.flops import count

REFERENCE_COMMIT = 'b92c3ed70029005fd5769d2cf4c65bc3d061a74b'
REFERENCE_URL = f'https://raw.githubusercontent.com/NVIDIA/Megatron-LM/{REFERENCE_COMMIT}/megatron/training/training.py'
REFERENCE_SHA256 = '370d7844ccc0b36e60bc7f8238b84671ed488c1d2b5c02b5e2f6c32deef8d177'


def load_reference(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError('Megatron source does not match the contest-pinned revision')
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'num_floating_point_operations')
    assert (node.lineno, node.end_lineno) == (821, 1407)
    namespace = {
        'is_hybrid_model': lambda args: False,
        'is_linear_attention_variant': lambda variant: variant == 'gated_delta_net',
        'is_gated_delta_net_variant': lambda variant: variant == 'gated_delta_net',
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['num_floating_point_operations']


def reference_args(c, seq):
    """Map the actual Qwen graph to the pinned Transformer's supported fields."""
    return SimpleNamespace(
        seq_length=seq, hidden_size=c['hidden_size'], num_layers=c['num_hidden_layers'],
        num_attention_heads=c['num_attention_heads'], num_query_groups=c['num_key_value_heads'],
        kv_channels=c['head_dim'], group_query_attention=True, attention_output_gate=True,
        ffn_hidden_size=c['moe_intermediate_size'], moe_ffn_hidden_size=c['moe_intermediate_size'],
        moe_shared_expert_intermediate_size=c['shared_expert_intermediate_size'],
        num_experts=c['num_experts'], moe_router_topk=c['num_experts_per_tok'],
        moe_layer_freq=1, moe_latent_size=None, mtp_num_layers=None, swiglu=True,
        multi_latent_attention=False, padded_vocab_size=c['vocab_size'],
        experimental_attention_variant='gated_delta_net',
        linear_attention_freq=[int(t == 'linear_attention') for t in c['layer_types']],
        linear_key_head_dim=c['linear_key_head_dim'], linear_value_head_dim=c['linear_value_head_dim'],
        linear_num_key_heads=c['linear_num_key_heads'], linear_num_value_heads=c['linear_num_value_heads'],
        linear_conv_kernel_dim=c['linear_conv_kernel_dim'])


def reference_forward_terms(c, seq):
    """Independent expansion, validated against the unmodified pinned function."""
    h = c['hidden_size']
    full = c['layer_types'].count('full_attention')
    lin = c['layer_types'].count('linear_attention')
    q = c['num_attention_heads'] * c['head_dim']
    kv = c['num_key_value_heads'] * c['head_dim']
    nk, nv = c['linear_num_key_heads'], c['linear_num_value_heads']
    dk, dv = c['linear_key_head_dim'], c['linear_value_head_dim']
    qk_dim, v_dim = nk * dk, nv * dv
    return {
        'full_projections_including_output_gate': full * 2 * h * (3 * q + 2 * kv),
        'full_causal_core': full * 2 * seq * q,
        'gdn_projections': lin * 2 * h * (2 * qk_dim + 3 * v_dim + 2 * nv),
        'gdn_conv': lin * 2 * c['linear_conv_kernel_dim'] * (2 * qk_dim + v_dim),
        # This is EXACTLY what the reference uses, including its dv**2 assumption.
        'gdn_recurrent_core_reference': lin * 8 * nv * dv**2,
        'routed_and_shared_swiglu': c['num_hidden_layers'] * 6 * h * (
            c['num_experts_per_tok'] * c['moe_intermediate_size'] + c['shared_expert_intermediate_size']),
        'logits': 2 * h * c['vocab_size'],
    }


def fla_chunk_gemm_ledger(dk, dv, chunk=64):
    """Per value head, per chunk; logical GEMMs in local FLA 0.5.2 Triton path.

    Restricted to its 64-token fused solve. KKT computes ten 16x16 blocks;
    solving/merging those four diagonal blocks uses sixteen 16x16x16 GEMMs.
    Forward substitution within the diagonal blocks is vector arithmetic and
    is explicitly excluded here, not silently assigned zero total FLOPs.
    The inverse gradient uses two FULL 64x64x64 GEMMs, so it is not 2x the
    forward block-merge term. No blanket '3 x forward' is used for this core.
    """
    if chunk != 64:
        raise ValueError('Only the audited 64-token fused solve is supported')
    C, BC = chunk, 16
    fwd = {
        'kkt_ten_lower_blocks': 10 * 2 * BC * BC * dk,
        'solve_sixteen_block_merge_gemms': 16 * 2 * BC**3,
        'wy_w_and_u': 2 * C * C * (dk + dv),
        'state_w_times_h_and_kt_times_vnew': 4 * C * dk * dv,
        'output_q_times_h': 2 * C * dk * dv,
        'output_qkt_and_attn_times_vnew': 2 * C * C * (dk + dv),
    }
    bwd = {
        'local_dv': 2 * C * C * dv,
        'state_dhu': 6 * C * dk * dv,
        'output_dqkwg': 6 * C * dk * dv + 4 * C * C * dk + 2 * C * C * dv,
        'wy_w_and_u_gradients': 4 * C * C * (dk + dv),
        'inverse_gradient_two_full_gemms': 4 * C**3,
        'kkt_input_gradients': 4 * C * C * dk,
    }
    recompute = {
        'wy_w_and_u': 2 * C * C * (dk + dv),
        'state_h_and_vnew': 4 * C * dk * dv,
        'qkt_for_local_dv': 2 * C * C * dk,
        'kkt_for_gate_gradient': 2 * C * C * dk,
    }
    return dict(forward_gemms=fwd, backward_gemms_excluding_activation_recompute=bwd,
                excluded_activation_recompute_gemms=recompute,
                forward_gemms_per_token=sum(fwd.values()) / C,
                train_gemms_per_token_excluding_activation_recompute=(sum(fwd.values()) + sum(bwd.values())) / C,
                complete_operator_count=False)


def check_local_fla_solve(path):
    """Check the nontrivial solve GEMM multiplicities against installed source."""
    tree = ast.parse(Path(path).read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'chunk_gated_delta_rule_fwd_kkt_solve_kernel')
    targets = {'b_Ai10': 2, 'b_Ai21': 2, 'b_Ai32': 2,
               'b_Ai20': 3, 'b_Ai31': 3, 'b_Ai30': 4}
    found = {}
    for node in ast.walk(kernel):
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in targets:
            found[name] = sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                              and isinstance(n.func.value, ast.Name) and n.func.value.id == 'tl'
                              and n.func.attr == 'dot' for n in ast.walk(node))
    if found != targets:
        raise ValueError(f'Installed FLA solve differs from the audited graph: {found}')
    return found


def legacy_train_flops_per_token(c, seq):
    """Pre-calibration counter, only for auditing archived experiment numbers."""
    current = count(c, seq)
    lin = c['layer_types'].count('linear_attention')
    nv = c['linear_num_value_heads']
    dk, dv = c['linear_key_head_dim'], c['linear_value_head_dim']
    chunk_core = lin * nv * (6*dk*dv + 6*64*dk + 4*64*dv)
    return 3 * (2*current['active_nonvocab_params'] + chunk_core
                + current['breakdown']['attn_score'] + current['breakdown']['logits'])


def audit(c, seq, reference):
    if c['layer_types'] != c['layer_types'][:c['num_hidden_layers']]:
        raise ValueError('Layer list length does not match layer count')
    current = count(c, seq)
    terms = reference_forward_terms(c, seq)
    ref = reference(reference_args(c, seq), 1) / seq
    assert ref == 3 * sum(terms.values()), 'Independent expansion differs from pinned function'
    L, h = c['num_hidden_layers'], c['hidden_size']
    lin, full = c['layer_types'].count('linear_attention'), c['layer_types'].count('full_attention')
    # The reference omits these two real, trainable linear projections.
    router_and_shared_gate = 6 * L * h * (c['num_experts'] + 1)
    # Parameters which the legacy implementation treats as dense GEMM weights,
    # though their operations actually require individual non-GEMM accounting.
    misclassified_params = (lin * (2 * c['linear_num_value_heads'] + c['linear_value_head_dim'])
                            + full * 2 * c['head_dim'] + L * 2 * h + h)
    dk, dv = c['linear_key_head_dim'], c['linear_value_head_dim']
    legacy_core = 3 * lin * c['linear_num_value_heads'] * (6*dk*dv + 6*64*dk + 4*64*dv)
    reference_core = 3 * terms['gdn_recurrent_core_reference']
    old = legacy_train_flops_per_token(c, seq)
    assert current['breakdown']['attn_score'] == terms['full_causal_core']
    assert current['megatron_reference_train_flops_per_token'] == ref
    assert current['train_flops_per_token'] == ref + router_and_shared_gate
    assert old - ref == (
        legacy_core - reference_core + router_and_shared_gate + 6 * misclassified_params)
    out = dict(legacy_train_flops_per_token=old,
               current_train_flops_per_token=current['train_flops_per_token'],
               current_accounting_method=current['accounting_method'],
               current_over_megatron_fraction=current['train_flops_per_token']/ref-1,
               megatron_train_flops_per_token=ref,
               megatron_forward_terms=terms,
               reference_plus_router_and_shared_gate_train_flops_per_token=ref + router_and_shared_gate,
               difference_train_flops_per_token=dict(
                   gdn_core=legacy_core-reference_core,
                   router_and_shared_gate=router_and_shared_gate,
                   non_gemm_params_treated_as_gemms=6*misclassified_params),
               legacy_over_megatron_fraction=old/ref-1,
               legacy_over_reference_plus_router_gate_fraction=old/(ref+router_and_shared_gate)-1,
               same_tokens_megatron_vs_legacy_budget_fraction=ref/old-1,
               complete_actual_graph_verified_within_one_percent=False)
    if seq % 64 == 0 and c['linear_key_head_dim'] == c['linear_value_head_dim'] == 64:
        ledger = fla_chunk_gemm_ledger(64, 64)
        core = lin * c['linear_num_value_heads'] * ledger['train_gemms_per_token_excluding_activation_recompute']
        # Explicitly partial: replace legacy core and remove the non-GEMM proxy,
        # retain all actual projection GEMMs (including router and shared gate).
        partial = old - legacy_core - 6*misclassified_params + core
        out['fla_partial_gemm_plus_conv_train_flops_per_token'] = partial
        out['fla_partial_vs_legacy_fraction'] = partial/old-1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seq', type=int, default=8192)
    ap.add_argument('--vocab', type=int, default=10112)
    a = ap.parse_args()
    reference = load_reference(a.reference_source)
    from q35.presets import model_config
    from scripts.make_isoflops_configs import shape_for
    rows = []
    for path in sorted((ROOT/'configs/isoflops').glob('*/*/config.json')):
        p = json.loads(path.read_text())
        seq = p['train']['seq']
        c = model_config(p['model'], a.vocab, seq).to_dict()
        assert p['accounting']['train_flops_per_token'] == count(c,seq)['train_flops_per_token'], f'Stale accounting: {path}'
        row = dict(config=str(path.relative_to(ROOT)), compute=p['compute'],
                   config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   shape=f'h{c["hidden_size"]}_L{c["num_hidden_layers"]}', **audit(c, seq, reference))
        D = p['accounting']['train_tokens']
        row['configured_train_tokens'] = D
        row['current_accounting_budget_fraction'] = D*row['current_train_flops_per_token']/p['compute']-1
        row['megatron_reaccounted_budget_fraction'] = D*row['megatron_train_flops_per_token']/p['compute']-1
        row['reference_plus_router_gate_reaccounted_budget_fraction'] = (
            D*row['reference_plus_router_and_shared_gate_train_flops_per_token']/p['compute']-1)
        rows.append(row)
    family = []
    for hidden in range(128, 1601, 64):
        c = model_config(shape_for(hidden), a.vocab, a.seq).to_dict()
        family.append(dict(shape=f'h{hidden}_L{c["num_hidden_layers"]}', **audit(c, a.seq, reference)))
    site = ROOT/'.venv_q35/lib/python3.11/site-packages'
    sources = {}
    for rel in ['fla/ops/gated_delta_rule/chunk.py', 'fla/ops/gated_delta_rule/chunk_fwd.py',
                'fla/ops/gated_delta_rule/wy_fast.py', 'fla/ops/common/chunk_delta_h.py',
                'fla/ops/common/chunk_o.py', 'fla/modules/l2norm.py']:
        path = site/rel
        if path.exists():
            sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    solve_counts = check_local_fla_solve(site/'fla/ops/gated_delta_rule/chunk_fwd.py')
    report = dict(reference=dict(commit=REFERENCE_COMMIT, url=REFERENCE_URL, sha256=REFERENCE_SHA256,
                                  function_lines=[821,1407], branch='Transformer with GDN and attention_output_gate=True'),
                  scope=dict(seq=a.seq, vocab=a.vocab, physical_kernel_flops=False,
                             actual_graph_within_one_percent_certified=False),
                  limitations=[
                      'Contest +/-1% is budget tolerance, not a blanket permission for uncertain operator estimates.',
                      'Pinned reference has a recurrent GDN core estimate; FLA chunk uses additional WY/solve operations.',
                      'Reference-plus-router/gate is a separate adjusted ledger, not the unmodified official formula.',
                      'Partial FLA GEMMs omit pointwise/reduction/normalization/SFU and vector solve operations.',
                      'Partial FLA GEMMs exclude activation recomputation and tensor-core tiling/padding overhead.',
                      'Audit command is read-only; token budgets are recalculated separately by the config generator.',
                  ],
                  findings=[
                      'Full attention causal factor is already correct; the old Chinese comment was incorrect.',
                      'Legacy GDN core is 16*d^2 per value head at C=dk=dv=64; pinned reference is 8*d^2.',
                      'Pinned reference includes attention output gate when enabled, but omits MoE router and shared gate.',
                      'Multiplying all active parameters by six misclassifies norm and gate scalar parameters as GEMM weights.',
                      'FLA uses WY construction and triangular solve; its forward and backward require separate accounting.',
                      'Current counter adopts the pinned recurrent GDN convention, plus actual router/shared gate projections.',
                      'Current reference component is checked for exact equality against the pinned function.',
                      'This selected model accounting convention does not claim to count physical FLA instructions.',
                  ],
                  installed_versions={name:importlib.metadata.version(name)
                                      for name in ['fla-core','flash-linear-attention','torch','triton']},
                  local_source_sha256=sources,
                  local_fla_solve_merge_gemms_checked=solve_counts,
                  fla_64x64_chunk64_partial_ledger=fla_chunk_gemm_ledger(64,64),
                  current_configurations=rows, canonical_family=family)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    for row in family:
        print(f"{row['shape']:12} current={row['current_train_flops_per_token']:12,d} "
              f"Megatron={row['megatron_train_flops_per_token']:12,.0f} "
              f"extra router/gate={row['current_over_megatron_fraction']:+.3%}")
    print(f'Checked {len(rows)} on-disk configs and {len(family)} canonical shapes; report: {a.output}')


if __name__ == '__main__':
    main()
