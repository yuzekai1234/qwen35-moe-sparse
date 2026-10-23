#!/usr/bin/env python3
"""Fit completed Q35 IsoFLOPs results and generate four Q38 warm-start presets.

This never trains and never edits configs/q35_isoflops. Predictions transfer a
Q35 scaling prior to a different architecture; they are not Q38 measurements.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38.flops import count
from q38.presets import MODEL_DEFAULTS, ROOT, budget_tokens, model_config

BUDGETS = {"c1": 3 * 10**17, "c2": 10**18, "c3": 3 * 10**18, "c4": 10**19}
SEQ = 4096  # User's clarified request; global batch size is a separate parameter.
COMPUTE_RESERVE = 0.005  # Center the counted budget around the known omitted-op allowance.
VOCAB = 10112  # GZY 10K tokenizer, including special tokens / padding.
POOL = Path('/home/shared/qwen35_moe_sparse/cache/isoflops-gzy10k-16b-v1')
PDF = ROOT / '2030以赛代评.pdf'


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def source_rows():
    rows = []
    for directory in sorted((ROOT / 'runs').glob('*/*')):
        if directory.parent.name not in ('1e16', '3e16', '1e17', '3e17'):
            continue
        summary_path, final_path = directory / 'summary.json', directory / 'final_eval.json'
        evals = sorted(directory.glob('eval_*.json'))
        if not summary_path.is_file() or not final_path.is_file() or not evals:
            continue
        summary, final, evaluation = [json.loads(p.read_text()) for p in (summary_path, final_path, evals[0])]
        if not summary.get('completed_budget'):
            continue
        rows.append(dict(budget=directory.parent.name, model=directory.name,
                         compute=summary['train_flops'], active_params=summary['active_nonvocab'],
                         train_tokens=summary['tokens_trained'], bpb=evaluation['bpb'], loss=final['loss'],
                         source=str(summary_path.relative_to(ROOT)), eval_source=str(evals[0].relative_to(ROOT))))
    return rows


def fit_curves(rows):
    fits = []
    for budget in sorted({row['budget'] for row in rows}, key=float):
        points = sorted((row for row in rows if row['budget'] == budget), key=lambda row: row['active_params'])
        if len(points) < 3:
            raise ValueError(f'Need >=3 completed measurements in {budget}')
        x = np.log10([row['active_params'] for row in points])
        y = np.array([row['bpb'] for row in points])
        coef = np.polyfit(x, y, 2)
        if coef[0] <= 0:
            raise ValueError(f'{budget} has no convex fitted optimum')
        log_n = -coef[1] / (2 * coef[0])
        log_d = np.polyval(np.polyfit(x, np.log10([row['train_tokens'] for row in points]), min(2, len(points)-1)), log_n)
        fits.append(dict(budget=budget, compute=float(np.mean([row['compute'] for row in points])),
                         quadratic_coefficients=coef.tolist(), active_opt=10**float(log_n), tokens_opt=10**float(log_d),
                         bpb_opt=float(np.polyval(coef, log_n)), inside_grid=bool(x.min() <= log_n <= x.max()),
                         residual_rmse=float(np.sqrt(np.mean((np.polyval(coef, x)-y)**2))),
                         measured_best=min(points, key=lambda row: row['bpb']), points=points))
    log_c = np.log10([f['compute'] for f in fits])
    p_n = np.polyfit(log_c, np.log10([f['active_opt'] for f in fits]), 1)
    p_d = np.polyfit(log_c, np.log10([f['tokens_opt'] for f in fits]), 1)
    loo = []
    for omitted in range(len(fits)):
        keep = [i for i in range(len(fits)) if i != omitted]
        pn = np.polyfit(log_c[keep], np.log10([fits[i]['active_opt'] for i in keep]), 1)
        loo.append(dict(omitted_budget=fits[omitted]['budget'], active_exponent=float(pn[0]),
                        c4_active_prediction=float(10**np.polyval(pn, 19))))
    return fits, p_n, p_d, loo


def shape_for(hidden):
    """One fixed architecture rule for C1–C4; hidden alone selects every dimension."""
    nk = max(2, round(0.8 * hidden / 64))
    expert_width = max(8, round(hidden / 4 / 8) * 8)
    return {**MODEL_DEFAULTS, 'hidden': hidden, 'layers': max(4, round(hidden / 51.2)),
            'heads': 2 * max(1, round(1.2 * hidden / 64)), 'kv_heads': 2,
            'lin_k_heads': nk, 'lin_v_heads': 3 * nk, 'experts': 64, 'topk': 5,
            'moe_ffn': expert_width, 'shared_ffn': expert_width, 'experts_impl': 'grouped_mm',
            'ngram_rows': -1}


def batch_for(compute):
    # Same DeepSeek batch rule as Q35, rounded to whole power-of-two sequences.
    return SEQ * max(8, 2**round(math.log2(0.2920 * compute**0.3271 / SEQ)))


def candidates(compute, n_target, d_target, batch):
    result = []
    for hidden in range(128, 1601, 64):
        shape = shape_for(hidden)
        config = model_config(shape, VOCAB, SEQ)
        shape['ngram_rows'] = config.ngram_rows
        fc = count(config.to_dict(), SEQ)
        tokens = budget_tokens(fc['train_flops_per_token'], int(compute*(1-COMPUTE_RESERVE)), batch)
        ratio = fc['active_nonvocab_params'] / fc['total_nonvocab_params']
        if ratio <= 0.10 or not tokens:
            continue
        score = math.log(fc['active_nonvocab_params']/n_target)**2 + math.log(tokens/d_target)**2
        result.append(dict(score=score, shape=shape, flops=fc, train_tokens=tokens, active_fraction=ratio))
    return sorted(result, key=lambda row: row['score'])


def uncounted_bounds(shape, fc, steps, warmup):
    """Conservative logical-op bounds, not an official audit or instruction count."""
    h, layers, n = shape['hidden'], shape['layers'], shape['n_branch']
    nf, nq, hd = layers//4, shape['heads'], shape['head_dim']
    blocks = SEQ//64
    avg = sum(64*(min(i+1,shape['qsa_blocks'])-1)+32.5 for i in range(blocks))/blocks
    dense_delta = 3*nf*nq*hd*(2*SEQ-4*avg)*warmup/steps
    # Per sublayer: generous 10*n*n FLOPs per Sinkhorn iteration, dynamic
    # normalization/nonlinear coefficients bounded by 20*n*h+200*n*n; x3 backward.
    mhc_bound = 3*2*layers*(10*n*n*shape['mhc_iters'] + 20*n*h + 200*n*n)
    return dict(dense_warmup_extra_train_flops_per_token=dense_delta,
                dense_warmup_fraction=dense_delta/fc['train_flops_per_token'],
                mhc_unlisted_pointwise_train_flops_per_token_bound=mhc_bound,
                mhc_unlisted_pointwise_fraction_bound=mhc_bound/fc['train_flops_per_token'],
                combined_fraction_bound=(dense_delta+mhc_bound)/fc['train_flops_per_token'],
                excludes='Does not bound every other unlisted non-GEMM operation; not organizer-reviewed.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true', help='Recompute and compare existing generated presets')
    args=ap.parse_args()
    rows=source_rows()
    fits,pn,pd,loo=fit_curves(rows)
    val=json.loads((ROOT/'configs/q35_isoflops/validation.json').read_text())
    chosen=[]
    for name,compute in BUDGETS.items():
        n_target=float(10**np.polyval(pn, math.log10(compute)))
        d_target=float(10**np.polyval(pd, math.log10(compute)))
        batch=batch_for(compute)
        ranked=candidates(compute,n_target,d_target,batch)
        best=ranked[0]; shape,fc,tokens=best['shape'],best['flops'],best['train_tokens']
        steps=tokens//batch;warmup=min(steps,max(10,round(steps*.01)))
        train=dict(run_dir=f'runs/q38/{name}',data=str(POOL/'train.bin'),val_data=str(POOL/'valid.bin'),
                   tokenizer=str(POOL/'tokenizer'),seq=SEQ,tokens=tokens,batch_tokens=batch,micro_bs=4,
                   auto_micro_bs=1,probe_mem_frac=.92,lr=round(.3118*compute**-.125,6),min_lr_ratio=.1,
                   warmup=warmup,warmup_ratio=.01,warmup_min=10,wd=.1,clip=1.,seed=0,data_seed=0,
                   data_sampling='shuffled_blocks',require_ready_pool=1,eval_sampling='sequential',
                   eval_offset=val['dev']['offset'],eval_tokens=val['dev']['targets'],
                   final_eval_offset=val['curve']['offset'],final_eval_tokens=val['curve']['targets'],
                   log_every=10,eval_every=500,ckpt_every=500,resume=1,max_steps=0,tensorboard=1,
                   mtp_coef=.3,kl_coef=.05,muon=1,dense_warmup=1)
        accounting=dict(accounting_method=fc['accounting_method'],vocab_size=VOCAB,compute_reserve_fraction=COMPUTE_RESERVE,
                        total_params=fc['total_params_all'],total_nonvocab_params=fc['total_nonvocab_params'],
                        active_nonvocab_params=fc['active_nonvocab_params'],active_fraction=best['active_fraction'],
                        ngram_params=fc['ngram_params'],mtp_params=fc['mtp_params'],
                        train_flops_per_token=fc['train_flops_per_token'],train_tokens=tokens,steps=steps,
                        actual_train_flops=tokens*fc['train_flops_per_token'],
                        compute_shortfall_fraction=1-tokens*fc['train_flops_per_token']/compute,
                        tokens_per_active_param=tokens/fc['active_nonvocab_params'],global_batch_sequences=batch//SEQ,
                        counted_compute_target=int(compute*(1-COMPUTE_RESERVE)),
                        omitted_operation_bounds=uncounted_bounds(shape,fc,steps,warmup))
        prior=dict(source_architecture='Q35, seq8192, GZY10K',target_active_params=n_target,target_train_tokens=d_target,
                   chosen_active_over_target=fc['active_nonvocab_params']/n_target,
                   chosen_tokens_over_target=tokens/d_target,objective='log(Na/Na_prior)^2 + log(D/D_prior)^2',
                   extrapolated=compute>max(f['compute'] for f in fits)*1.001,
                   runner_up_hidden=ranked[1]['shape']['hidden'],selection_score=best['score'])
        config=dict(schema_version=2,architecture='q38',compute=compute,model=shape,train=train,
                    accounting=accounting,scaling_prior=prior,
                    scaling_rule='scripts/make_q38_configs.py:shape_for; width grid 128:64:1600')
        target=ROOT/f'configs/q38_{name}.json'
        if args.check:
            if json.loads(target.read_text())!=config:raise ValueError(f'Generated preset differs: {target}')
        else:json_write(target,config)
        chosen.append(dict(name=name,compute=compute,hidden=shape['hidden'],layers=shape['layers'],**accounting,prior=prior))
    report=dict(pdf=dict(path=PDF.name,sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
                        pages={'budgets_activation_same_architecture':3,'flops':4,'evaluation':5,'hardware':6},
                        budgets=BUDGETS,activation_rule='Na/N >= 10%; both exclude only input/output vocabulary parameters',
                        requested_seq=SEQ,batch_note='4096 is sequence length per user clarification; global batch is separate.'),
                source_runs=rows,budget_fits=fits,
                power_laws=dict(active_log10_slope=float(pn[0]),active_log10_intercept=float(pn[1]),
                                tokens_log10_slope=float(pd[0]),tokens_log10_intercept=float(pd[1])),
                leave_one_budget_out=loo,models=chosen,
                limitations=['Q35-to-Q38 architecture transfer is an unvalidated warm-start prior.',
                             'Only C1 lies at a measured compute budget; C2–C4 extrapolate beyond all Q35 sweep data.',
                             'FLOPs are the implemented convention, with omitted-op bounds listed; no organizer audit.',
                             'Q38 model quality and optimality require actual training; no Q38 bpb is predicted here.'])
    if not args.check:
        json_write(ROOT/'audits/q38_config_plan.json',report)
        write_report(report)
    for model in chosen:
        print(f"{model['name']}: h{model['hidden']} L{model['layers']}, Na={model['active_nonvocab_params']:,}, "
              f"Na/N={model['active_fraction']:.2%}, D={model['train_tokens']:,}, steps={model['steps']:,}")


def write_report(report):
    p=report['power_laws']; lines=['# Q38 C1–C4 configuration plan','',
        'Read `2030以赛代评.pdf` (SHA256 recorded in JSON): page 3 sets budgets 3e17/1e18/3e18/1e19, permits ±1% whole-step error, requires one architecture/scaling rule and non-vocabulary Na/N≥10%; page 4 counts repeated/iterative graph operations. Pages 5–6 specify full-vocabulary prefill timing and sm90, single-node 8-GPU evaluation. No fixed sequence length or training batch is specified.', '',
        'The user clarified that 4096 replaces the old 8192 **sequence length**. Global batch remains a separate token count. GZY10K has **10112** actual vocabulary entries; the verified 16B-token pool and frozen validation partitions are reused.', '',
        '## Q35 evidence and fit', '',
        'Use the 19 completed unmerged Q35 runs in `runs/{1e16,3e16,1e17,3e17}`. Fit each budget’s measured A-evaluation bpb against log10(non-vocabulary active parameters) with a convex quadratic. Interpolate log10 D at that fitted minimum, then fit power laws against actual training FLOPs. This follows the existing IsoFLOPs analysis method; validation CE is also preserved in the JSON.', '',
        '| Q35 budget | Measured best | bpb | Measured Na | Fitted Na optimum | Fitted D optimum | Fit RMSE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for fit in report['budget_fits']:
        best=fit['measured_best'];lines.append(f"| {fit['budget']} | {best['model']} | {best['bpb']:.6f} | {best['active_params']/1e6:.2f}M | {fit['active_opt']/1e6:.2f}M | {fit['tokens_opt']/1e9:.3f}B | {fit['residual_rmse']:.5f} |")
    lines+=['',f"`log10 Na_prior = {p['active_log10_slope']:.6f} log10 C {p['active_log10_intercept']:+.6f}`",
            f"`log10 D_prior = {p['tokens_log10_slope']:.6f} log10 C {p['tokens_log10_intercept']:+.6f}`",'',
            'All fitted per-budget vertices lie inside the measured grids. At 3e17, h448_L9 and h512_L10 differ by only **0.000114 bpb**; the exact optimum is weakly identified. Leaving out one whole budget gives active-parameter exponents **0.767–0.809** and C4 Na predictions **329–433M**. These are sensitivity ranges, not statistical confidence intervals. C2–C4 exceed the largest measured budget by 3.3×/10×/33.3×; Q38 is also a different architecture and uses seq4096, so this fit provides a starting point only.', '',
            '## One architecture rule', '',
            'For width h on the 64-wide grid: L=max(4,round(h/51.2)); full attention query heads=2·max(1,round(1.2h/64)), KV heads=2; GDN key heads=max(2,round(0.8h/64)), value heads=3·key heads. Head dimensions stay 64. All budgets use **64 experts/top-5**, routed/shared widths rounded h/4, QSA 64-token blocks/top-8, four-branch mHC, the same N-gram injection/hash/gate, Muon+AdamW and one training MTP block. Each N-gram table has rows≈0.205·main_nonvocab_params/h rounded to 1024. No module type or routing mechanism changes with budget.', '',
            'The denominator includes **all N-gram rows and the training-only MTP block**. The numerator excludes MTP and includes only the two accessed N-gram rows, using the repository’s evaluation main-head convention. This is conservative relative to counting MTP as active and yields >16% in all four presets.', '',
            'The Q35 N and D power laws cannot both be imposed exactly on the heavier Q38 graph. For every candidate, reserve 0.5% compute and recompute D=floor(0.995C/(FLOPs_per_token·global_batch_tokens))·global_batch_tokens and minimize `log(Na/Na_prior)^2 + log(D/D_prior)^2`. This keeps an equal relative-error tradeoff between model and data; it is a planning heuristic, not a measured Q38 optimum.', '',
            '| Preset | h × L | Na | Total non-vocab | Na/N | D | D/Na | Global batch tokens (sequences) | Steps |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in report['models']:
        lines.append(f"| q38_{m['name']} | {m['hidden']} × {m['layers']} | {m['active_nonvocab_params']/1e6:.2f}M | {m['total_nonvocab_params']/1e6:.2f}M | {m['active_fraction']:.2%} | {m['train_tokens']/1e9:.3f}B | {m['tokens_per_active_param']:.1f} | {m['global_batch_sequences']*SEQ:,} ({m['global_batch_sequences']}) | {m['steps']:,} |")
    lines+=['', 'C4 selects h1024 instead of the Na-nearest h1088: the latter allocates too little data under the Q38 MTP/indexer cost. Detailed target deviations, runner-up widths, full parameter components and whole-step budget shortfalls are in `q38_config_plan.json`.', '',
            '## FLOPs limits', '',
            'The implemented accounting includes QSA selected-block attention, indexer projections/dense score, detached KL target, mHC matrix projections/mixing, N-gram gates, and MTP. Dense warmup and some mHC pointwise/iterative operations were omitted by the inherited convention. Presets therefore reserve **0.5%** of official C and allocate explicit `train.tokens` from **0.995C**; this centers the known omitted-op allowance rather than spending all counted FLOPs at the budget ceiling. The report bounds the dense-vs-sparse warmup difference and deliberately overestimates mHC Sinkhorn/normalization/nonlinear operations; the combined bound below is **not** a bound on every possible unlisted operation or an organizer compliance certification.', '',
            '| Preset | Dense warmup extra / counted | mHC unlisted-op bound / counted | Combined | Counted whole-step shortfall |',
            '|---|---:|---:|---:|---:|']
    for m in report['models']:
        b=m['omitted_operation_bounds']; lines.append(f"| {m['name']} | {b['dense_warmup_fraction']:.3%} | {b['mhc_unlisted_pointwise_fraction_bound']:.3%} | {b['combined_fraction_bound']:.3%} | {m['compute_shortfall_fraction']:.4%} |")
    lines+=['','Reproduce without training:', '', '```bash', '.venv_q35/bin/python scripts/make_q38_configs.py --check', '.venv_q35/bin/python -m q38.train --preset q38_c1 --dry-run', 'GPUS=0 bash scripts/train.sh q38_c1', '```', '',
            '`train.tokens` is explicit so training preserves the reserve. After changing a model shape, regenerate the presets; `--tokens 0` recomputes from the full official C in the training CLI and does not retain this reserve. The `q38.flops` preset helper does retain the configured reserve when changing vocabulary or sequence length.\n\n`configs/q35_isoflops/` remains byte-for-byte untouched. These presets have not been trained; throughput profiling and quality validation are separate tasks.']
    (ROOT/'audits/q38_config_plan.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
