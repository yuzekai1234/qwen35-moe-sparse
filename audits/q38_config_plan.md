# Q38 C1–C4 configuration plan

Read `2030以赛代评.pdf` (SHA256 recorded in JSON): page 3 sets budgets 3e17/1e18/3e18/1e19, permits ±1% whole-step error, requires one architecture/scaling rule and non-vocabulary Na/N≥10%; page 4 counts repeated/iterative graph operations. Pages 5–6 specify full-vocabulary prefill timing and sm90, single-node 8-GPU evaluation. No fixed sequence length or training batch is specified.

The user clarified that 4096 replaces the old 8192 **sequence length**. Global batch remains a separate token count. GZY10K has **10112** actual vocabulary entries; the verified 16B-token pool and frozen validation partitions are reused.

## Q35 evidence and fit

Use the 19 completed unmerged Q35 runs in `runs/{1e16,3e16,1e17,3e17}`. Fit each budget’s measured A-evaluation bpb against log10(non-vocabulary active parameters) with a convex quadratic. Interpolate log10 D at that fitted minimum, then fit power laws against actual training FLOPs. This follows the existing IsoFLOPs analysis method; validation CE is also preserved in the JSON.

| Q35 budget | Measured best | bpb | Measured Na | Fitted Na optimum | Fitted D optimum | Fit RMSE |
|---|---|---:|---:|---:|---:|---:|
| 1e16 | h192_L4 | 1.373127 | 2.30M | 1.65M | 0.270B | 0.01046 |
| 3e16 | h256_L5 | 1.278032 | 5.10M | 3.78M | 0.500B | 0.01110 |
| 1e17 | h384_L8 | 1.201701 | 18.09M | 10.63M | 0.759B | 0.00583 |
| 3e17 | h448_L9 | 1.112111 | 27.72M | 22.39M | 1.236B | 0.00485 |

`log10 Na_prior = 0.776705 log10 C -6.207939`
`log10 D_prior = 0.436074 log10 C +1.476416`

All fitted per-budget vertices lie inside the measured grids. At 3e17, h448_L9 and h512_L10 differ by only **0.000114 bpb**; the exact optimum is weakly identified. Leaving out one whole budget gives active-parameter exponents **0.767–0.809** and C4 Na predictions **329–433M**. These are sensitivity ranges, not statistical confidence intervals. C2–C4 exceed the largest measured budget by 3.3×/10×/33.3×; Q38 is also a different architecture and uses seq4096, so this fit provides a starting point only.

## One architecture rule

For width h on the 64-wide grid: L=max(4,round(h/51.2)); full attention query heads=2·max(1,round(1.2h/64)), KV heads=2; GDN key heads=max(2,round(0.8h/64)), value heads=3·key heads. Head dimensions stay 64. All budgets use **64 experts/top-5**, routed/shared widths rounded h/4, QSA 64-token blocks/top-8, four-branch mHC, the same N-gram injection/hash/gate, Muon+AdamW and one training MTP block. Each N-gram table has rows≈0.205·main_nonvocab_params/h rounded to 1024. No module type or routing mechanism changes with budget.

The denominator includes **all N-gram rows and the training-only MTP block**. The numerator excludes MTP and includes only the two accessed N-gram rows, using the repository’s evaluation main-head convention. This is conservative relative to counting MTP as active and yields >16% in all four presets.

The Q35 N and D power laws cannot both be imposed exactly on the heavier Q38 graph. For every candidate, reserve 0.5% compute and recompute D=floor(0.995C/(FLOPs_per_token·global_batch_tokens))·global_batch_tokens and minimize `log(Na/Na_prior)^2 + log(D/D_prior)^2`. This keeps an equal relative-error tradeoff between model and data; it is a planning heuristic, not a measured Q38 optimum.

| Preset | h × L | Na | Total non-vocab | Na/N | D | D/Na | Global batch tokens (sequences) | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q38_c1 | 448 × 9 | 27.12M | 163.22M | 16.61% | 1.023B | 37.7 | 131,072 (32) | 7,806 |
| q38_c2 | 576 × 11 | 52.25M | 320.79M | 16.29% | 2.046B | 39.2 | 262,144 (64) | 7,804 |
| q38_c3 | 768 × 15 | 129.41M | 770.26M | 16.80% | 2.822B | 21.8 | 262,144 (64) | 10,764 |
| q38_c4 | 1024 × 20 | 300.24M | 1794.30M | 16.73% | 4.355B | 14.5 | 524,288 (128) | 8,307 |

C4 selects h1024 instead of the Na-nearest h1088: the latter allocates too little data under the Q38 MTP/indexer cost. Detailed target deviations, runner-up widths, full parameter components and whole-step budget shortfalls are in `q38_config_plan.json`.

## FLOPs limits

The implemented accounting includes QSA selected-block attention, indexer projections/dense score, detached KL target, mHC matrix projections/mixing, N-gram gates, and MTP. Dense warmup and some mHC pointwise/iterative operations were omitted by the inherited convention. Presets therefore reserve **0.5%** of official C and allocate explicit `train.tokens` from **0.995C**; this centers the known omitted-op allowance rather than spending all counted FLOPs at the budget ceiling. The report bounds the dense-vs-sparse warmup difference and deliberately overestimates mHC Sinkhorn/normalization/nonlinear operations; the combined bound below is **not** a bound on every possible unlisted operation or an organizer compliance certification.

| Preset | Dense warmup extra / counted | mHC unlisted-op bound / counted | Combined | Counted whole-step shortfall |
|---|---:|---:|---:|---:|
| c1 | 0.134% | 0.782% | 0.916% | 0.5087% |
| c2 | 0.111% | 0.712% | 0.823% | 0.5032% |
| c3 | 0.098% | 0.577% | 0.675% | 0.5037% |
| c4 | 0.102% | 0.464% | 0.566% | 0.5022% |

Reproduce without training:

```bash
.venv_q35/bin/python scripts/make_q38_configs.py --check
.venv_q35/bin/python -m q38.train --preset q38_c1 --dry-run
GPUS=0 bash scripts/train.sh q38_c1
```

`train.tokens` is explicit so training preserves the reserve. After changing a model shape, regenerate the presets; `--tokens 0` recomputes from the full official C in the training CLI and does not retain this reserve. The `q38.flops` preset helper does retain the configured reserve when changing vocabulary or sequence length.

`configs/q35_isoflops/` remains byte-for-byte untouched. These presets have not been trained; throughput profiling and quality validation are separate tasks.
