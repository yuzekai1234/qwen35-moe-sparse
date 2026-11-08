"""Fit four measured BPBs and recompute every team's score under the November 6 rule."""
import argparse
import json
import math
from pathlib import Path
from common import HERE, read, write


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def rerank(board, performance, evaltime, team='SQ2026AAA012562'):
    rows = [dict(row) for row in board]
    target = next(row for row in rows if row['team'] == team)
    target.update(performance=performance, evaltime=evaltime)
    p1, p3 = [quantile([r['performance'] for r in rows], q) for q in (.25, .75)]
    t1, t3 = [quantile([r['evaltime'] for r in rows], q) for q in (.25, .75)]
    if not (0 < p1 < p3 and 0 < t1 < t3):
        raise ValueError('Both metric interquartile ratios must be positive and nondegenerate')
    weight = math.log(p3/p1)/(5*math.log(t3/t1))
    for row in rows:
        row['score'] = row['performance']*row['evaltime']**weight
    rows.sort(key=lambda r: r['score'])
    for row in rows:
        row['rank'] = 1 + sum(round(r['score'], 6) < round(row['score'], 6) for r in rows)
    return dict(lambda_=weight, target=target, ranking=rows,
                warning='Scenario against the saved rounded November 6 board, not a prediction of B standings')


def fit_performance(points):
    import numpy as np
    from scipy.optimize import least_squares
    compute = np.array([p['train_flops'] for p in points], dtype=float)
    bpb = np.array([p['result']['bpb'] for p in points], dtype=float)
    if len(points) != 4 or len(set(compute)) != 4 or min(bpb) <= 0:
        raise ValueError('Four distinct compute budgets and finite positive BPBs required')
    x, y = compute/3e17, np.log(bpb)
    fits = []
    for fraction in (0., .25, .5, .8, .95):
        e = fraction*float(min(bpb))
        fits.append(least_squares(lambda z: np.log(z[0]+z[1]*x**(-z[2]))-y,
                    [e, max(float(max(bpb))-e, .01), .1],
                    bounds=([0., 1e-12, 1e-5], [float(min(bpb)), 10., 5.]),
                    max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12))
    best = min(fits, key=lambda f: float(np.sum(f.fun**2)))
    variance = float(np.sum((y-y.mean())**2))
    if variance == 0:
        raise ValueError('Constant BPB: log-space R squared is undefined')
    r2 = 1-float(np.sum(best.fun**2))/variance
    if r2 <= 0:
        raise ValueError('Nonpositive log-space R squared')
    e, k, power = map(float, best.x)
    ratio = 1e19/3e17
    area = e+k*(1-ratio**(-power))/(power*math.log(ratio))
    return dict(performance=area/r2, a_front=area, log_space_r2=r2,
                fit=dict(E=e, k_at_3e17=k, p=power), actual_train_flops=compute.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True,
                   help='JSON mapping c1,c2,c3,c4 to their campaign-result.json paths')
    p.add_argument('--board', type=Path, default=HERE/'provenance/leaderboard-20261106.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    suite = read(a.suite)
    if set(suite) != {'c1','c2','c3','c4'}:
        p.error('Do not estimate Performance from an incomplete suite')
    points = [read(suite[lev]) for lev in ('c1','c2','c3','c4')]
    if not all(row['complete_A'] for row in points):
        p.error('All four must be complete A results')
    if len({json.dumps(row['profile'], sort_keys=True) for row in points}) != 1:
        p.error('The four evaluation profiles differ')
    if len({row['source_hash'] for row in points}) != 1:
        p.error('The four source revisions differ')
    fit = fit_performance(points)
    result = dict(fit=fit, evaltime=points[-1]['result']['evaltime'],
                  scenario=rerank(read(a.board)['teams'], fit['performance'], points[-1]['result']['evaltime']))
    write(a.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
