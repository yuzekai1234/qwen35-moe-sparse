"""Export exploratory improved examples; report all outcomes on held-out documents.

Input JSONL: id, split (discovery/validation), utf8_bytes, baseline_nll,
longconv_nll, and optional text/features. NLLs must use identical scored inputs.
This utility never infers per-document losses from aggregate BPB.
"""
import argparse
import json
import math
from pathlib import Path


def summarize(rows):
    byte_count = sum(r['utf8_bytes'] for r in rows)
    if not byte_count:
        return dict(documents=0, utf8_bytes=0, baseline_bpb=None, longconv_bpb=None)
    base = sum(r['baseline_nll'] for r in rows)/(math.log(2)*byte_count)
    lc = sum(r['longconv_nll'] for r in rows)/(math.log(2)*byte_count)
    return dict(documents=len(rows), utf8_bytes=byte_count, baseline_bpb=base,
                longconv_bpb=lc, delta_bpb=lc-base,
                improved=sum(r['longconv_nll'] < r['baseline_nll'] for r in rows),
                worsened=sum(r['longconv_nll'] > r['baseline_nll'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pairs', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frozen-features', type=Path,
                   help='JSON list of Boolean feature names, frozen using discovery only')
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Use an empty output directory to preserve earlier analysis')
    rows, ids = [], set()
    for line in a.pairs.read_text().splitlines():
        r = json.loads(line)
        if r['id'] in ids or r['split'] not in ('discovery', 'validation'):
            raise ValueError('Duplicate ID or unsupported split')
        ids.add(r['id'])
        if r['utf8_bytes'] <= 0 or any(not math.isfinite(r[k]) or r[k] < 0
                for k in ('baseline_nll', 'longconv_nll')):
            raise ValueError('Invalid byte count or NLL')
        rows.append(r)
    a.output.mkdir(parents=True, exist_ok=True)
    discoveries = [r for r in rows if r['split'] == 'discovery']
    validations = [r for r in rows if r['split'] == 'validation']
    if not discoveries or not validations:
        raise ValueError('Both discovery and validation documents are required')
    for label, chosen in [
        ('improved_discovery', [r for r in discoveries if r['longconv_nll'] < r['baseline_nll']]),
        ('worsened_discovery', [r for r in discoveries if r['longconv_nll'] > r['baseline_nll']])]:
        (a.output/(label+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
    features = json.loads(a.frozen_features.read_text()) if a.frozen_features else []
    report = dict(all_documents=summarize(rows), discovery=summarize(discoveries),
                  validation=summarize(validations), validation_features={},
                  selection_scope='Improved/worsened exports are post-hoc discovery examples, not validation evidence')
    for name in features:
        if any(type(r.get('features',{}).get(name)) is not bool for r in validations):
            raise ValueError('Missing/non-Boolean frozen feature: '+name)
        chosen = [r for r in validations if r['features'][name]]
        report['validation_features'][name] = summarize(chosen) | {'document_coverage':len(chosen)/len(validations)}
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
