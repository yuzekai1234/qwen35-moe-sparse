"""Hash-attestation fixtures; no dataset files are read or created."""
from scripts.q38_isoflops_contract import DATA_BYTES, DATA_SHA256, TOKENIZER_SHA256


def data_identity():
    result = {'tokenizer_sha256': TOKENIZER_SHA256}
    for name, filename in [('train', 'train.bin'), ('validation', 'valid.bin')]:
        result[name] = dict(path='/data/q38/fixture/'+filename, bytes=DATA_BYTES[name],
                            mtime_ns=1000000, sha256=DATA_SHA256[name])
    return result


def runtime_data_identity(snapshot):
    return {name:{key:snapshot[name][key] for key in ('path', 'bytes', 'mtime_ns')}
            for name in ('train', 'validation')}


def synthetic_fit():
    import math
    import numpy as np
    from scripts.fit_q38_isoflops import fit_curves
    from scripts.make_q38_isoflops_configs import build_plan, BUDGETS
    from scripts.q38_isoflops_contract import FAMILY
    snapshot = data_identity()
    _, manifest = build_plan('/unused/configs', '/unused/runs', 'test-fixture',
        data_identity=snapshot, data_root='/data/q38/fixture', tokenizer='/data/q38/fixture/tokenizer')
    rows = []
    for label, compute in BUDGETS.items():
        jobs = [job for job in manifest['jobs'] if job['compute'] == compute]
        center = float(np.median(np.log10([job['Na'] for job in jobs])))
        for job in jobs:
            rows.append(dict(id=job['id'], family=FAMILY, budget=label, hidden=job['hidden'],
                target_compute=compute, compute=job['expected_tokens']*job['train_flops_per_token'],
                Na=job['Na'], D=job['expected_tokens'], loss=2+.4*(math.log10(job['Na'])-center)**2))
    fit = fit_curves(rows)
    fit.update(metric='pure_cross_entropy', metric_units='nats/target including EOS',
        data_identity=snapshot, validation=manifest['validation'], source_commit='test-fixture',
        manifest_sha256='0'*64, fitting_code_sha256='1'*64,
        data_root='/data/q38/fixture', tokenizer='/data/q38/fixture/tokenizer')
    return fit, rows
