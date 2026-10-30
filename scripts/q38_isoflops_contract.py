"""Pure-Python graph, grid and independently hashed data identities."""
from pathlib import PurePath

FAMILY = 'dense_no_mhc_full_attention_bf16'
BUDGETS = {'3e15': 3*10**15, '1e16': 10**16, '3e16': 3*10**16, '1e17': 10**17}
GRID = {'3e15': (64, 128, 192, 256), '1e16': (128, 192, 256, 320),
        '3e16': (192, 256, 320, 384), '1e17': (256, 320, 384, 448, 512)}
TOKENIZER_SHA256 = '5186311f982eb27783fe3c0f4d6417f61c92a1f2aff4c9b0611b5d00bd6645a7'
DATA_SHA256 = {
    'train': '9990c2b1345f1fc2a567c76ced14618b5315bebe4bcaeed17359a834f0bd9cf5',
    'validation': '19d30b21482012f07062f46d2d204ef2b27a101728b33a66f992348623fda924',
}
DATA_BYTES = {'train': 22468889244, 'validation': 313526136}
VALIDATION_OFFSET = 2002536
VALIDATION_TARGETS = 16001875
RUNTIME_OVERRIDES = frozenset({'micro_bs', 'auto_micro_bs', 'resume', 'tensorboard', 'ckpt_every', 'log_every'})


def _canonical_absolute_path(value):
    return (isinstance(value, str) and '\0' not in value and not value.startswith('//') and
            PurePath(value).is_absolute() and '..' not in PurePath(value).parts and
            str(PurePath(value)) == value)


def validate_data_identity(snapshot):
    """Validate the supplied hash attestation without opening dataset paths."""
    if not isinstance(snapshot, dict) or snapshot.get('tokenizer_sha256') != TOKENIZER_SHA256:
        raise ValueError('Missing or different independently hashed tokenizer identity')
    normalized = {'tokenizer_sha256': TOKENIZER_SHA256}
    for name in ('train', 'validation'):
        item = snapshot.get(name)
        if (not isinstance(item, dict) or not _canonical_absolute_path(item.get('path')) or
                type(item.get('bytes')) is not int or item['bytes'] != DATA_BYTES[name] or
                type(item.get('mtime_ns')) is not int or item['mtime_ns'] <= 0 or
                item.get('sha256') != DATA_SHA256[name]):
            raise ValueError(f'Missing or different independently hashed {name} identity')
        normalized[name] = {key: item[key] for key in ('path', 'bytes', 'mtime_ns', 'sha256')}
    return normalized


def validate_config_data_paths(config, snapshot):
    """Require the attested canonical paths, not an unverified local alias."""
    identity = validate_data_identity(snapshot)
    train = config.get('train', {})
    for stream,key in [('train','data'), ('validation','val_data')]:
        value = train.get(key)
        if not _canonical_absolute_path(value) or value != identity[stream]['path']:
            raise ValueError(f'Configured {key} must match the snapshot canonical path')


def runtime_identity_matches(actual, snapshot):
    """Bind runtime path/stat records to the explicit whole-file hash snapshot."""
    expected = validate_data_identity(snapshot)
    return isinstance(actual, dict) and all(
        isinstance(actual.get(name), dict) and
        all(actual[name].get(key) == expected[name][key] for key in ('path', 'bytes', 'mtime_ns'))
        for name in ('train', 'validation'))


def validate_fit_grid(points):
    expected = {(BUDGETS[label], hidden) for label, widths in GRID.items() for hidden in widths}
    points = list(points)
    if len(points) != len(expected) or set(points) != expected:
        raise ValueError('Need the complete declared 17-point grid, without C1 or duplicate sizes')


def validate_model_family(config):
    model, train = config.get('model', {}), config.get('train', {})
    if (any(model.get(key) != value for key,value in
            {'experts':0, 'topk':0, 'mhc_enabled':0, 'qsa_mode':'full', 'mtp':1}.items()) or
            any(train.get(key) != value for key,value in
            {'ffn_precision':'bf16', 'projection_precision':'bf16', 'ffn_calibration':'',
             'precision_transition':0, 'mtp_coef':.3, 'kl_coef':0., 'dense_warmup':0}.items())):
        raise ValueError('Configuration is not the current no-mHC/full-attention BF16 family')
