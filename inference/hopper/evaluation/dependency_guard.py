"""Portable package version/source guard; no machine-specific site-packages path."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def verify():
    manifest=json.loads(Path(__file__).with_name('dependency-manifest.json').read_text())
    if list(sys.version_info[:2])!=manifest['python_major_minor']:
        raise RuntimeError('Qualified release Python version mismatch')
    versions={name:importlib.metadata.version(name) for name in manifest['distributions']}
    if versions!=manifest['distributions']:
        raise RuntimeError('Qualified release dependency versions differ: '+json.dumps(versions))
    checked=0
    for package,files in manifest['source_files'].items():
        spec=importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError('Required package missing: '+package)
        base=Path(next(iter(spec.submodule_search_locations)))
        for relative,expected in files.items():
            path=base/relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Qualified dependency source mismatch: '+package+'/'+relative)
            checked+=1
    return dict(status='ok',versions=versions,source_files_checked=checked,
                manifest_sha256=hashlib.sha256(Path(__file__).with_name('dependency-manifest.json').read_bytes()).hexdigest())
