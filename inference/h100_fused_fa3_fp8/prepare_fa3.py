"""Copy the exact qualified external FA3 runtime without changing site-packages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--from-hopper', type=Path, required=True)
p.add_argument('--destination', type=Path,
               default=Path(__file__).resolve().parent / 'external/fa3/hopper')
a = p.parse_args()
manifest = json.loads(Path(__file__).with_name('evaluation').joinpath('fa3-dependency.json').read_text())
for name, expected in manifest['files'].items():
    source = a.from_hopper / name
    with source.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise SystemExit('Qualified source artifact differs: ' + str(source))
a.destination.mkdir(parents=True, exist_ok=False)
for name in manifest['files']:
    target = a.destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.from_hopper / name, target)
(a.destination / 'qualified-artifact.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(a.destination.resolve())
