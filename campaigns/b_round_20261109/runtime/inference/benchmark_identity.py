"""Bind packed inputs and baseline comparisons to one immutable HF checkpoint."""
import hashlib,json
from pathlib import Path
TOKENIZER_FILES=('tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','tokenizer.model','spiece.model')
def sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def tokenizer_identity(model_dir):
    model_dir=Path(model_dir)
    files={n:sha256(model_dir/n) for n in TOKENIZER_FILES if (model_dir/n).is_file()}
    if not files:raise ValueError('No tokenizer files in '+str(model_dir))
    return dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
def write_checkpoint_manifest(model_dir,destination):
    model_dir=Path(model_dir).resolve()
    weights=sorted(model_dir.glob('*.safetensors')) or sorted(model_dir.glob('pytorch_model*.bin'))
    if not weights:raise ValueError('Checkpoint must first be exported to Hugging Face format')
    paths=[*weights,model_dir/'config.json',*model_dir.glob('*.py')]
    files={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
    identity={n:v['sha256'] for n,v in files.items()}
    record=dict(model_path=str(model_dir),files=files,tokenizer=tokenizer_identity(model_dir),
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest())
    Path(destination).write_text(json.dumps(record,indent=2)+'\n');return record
def verify_checkpoint_manifest(model_dir,manifest_path):
    model_dir=Path(model_dir).resolve();record=json.loads(Path(manifest_path).read_text())
    if str(model_dir)!=record['model_path']:raise ValueError('Checkpoint path differs from prepared manifest')
    for name,expected in record['files'].items():
        p=model_dir/name;s=p.stat()
        if (s.st_size,s.st_mtime_ns)!=(expected['bytes'],expected['mtime_ns']):
            raise ValueError('Checkpoint file changed since preparation: '+str(p))
        if not p.name.endswith(('.safetensors','.bin')) and sha256(p)!=expected['sha256']:
            raise ValueError('Checkpoint source/config changed: '+str(p))
    if tokenizer_identity(model_dir)!=record['tokenizer']:raise ValueError('Tokenizer changed since preparation')
    return record


def resolve_model_dir(root):
    settings=json.loads((Path(root)/"benchmark-config.json").read_text())
    return (Path(root)/settings["model_path"]).resolve()
