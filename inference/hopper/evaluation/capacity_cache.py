"""Explicit capacity-only repetition of genuine packed tokens; never a quality cache."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import numpy as np


def build_from_packed(source,required_batch,sequence_length,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination:
        raise ValueError('Capacity cache may not overwrite its source/quality cache')
    if required_batch<1 or sequence_length<1:
        raise ValueError('Positive capacity dimensions required')
    metadata=json.loads((source/'metadata.json').read_text())
    token_bytes=(source/'tokens.bin').stat().st_size
    if token_bytes%4 or (source/'is_text.bin').stat().st_size!=token_bytes//4:
        raise ValueError('Corrupt packed source lengths')
    base_length=token_bytes//4
    if base_length<2:
        raise ValueError('Capacity source must contain real token targets')
    requested_length=required_batch*sequence_length+1
    repeats=max(1,math.ceil(requested_length/base_length))
    n_sequences=math.ceil((base_length*repeats-1)/sequence_length)
    expected=dict(scope='capacity_only_repeated_real_tokens_not_quality_not_calibration',
                  source_cache=str(source),source_token_file_sha256=hashlib.file_digest((source/'tokens.bin').open('rb'),'sha256').hexdigest(),
                  source_mask_file_sha256=hashlib.file_digest((source/'is_text.bin').open('rb'),'sha256').hexdigest(),
                  tokenizer_identity=metadata['tokenizer_identity'],
                  pad_token_id=metadata['pad_token_id'],repeats=repeats,
                  requested_batch=required_batch,sequence_length=sequence_length,
                  available_sequences=n_sequences,
                  source_quality_scope=metadata.get('scope','provided_real_window'),
                  source_dataset_sha256=metadata.get('dataset_sha256'),
                  total_tokens=metadata.get('total_tokens',0)*repeats,
                  total_bytes=metadata.get('total_bytes',0)*repeats)
    if (destination/'metadata.json').exists():
        previous=json.loads((destination/'metadata.json').read_text())
        if all(previous.get(k)==expected[k] for k in ['source_token_file_sha256','source_mask_file_sha256','tokenizer_identity','sequence_length']) and previous.get('available_sequences',0)>=required_batch:
            return destination
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['tokens.bin','is_text.bin']:
        temporary=destination/(name+'.tmp')
        with temporary.open('wb') as out:
            for _ in range(repeats):
                with (source/name).open('rb') as src:
                    shutil.copyfileobj(src,out,length=8*1024*1024)
        temporary.replace(destination/name)
    (destination/'metadata.json').write_text(json.dumps(expected,indent=2)+'\n')
    return destination


def prepare_real_window(dataset,model,required_batch,sequence_length,destination,records=256):
    """Tokenize a bounded real window on CPU, then explicitly repeat for capacity."""
    from transformers import PreTrainedTokenizerFast
    from benchmark_identity import tokenizer_identity
    from eval import pack_jsonl
    destination=Path(destination);base=destination/'source-window'
    base.mkdir(parents=True,exist_ok=True)
    window=base/'source.jsonl'
    with Path(dataset).open(encoding='utf-8') as src,window.open('w',encoding='utf-8') as out:
        count=0
        for line in src:
            if not line.strip():continue
            json.loads(line)['text']
            out.write(line if line.endswith('\n') else line+'\n')
            count+=1
            if count>=records:break
    tokenizer=PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True)
    metadata=pack_jsonl(window,tokenizer,base)
    metadata.update(scope='capacity_source_window_not_quality',
                    tokenizer_identity=tokenizer_identity(model),
                    dataset_sha256=hashlib.file_digest(window.open('rb'),'sha256').hexdigest())
    (base/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return build_from_packed(base,required_batch,sequence_length,destination/'repeated')


def actual_pilot_batches(total_sequences,requested_batch):
    if total_sequences<1 or requested_batch<1:
        raise ValueError('Positive pilot dimensions required')
    return [min(requested_batch,total_sequences-start)
            for start in range(0,total_sequences,requested_batch)]
