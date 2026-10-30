"""Fresh-process exact-shape capacity check; never reports a quality/EvalTime score."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

from entry_defaults import apply_defaults,ensure_python
ensure_python()
apply_defaults()
p=argparse.ArgumentParser()
p.add_argument('--model-path',required=True)
p.add_argument('--capacity-cache',type=Path,required=True)
p.add_argument('--batch-size',type=int,required=True)
p.add_argument('--sequence-length',type=int,default=4096)
p.add_argument('--result',type=Path,required=True)
args=p.parse_args()
report=dict(scope='capacity_only_not_quality_not_EvalTime',requested_batch=args.batch_size,
            sequence_length=args.sequence_length,model_path=args.model_path)
try:
    from dependency_guard import verify
    report['dependency_guard']=verify()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,set_seed
    from benchmark_identity import tokenizer_identity
    from eval import packed_batch
    from hopper_compat import install_flex_compat
    from runtime import build_forward
    metadata=json.loads((args.capacity_cache/'metadata.json').read_text())
    if not metadata.get('scope','').startswith('capacity_only_'):
        raise ValueError('Worker requires explicitly marked capacity-only cache')
    if metadata['tokenizer_identity']!=tokenizer_identity(args.model_path):
        raise ValueError('Capacity cache tokenizer identity differs from model')
    tokens=np.memmap(args.capacity_cache/'tokens.bin',dtype=np.int32,mode='r')
    mask=np.memmap(args.capacity_cache/'is_text.bin',dtype=np.bool_,mode='r')
    sequences=math.ceil((len(tokens)-1)/args.sequence_length)
    if args.batch_size>sequences:
        raise ValueError('Capacity-only cache does not cover requested batch')
    report['available_sequences']=sequences
    report['capacity_cache_scope']=metadata['scope']
    report['source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
        for f in Path(__file__).parent.glob('*.py')}
    set_seed(0);torch.cuda.set_device(0)
    torch._dynamo.config.suppress_errors=False
    install_flex_compat()
    model=AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True,
        dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    forward,report['paths']=build_forward(model)
    report['gpu']=torch.cuda.get_device_name(0)
    report['actual_batch_shapes']=[]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for number,start in enumerate((0,min(sequences//2,sequences-args.batch_size))):
            x,y=packed_batch(tokens,mask,start,start+args.batch_size,args.sequence_length,metadata['pad_token_id'])
            assert tuple(x.shape)==(args.batch_size,args.sequence_length)
            x=torch.from_numpy(x).cuda();y=torch.from_numpy(y).cuda()
            report['actual_batch_shapes'].append(list(x.shape))
            if number==0:
                out=forward(x);del out
                torch.cuda.synchronize()
            out=forward(x)
            torch.cuda.synchronize()
            assert out.shape==(*x.shape,model.config.vocab_size)
            assert out.dtype==torch.bfloat16,'Qualified entry must materialize BF16 logits'
            report['logits_shape']=list(out.shape)
            report['logits_dtype']=str(out.dtype)
            flat=out.reshape(-1,out.shape[-1]);labels=y.reshape(-1)
            loss=torch.zeros((),device='cuda',dtype=torch.float64)
            for offset in range(0,labels.numel(),512):
                chunk=torch.nn.functional.cross_entropy(flat[offset:offset+512].float(),labels[offset:offset+512],
                    ignore_index=-100,reduction='sum')
                loss+=chunk.to(torch.float64)
            assert math.isfinite(loss.item())
            del out,flat,labels,loss,x,y
    report.update(status='ok',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception as error:
    message=str(error)
    oom=isinstance(error,torch.OutOfMemoryError) if 'torch' in globals() else False
    oom=oom or 'CUDA out of memory' in message or 'CUDA error: out of memory' in message
    report.update(status='oom' if oom else 'error',error_type=type(error).__name__,error=message)
    traceback.print_exc()
args.result.parent.mkdir(parents=True,exist_ok=True)
args.result.write_text(json.dumps(report,indent=2)+'\n')
print('CAPACITY_RESULT',json.dumps({key:report.get(key) for key in ['status','requested_batch','actual_batch_shapes','error_type']}),flush=True)
raise SystemExit(0 if report['status']=='ok' else 42 if report['status']=='oom' else 1)
