"""Recover control scalars from this completed history's pre-export FP32 master.

Training intentionally exports ordinary NN tensors as BF16. Its final ckpt.pt
was saved before that cast. Keep the original export and master untouched; a
small sidecar supplies only the exact already-trained control parameters.
"""
import fcntl,hashlib,json,os
from pathlib import Path


def ensure_fp32_controls(cfg,model_path):
    flags={k:int(cfg['model'].get(k,0)) for k in ['paper_dwa','paper_loss_free','paper_diff']}
    if not any(flags.values()):return None
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    model_path=Path(model_path).resolve();train=model_path.parent;source=train/'ckpt.pt'
    destination=model_path/'paper-control-fp32.safetensors';proof_path=model_path/'paper-control-fp32.json'
    summary=json.loads((train/'summary.json').read_text())
    assert summary['completed_budget'] and summary['tokens_trained']==cfg['train']['tokens']
    assert summary['train_flops']==cfg['accounting']['actual_train_flops']
    st=source.stat();source_identity={'path':str(source),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino}
    config_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    with (train/'fp32-control-export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if proof_path.exists():
            proof=json.loads(proof_path.read_text())
            assert proof['source_checkpoint_identity']==source_identity and proof['config_sha256']==config_sha
            assert hashlib.sha256(destination.read_bytes()).hexdigest()==proof['control_file_sha256']
            return proof
        checkpoint=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        assert checkpoint['step']==cfg['accounting']['steps']-1 and checkpoint['tok_seen']==cfg['train']['tokens']
        assert checkpoint['world']==summary['world']
        weights=checkpoint['model'];keys=[]
        if flags['paper_dwa']:keys.extend(f'model.depth_average.weights.{i}' for i in range(cfg['model']['layers']))
        if flags['paper_loss_free']:keys.extend(f'model.layers.{i}.mlp.gate.routing_bias' for i in range(cfg['model']['layers']))
        if flags['paper_diff']:
            diff_keys=[k for k in weights if '.self_attn.' in k and k.rsplit('.',1)[-1] in ['lambda_q1','lambda_k1','lambda_q2','lambda_k2']]
            types=cfg['model']['layer_types'].split(',') if cfg['model']['layer_types'] else ['full_attention' if i%4==3 else 'linear_attention' for i in range(cfg['model']['layers'])]
            assert len(diff_keys)==4*types.count('full_attention');keys.extend(sorted(diff_keys))
        controls={}
        for key in keys:
            value=weights[key];assert value.dtype==torch.float32 and torch.isfinite(value).all(),key
            controls[key]=value.detach().clone().contiguous()
        if destination.exists():
            with safe_open(str(destination),framework='pt',device='cpu') as reader:
                assert set(reader.keys())==set(controls)
                assert all(torch.equal(reader.get_tensor(k),v) for k,v in controls.items())
        else:
            temporary=destination.with_suffix('.tmp');save_file(controls,str(temporary));os.replace(temporary,destination)
        proof=dict(passed=True,config_sha256=config_sha,source_checkpoint_identity=source_identity,
            source_step=checkpoint['step'],source_tokens=checkpoint['tok_seen'],world=checkpoint['world'],flags=flags,
            control_file=str(destination),control_file_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            scalar_count=sum(v.numel() for v in controls.values()),
            tensors_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in controls.items()},
            original_bf16_hf_and_fp32_master_unchanged=True,new_training_or_nn_forwards=0,
            parameter_counting='Original learned control parameters counted once; unused BF16 control copies are deterministic rounding of the same FP32 values, like other inference caches.',
            scope='Only this completed fixed-seed history; no foreign checkpoint or optimizer transplant')
        temporary=proof_path.with_suffix('.tmp');temporary.write_text(json.dumps(proof,indent=2)+'\n');os.replace(temporary,proof_path)
        return proof
