"""Keep the learned depth-mixture control scalars at checkpoint FP32 precision."""
import hashlib,json,os
from pathlib import Path
import torch


def restore_frozen_dwa_coefficients(model):
    if not getattr(model.config,'paper_dwa',0):return dict(enabled=False)
    from safetensors import safe_open
    directory=Path(os.environ.get('Q38_PAPER_NEURAL_MODEL') or model.config._name_or_path)
    index=directory/'model.safetensors.index.json'
    mapping=json.loads(index.read_text())['weight_map'] if index.exists() else None
    rows=[];scalars=0
    for i,parameter in enumerate(model.model.depth_average.weights):
        key=f'model.depth_average.weights.{i}'
        filename='paper-control-fp32.safetensors' if (directory/'paper-control-fp32.safetensors').exists() else mapping[key] if mapping else 'model.safetensors'
        with safe_open(str(directory/filename),framework='pt',device='cpu') as reader:raw=reader.get_tensor(key)
        assert raw.dtype==torch.float32 and raw.shape==parameter.shape and torch.isfinite(raw).all()
        parameter.data=raw.to(device=parameter.device,dtype=torch.float32)
        assert parameter.dtype==torch.float32 and torch.equal(parameter.detach().cpu(),raw)
        rows.append(dict(key=key,sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),device=str(parameter.device)))
        scalars+=raw.numel()
    return dict(enabled=True,source=str(directory),dtype='torch.float32',exact_checkpoint_roundtrip=True,
        independent_control_scalars=scalars,weights=rows)
