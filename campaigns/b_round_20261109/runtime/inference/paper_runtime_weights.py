"""Preserve frozen routing selection scalars in FP32 during BF16/FP8 evaluation."""
import hashlib,json,os
from pathlib import Path
import torch


def restore_frozen_routing_bias(model):
    if not getattr(model.config,'paper_loss_free',0):return None
    from safetensors import safe_open
    directory=Path(os.environ.get('Q38_PAPER_NEURAL_MODEL') or model.config._name_or_path)
    if not directory.is_dir():raise ValueError('Local checkpoint path required for exact frozen routing bias')
    index=directory/'model.safetensors.index.json'
    mapping=json.loads(index.read_text())['weight_map'] if index.exists() else None
    rows=[]
    for i,layer in enumerate(model.model.layers):
        key=f'model.layers.{i}.mlp.gate.routing_bias'
        filename='paper-control-fp32.safetensors' if (directory/'paper-control-fp32.safetensors').exists() else mapping[key] if mapping else 'model.safetensors'
        with safe_open(str(directory/filename),framework='pt',device='cpu') as reader:raw=reader.get_tensor(key)
        assert raw.dtype==torch.float32 and raw.shape==layer.mlp.gate.routing_bias.shape and torch.isfinite(raw).all()
        layer.mlp.gate.routing_bias.data=raw.to(device=layer.mlp.gate.weight.device,dtype=torch.float32)
        assert torch.equal(layer.mlp.gate.routing_bias.detach().cpu(),raw)
        rows.append(dict(key=key,sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest()))
    return dict(source=str(directory),dtype='torch.float32',frozen=True,selection_only=True,biases=rows)
