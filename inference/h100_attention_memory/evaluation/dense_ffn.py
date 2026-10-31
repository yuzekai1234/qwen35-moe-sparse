"""Inference-only dense SwiGLU: packed FP8 GEMM and fused dynamic quantization.

No expert or routing kernels. Checkpoint parameters remain unchanged; installation
creates nonpersistent caches. This candidate must be qualified on the actual
Dense checkpoint before being selected.
"""
import types
import torch
import triton
import triton.language as tl
from low_precision import quant_rows


@triton.jit
def _swiglu_quant(GU,Q,S,N:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);column=tl.arange(0,BLOCK)
    gate=tl.load(GU+row*(2*N)+column,column<N,other=0.).to(tl.float32)
    up=tl.load(GU+row*(2*N)+N+column,column<N,other=0.).to(tl.float32)
    # Preserve the BF16 SiLU and product rounding of the unfused dense FFN.
    activated=(gate*tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    hidden=(activated*up).to(tl.bfloat16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(hidden),0)/448.,1.e-12)
    tl.store(Q+row*N+column,tl.minimum(tl.maximum(hidden/scale,-448.),448.),column<N)
    tl.store(S+row,scale)


def _forward(self,x):
    if self.training or torch.is_grad_enabled():raise RuntimeError('Dense FP8 FFN is inference-only')
    shape=x.shape;flat=x.reshape(-1,shape[-1])
    q,scale=quant_rows(flat)
    gu=torch._scaled_mm(q,self._dense_q_gate_up.t(),scale_a=scale[:,None],
        scale_b=self._dense_s_gate_up[None,:],out_dtype=torch.bfloat16,use_fast_accum=True)
    del q,scale
    rows=flat.shape[0];width=self._dense_width
    hidden=torch.empty((rows,width),device=x.device,dtype=torch.float8_e4m3fn)
    hidden_scale=torch.empty(rows,device=x.device,dtype=torch.float32)
    _swiglu_quant[(rows,)](gu,hidden,hidden_scale,width,triton.next_power_of_2(width),num_warps=4)
    del gu
    y=torch._scaled_mm(hidden,self._dense_q_down.t(),scale_a=hidden_scale[:,None],
        scale_b=self._dense_s_down[None,:],out_dtype=torch.bfloat16,use_fast_accum=True)
    return y.reshape(*shape[:-1],self.down_proj.out_features)


def _weight_cache(linear):
    if hasattr(linear,'_q8_weight'):
        return linear._q8_weight,linear._q8_weight_scale
    q,scale=quant_rows(linear.weight.detach())
    linear.register_buffer('_q8_weight',q,persistent=False)
    linear.register_buffer('_q8_weight_scale',scale,persistent=False)
    linear.weight.data=linear.weight.detach().cpu()
    return q,scale


def install(model):
    if model.training or getattr(model.config,'num_experts',0):
        raise ValueError('Dense FFN fusion requires an eval checkpoint with num_experts=0')
    if getattr(model.config,'hidden_act','silu') not in ('silu','swish'):
        raise ValueError('Dense FFN candidate requires SwiGLU')
    if torch.cuda.get_device_capability()[0]!=9:raise RuntimeError('Qualified target is Hopper SM90')
    count=0;names=[]
    with torch.no_grad():
        for name,module in model.named_modules():
            if not all(isinstance(getattr(module,n,None),torch.nn.Linear) for n in ('gate_proj','up_proj','down_proj')):continue
            gate,up,down=module.gate_proj,module.up_proj,module.down_proj
            width=gate.out_features
            if gate.in_features%16 or width%16 or up.weight.shape!=gate.weight.shape or down.in_features!=width:
                raise ValueError('Unsupported dense FFN shape: '+name)
            if any(p.bias is not None for p in (gate,up,down)):raise ValueError('Expected bias-free dense FFN')
            g,sg=_weight_cache(gate);u,su=_weight_cache(up);d,sd=_weight_cache(down)
            packed=torch.cat((g,u),dim=0);packed_scale=torch.cat((sg,su))
            module.register_buffer('_dense_q_gate_up',packed,persistent=False)
            module.register_buffer('_dense_s_gate_up',packed_scale,persistent=False)
            module.register_buffer('_dense_q_down',d,persistent=False)
            module.register_buffer('_dense_s_down',sd,persistent=False)
            # Retain usable projection buffers as views, avoiding duplicate caches.
            gate._q8_weight=packed[:width];up._q8_weight=packed[width:]
            gate._q8_weight_scale=packed_scale[:width];up._q8_weight_scale=packed_scale[width:]
            module._dense_width=width;module.forward=types.MethodType(_forward,module)
            count+=1;names.append(name)
    if not count:raise ValueError('No supported dense SwiGLU modules found')
    return dict(modules=count,selected=names,implementation='SM90_native_E4M3_W8A8',
        activation_scale='dynamic_per_row',weight_scale='per_output_row',
        gate_up_gemm='combined',activation_multiply_quantization='fused_BF16_rounding',
        torch_scaled_mm_use_fast_accum=True,checkpoint_parameters_changed=False)
