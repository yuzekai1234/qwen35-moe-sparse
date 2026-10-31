"""Fuse full-attention output layout, BF16 gate rounding, and FP8 row quantization."""
import inspect
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_quant(O, G, Q, Scale, SEQ:tl.constexpr, HEADS:tl.constexpr, D:tl.constexpr,
                S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    col=tl.arange(0,BLOCK)
    width=HEADS*D
    b=row//SEQ;t=row%SEQ;h=col//D;d=col%D
    out=tl.load(O+b*S0+h*S1+t*S2+d*S3,col<width,other=0).to(tl.float32)
    gate=tl.load(G+row*width+col,col<width,other=0).to(tl.float32)
    sigmoid=tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    value=(out*sigmoid).to(tl.bfloat16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(value),0)/448.,1.e-12)
    tl.store(Q+row*width+col,tl.minimum(tl.maximum(value/scale,-448.),448.),col<width)
    tl.store(Scale+row,scale)


def project(attn,o,gate):
    if torch.is_grad_enabled() or attn.training:
        raise RuntimeError('Attention fusion is inference-only')
    b,h,s,d=o.shape;linear=attn.o_proj;width=h*d
    assert gate.dtype==o.dtype==torch.bfloat16
    assert linear.in_features==width and hasattr(linear,'_q8_weight')
    gate=gate.contiguous()
    q=torch.empty((b*s,width),device=o.device,dtype=torch.float8_e4m3fn)
    scales=torch.empty(b*s,device=o.device,dtype=torch.float32)
    _gate_quant[(b*s,)](o,gate,q,scales,s,h,d,*o.stride(),triton.next_power_of_2(width),num_warps=4)
    import scaled_pending
    if scaled_pending.enabled():
        return scaled_pending.gemm(q,scales,linear._q8_weight,linear._q8_weight_scale,(b,s,linear.out_features))
    result=torch._scaled_mm(q,linear._q8_weight.t(),scale_a=scales[:,None],
        scale_b=linear._q8_weight_scale[None,:],bias=linear._q8_bias,
        out_dtype=torch.bfloat16,use_fast_accum=True)
    return result[:,:linear.out_features].reshape(b,s,linear.out_features)


def install(model):
    trunk=model.model
    original=trunk._common_full_qsa
    source=inspect.getsource(original)
    old='o = o.transpose(1, 2).reshape(B, S, -1) * torch.sigmoid(gate)\n    return attn.o_proj(o), kl'
    assert source.count(old)==1,'Unexpected attention implementation; refuse replacement'
    namespace=dict(original.__globals__)
    namespace['_infra_attention_project']=project
    exec(compile(source.replace(old,'return _infra_attention_project(attn, o, gate), kl'),
                 __file__+':qualified_full_attention','exec'),namespace)
    trunk._common_full_qsa=namespace[original.__name__]
    return dict(implementation='fused_attention_layout_gate_FP8_quant',
                rounding='BF16_sigmoid_then_BF16_product',checkpoint_parameters_changed=False,
                experts_modified=False,router_modified=False)
