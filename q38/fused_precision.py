"""Inference-only dense FFN block backends with immutable weight caches.

The source parameters and checkpoint format are retained. Install only after
loading an eval model. Fixed FP8 activation ranges must come from training data,
never from the evaluation corpus. All GEMMs return BF16 with FP32 accumulation.
"""
import types
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _add_norm(X, Y, W, R, N, Scale, WIDTH: tl.constexpr, EPS: tl.constexpr,
              ADD: tl.constexpr, FP8: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X+row*WIDTH+col, col < WIDTH, other=0).to(tl.float32)
    if ADD:
        y = tl.load(Y+row*WIDTH+col, col < WIDTH, other=0).to(tl.float32)
        x = x+y
        tl.store(R+row*WIDTH+col, x, col < WIDTH)
    var = tl.sum(x*x, 0)/WIDTH
    w = 1.+tl.load(W+col, col < WIDTH, other=0).to(tl.float32)
    norm = (x*tl.rsqrt(var+EPS))*w
    norm = norm.to(tl.bfloat16)
    if FP8:
        norm = tl.minimum(tl.maximum(norm.to(tl.float32)/tl.load(Scale), -448.), 448.)
    tl.store(N+row*WIDTH+col, norm.to(N.dtype.element_ty), col < WIDTH)


def add_norm(x, y, norm, scale=None):
    residual = x if y is None else torch.empty_like(x)
    normalized = torch.empty_like(x, dtype=torch.bfloat16 if scale is None else torch.float8_e4m3fn)
    width = x.shape[-1]
    _add_norm[(x.numel()//width,)](x, y, norm.weight, residual, normalized, scale,
        width, norm.eps, y is not None, scale is not None, triton.next_power_of_2(width),
        num_warps=4, enable_fp_fusion=False)
    return residual, normalized


def _fused_trunk(self, ids):
    if torch.is_grad_enabled():
        raise RuntimeError('Fused trunk is inference-only')
    if ids.ndim != 2 or ids.shape[1] < 1:
        raise ValueError('Fused trunk requires a nonempty packed token matrix')
    from .models.qwen3_8_moe.modeling_qwen3_8_moe import qsa_attention
    e = self.embed_tokens(ids)
    pos = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, ids.shape[0], -1)
    cos, sin = self.rotary_emb(e, pos)
    residual, pending = e.float(), None
    for i, layer in enumerate(self.layers):
        residual, x = add_norm(residual, pending, layer.input_layernorm)
        if self.config.layer_types[i] == 'linear_attention':
            y = layer.linear_attn(hidden_states=x, cache_params=None, attention_mask=None)
        else:
            if getattr(layer.self_attn, '_q38_fp8_attention', False):
                y = _fp8_attention(layer.self_attn, x, cos, sin)
            else:
                y, _ = qsa_attention(layer.self_attn, x, cos, sin, None, 1, True, False)
        scale = getattr(layer.mlp, '_q38_x_scale', None)
        residual, x = add_norm(residual, y, layer.post_attention_layernorm, scale)
        pending = layer.mlp(x)
        if i == 0 and self.ngram is not None:
            residual = residual+pending.float()
            residual = residual+self.ngram.delta(residual.to(e.dtype), ids)
            pending = None
    return (residual if pending is None else residual+pending.float()).to(e.dtype), []


def install_fused_trunk(model):
    if model.training or model.config.mhc_enabled or model.config.qsa_mode != 'full':
        raise ValueError('Fused trunk requires the no-mHC/full-attention eval baseline')
    model.model.trunk = types.MethodType(_fused_trunk, model.model)


def _packed_gdn(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    if cache_params is not None or attention_mask is not None or kwargs:
        raise ValueError('Packed GDN supports uncached fixed-length prefill only')
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    bsz, seq, _ = hidden_states.shape
    projected = F.linear(hidden_states, self._q38_input_weight)
    qkv, z, b, a, _ = torch.split(projected, [self.conv_dim, self.value_dim,
        self.num_v_heads, self.num_v_heads, self._q38_input_padding], dim=-1)
    qkv, _ = causal_conv1d(qkv, weight=self.conv1d.weight.squeeze(1),
                          bias=self.conv1d.bias, activation=self.activation)
    q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(bsz, seq, -1, self.head_k_dim)
    k = k.reshape(bsz, seq, -1, self.head_k_dim)
    v = v.reshape(bsz, seq, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp()*F.softplus(a.float()+self.dt_bias)
    out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
        use_qk_l2norm_in_kernel=True, chunk_size=self._q38_chunk_size)
    out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(out.reshape(bsz, seq, -1))


def install_packed_gdn(model, chunk_size=64):
    if model.training:
        raise ValueError('Packed projection cache is inference-only')
    for layer in model.model.layers:
        if not hasattr(layer, 'linear_attn'):
            continue
        gdn = layer.linear_attn
        weight = torch.cat([getattr(gdn, 'in_proj_'+n).weight.detach()
                            for n in ('qkv', 'z', 'b', 'a')])
        padding = (-weight.shape[0]) % 64
        weight = F.pad(weight, (0, 0, 0, padding))
        gdn.register_buffer('_q38_input_weight', weight, persistent=False)
        gdn._q38_input_padding = padding
        gdn._q38_chunk_size = chunk_size
        gdn.forward = types.MethodType(_packed_gdn, gdn)


@triton.jit
def _quantize(X, Y, Scale, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    x = tl.load(X+i, i < N, other=0).to(tl.float32)
    scale = tl.load(Scale)
    y = tl.minimum(tl.maximum(x/scale, -448.), 448.)
    tl.store(Y+i, y.to(Y.dtype.element_ty), i < N)


@triton.jit
def _swiglu(GU, H, Scale, WIDTH: tl.constexpr, TOTAL: tl.constexpr,
            FP8: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    row, col = i//WIDTH, i % WIDTH
    g = tl.load(GU+row*(2*WIDTH)+col, i < TOTAL, other=0).to(tl.float32)
    u = tl.load(GU+row*(2*WIDTH)+WIDTH+col, i < TOTAL, other=0).to(tl.float32)
    # Match the BF16 rounding boundaries of the original SiLU followed by mul.
    silu = (g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    h = (silu*u).to(tl.bfloat16).to(tl.float32)
    if FP8:
        h = tl.minimum(tl.maximum(h/tl.load(Scale), -448.), 448.)
    tl.store(H+i, h.to(H.dtype.element_ty), i < TOTAL)


def quantize_static(x, scale):
    if not x.is_contiguous():
        raise ValueError('FP8 cast expects a contiguous tensor')
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    _quantize[(triton.cdiv(x.numel(), 1024),)](x, out, scale, x.numel(), 1024)
    return out


def swiglu(gu, scale=None):
    width = gu.shape[-1]//2
    out = torch.empty((*gu.shape[:-1], width), device=gu.device,
                      dtype=torch.bfloat16 if scale is None else torch.float8_e4m3fn)
    _swiglu[(triton.cdiv(out.numel(), 1024),)](
        gu, out, scale, width, out.numel(), scale is not None, 1024)
    return out


def _forward(self, x):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Fused inference FFN requires torch.inference_mode()')
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    if self._q38_backend == 'bf16_fused':
        gu = F.linear(x, self._q38_gate_up)
        h = swiglu(gu)
        y = F.linear(h, self.down_proj.weight)
    else:
        if x.shape[0] % 16:
            # Tiny tails keep the same quantization rule, padding only rows.
            padded = torch.zeros((x.shape[0]+(-x.shape[0]) % 16, x.shape[1]),
                                 dtype=x.dtype, device=x.device)
            padded[:x.shape[0]] = x
            x = padded
        q = x if x.dtype == torch.float8_e4m3fn else quantize_static(x, self._q38_x_scale)
        gu = torch._scaled_mm(q, self._q38_gate_up.t(), scale_a=self._q38_x_scale,
                              scale_b=self._q38_gu_scale, out_dtype=torch.bfloat16,
                              use_fast_accum=False)
        h = swiglu(gu, self._q38_h_scale)
        y = torch._scaled_mm(h, self._q38_down.t(), scale_a=self._q38_h_scale,
                             scale_b=self._q38_down_scale, out_dtype=torch.bfloat16,
                             use_fast_accum=False)[:shape.numel()//shape[-1]]
    return y.reshape(*shape[:-1], self.down_proj.out_features)


def calibrate_ffn(model, batches, margin=1.2):
    """Collect tensor absmax ranges from caller-supplied training batches."""
    if model.training or margin < 1:
        raise ValueError('Calibration requires eval mode and margin >= 1')
    maxima, projection_maxima, handles = {}, {}, []
    for i, layer in enumerate(model.model.layers):
        maxima[str(i)] = dict(x=torch.zeros((), device=layer.mlp.gate_proj.weight.device),
                             h=torch.zeros((), device=layer.mlp.gate_proj.weight.device))
        for module, key in ((layer.mlp.gate_proj, 'x'), (layer.mlp.down_proj, 'h')):
            def collect(mod, args, target=maxima[str(i)], key=key):
                target[key] = torch.maximum(target[key], args[0].detach().abs().amax().float())
            handles.append(module.register_forward_pre_hook(collect))
        # These statistics are collected even for FFN-only candidates so the
        # same training-only pass can support a wider projection ablation.
        if hasattr(layer, 'linear_attn'):
            module = layer.linear_attn
            selected = [('gdn_x', module.in_proj_qkv), ('gdn_out', module.out_proj)]
        else:
            module = layer.self_attn
            selected = [('attn_x', module.q_proj), ('attn_out', module.o_proj)]
        projection_maxima[str(i)] = {}
        for key, projection in selected:
            projection_maxima[str(i)][key] = torch.zeros((), device=projection.weight.device)
            def collect_projection(mod, args, target=projection_maxima[str(i)], key=key):
                target[key] = torch.maximum(target[key], args[0].detach().abs().amax().float())
            handles.append(projection.register_forward_pre_hook(collect_projection))
    count = 0
    try:
        with torch.inference_mode():
            for ids in batches:
                model.model.trunk(ids)
                count += 1
    finally:
        for handle in handles:
            handle.remove()
    if not count:
        raise ValueError('Empty calibration stream')
    return dict(batches=count, margin=margin, layers={i: {k: max(float(v)*margin, 1e-12)/448
                for k, v in values.items()} for i, values in maxima.items()},
                projections={i: {k: max(float(v)*margin, 1e-12)/448
                for k, v in values.items()} for i, values in projection_maxima.items()})


def install_fused_ffn(model, backend, calibration=None):
    if backend not in ('bf16_fused', 'fp8_fused'):
        raise ValueError('Unknown fused FFN backend')
    if model.training or model.config.num_experts:
        raise ValueError('Fused FFN requires a dense eval model')
    if getattr(model, '_q38_fused_ffn', False):
        raise ValueError('Reload the source model before changing backends')
    if backend == 'fp8_fused' and (calibration is None or
                                  torch.cuda.get_device_capability() < (8, 9)):
        raise ValueError('FP8 requires training-data calibration and SM89+')
    if backend == 'fp8_fused':
        validate_calibration(model, calibration)
    for layer in model.model.layers:
        mlp = layer.mlp
        if mlp.gate_proj.weight.dtype != torch.bfloat16:
            raise ValueError('Load BF16 inference weights first')
        if backend == 'fp8_fused' and (mlp.gate_proj.in_features % 16 or mlp.gate_proj.out_features % 16):
            raise ValueError('FP8 GEMM dimensions must be divisible by 16')
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gu = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight)).detach()
        mlp._q38_backend = backend
        if backend == 'bf16_fused':
            mlp.register_buffer('_q38_gate_up', gu, persistent=False)
        else:
            for name, value in (('gate_up', gu), ('down', mlp.down_proj.weight.detach())):
                scale = value.float().abs().amax().clamp_min(1e-12)/448
                quant = quantize_static(value, scale)
                mlp.register_buffer('_q38_'+name, quant, persistent=False)
                mlp.register_buffer('_q38_'+('gu' if name == 'gate_up' else 'down')+'_scale', scale,
                                    persistent=False)
            for key in ('x', 'h'):
                mlp.register_buffer('_q38_'+key+'_scale', torch.tensor(
                    calibration['layers'][str(i)][key], dtype=torch.float32,
                    device=gu.device), persistent=False)
        mlp.forward = types.MethodType(_forward, mlp)
    model._q38_fused_ffn = True
    return dict(backend=backend, blocks=len(model.model.layers),
                activation_scaling='fixed_training_calibration' if calibration else None)


class _Fp8ForwardBf16Backward(torch.autograd.Function):
    """Real FP8 forward GEMM, straight-through BF16 backward GEMMs."""
    @staticmethod
    def forward(ctx, x, weight, scale):
        x_bf16, w_bf16 = x.to(torch.bfloat16), weight.to(torch.bfloat16)
        ctx.save_for_backward(x_bf16, w_bf16)
        ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
        original_shape = x.shape
        flat = x_bf16.reshape(-1, x.shape[-1])
        if flat.shape[0] % 16:
            flat = F.pad(flat, (0, 0, 0, (-flat.shape[0]) % 16))
        wscale = w_bf16.float().abs().amax().clamp_min(1e-12)/448
        out = torch._scaled_mm(quantize_static(flat, scale), quantize_static(w_bf16, wscale).t(),
            scale_a=scale, scale_b=wscale, out_dtype=torch.bfloat16, use_fast_accum=False)
        return out[:x.numel()//x.shape[-1]].reshape(*original_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1])
        dy = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        dx = (dy@w).reshape(x.shape).to(ctx.input_dtype) if ctx.needs_input_grad[0] else None
        dw = (dy.t()@flat).to(ctx.weight_dtype) if ctx.needs_input_grad[1] else None
        return dx, dw, None


def _training_forward(self, x):
    weight = torch.cat((self.gate_proj.weight, self.up_proj.weight))
    gu = _Fp8ForwardBf16Backward.apply(x, weight, self._q38_x_scale)
    gate, up = gu.chunk(2, dim=-1)
    h = F.silu(gate)*up
    return _Fp8ForwardBf16Backward.apply(h, self.down_proj.weight, self._q38_h_scale)


def validate_calibration(model, calibration):
    if set(calibration['layers']) != {str(i) for i in range(len(model.model.layers))}:
        raise ValueError('Calibration does not cover every main-path FFN')
    for values in calibration['layers'].values():
        for key in ('x', 'h'):
            if not math.isfinite(values[key]) or values[key] <= 0:
                raise ValueError('Invalid calibration scale')


def _projection_training_forward(self, x):
    return _Fp8ForwardBf16Backward.apply(x, self.weight, self._q38_projection_scale)


def install_fp8_training(model, calibration, projections=False):
    if model.config.num_experts or model.config.mhc_enabled or model.config.qsa_mode != 'full':
        raise ValueError('FP8 training requires the current dense baseline')
    if getattr(model, '_q38_fused_ffn', False):
        raise ValueError('Cannot train immutable inference weight caches')
    validate_calibration(model, calibration)
    if projections:
        validate_projection_calibration(model, calibration)
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for key in ('x', 'h'):
            value = calibration['layers'][str(i)][key]
            mlp.register_buffer('_q38_'+key+'_scale', torch.tensor(value, dtype=torch.float32,
                device=mlp.gate_proj.weight.device), persistent=False)
        mlp.forward = types.MethodType(_training_forward, mlp)
    if projections:
        for i, layer in enumerate(model.model.layers):
            values = calibration['projections'][str(i)]
            if hasattr(layer, 'linear_attn'):
                mod = layer.linear_attn
                selected = [(getattr(mod, 'in_proj_'+n), values['gdn_x']) for n in ('qkv', 'z')]
                selected.append((mod.out_proj, values['gdn_out']))
            else:
                mod = layer.self_attn
                selected = [(getattr(mod, n+'_proj'), values['attn_x']) for n in ('q', 'k', 'v')]
                selected.append((mod.o_proj, values['attn_out']))
            for projection, scale in selected:
                projection.register_buffer('_q38_projection_scale', torch.tensor(scale,
                    device=projection.weight.device, dtype=torch.float32), persistent=False)
                projection.forward = types.MethodType(_projection_training_forward, projection)
    model.config.fp8_ffn_calibration = calibration
    model.config.ffn_training_precision = 'fp8'
    model.config.fp8_projection_training = projections
    return dict(main_ffn_forward='FP8 E4M3', backward='BF16 GEMMs',
                master_parameters='FP32', activation_scales='fixed_training_calibration',
                gdn_and_attention_projections='FP8 E4M3' if projections else 'BF16')


def validate_projection_calibration(model, calibration):
    values = calibration.get('projections', {})
    for i, layer in enumerate(model.model.layers):
        names = ('gdn_x', 'gdn_out') if hasattr(layer, 'linear_attn') else ('attn_x', 'attn_out')
        for name in names:
            value = values.get(str(i), {}).get(name, 0)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Missing/invalid projection calibration: layer {i}, {name}')


def _fp8_mm(module, name, q, scale):
    return torch._scaled_mm(q, getattr(module, '_q38_'+name+'_weight').t(), scale_a=scale,
        scale_b=getattr(module, '_q38_'+name+'_wscale'), out_dtype=torch.bfloat16, use_fast_accum=False)


def _quantized_projection(module, name, x, scale):
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).contiguous()
    rows = flat.shape[0]
    if rows % 16:
        flat = F.pad(flat, (0, 0, 0, (-rows) % 16))
    y = _fp8_mm(module, name, quantize_static(flat, scale), scale)[:rows]
    return y.reshape(*shape[:-1], y.shape[-1])


def _fp8_gdn(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    if torch.is_grad_enabled() or cache_params is not None or attention_mask is not None or kwargs:
        raise ValueError('FP8 GDN deployment supports uncached inference prefill only')
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    bsz, seq, _ = hidden_states.shape
    # Reuse one quantization for both large input projections. The small a/b
    # control projections retain their original BF16 input.
    flat = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    rows = flat.shape[0]
    if rows % 16:
        flat = F.pad(flat, (0, 0, 0, (-rows) % 16))
    qx = quantize_static(flat, self._q38_x_scale)
    qkv = _fp8_mm(self, 'qkv', qx, self._q38_x_scale)[:rows].reshape(bsz, seq, -1)
    z = _fp8_mm(self, 'z', qx, self._q38_x_scale)[:rows].reshape(bsz, seq, -1)
    qkv, _ = causal_conv1d(qkv, weight=self.conv1d.weight.squeeze(1),
                          bias=self.conv1d.bias, activation=self.activation)
    q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(bsz, seq, -1, self.head_k_dim)
    k = k.reshape(bsz, seq, -1, self.head_k_dim)
    v = v.reshape(bsz, seq, -1, self.head_v_dim)
    beta = self.in_proj_b(hidden_states).sigmoid()
    g = -self.A_log.float().exp()*F.softplus(self.in_proj_a(hidden_states).float()+self.dt_bias)
    out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return _quantized_projection(self, 'out', out.reshape(bsz, seq, -1), self._q38_out_scale)


def _fp8_attention(attn, x, cos, sin):
    from .models.qwen3_8_moe.modeling_qwen3_8_moe import apply_rotary_pos_emb
    bsz, seq, _ = x.shape
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    rows = flat.shape[0]
    if rows % 16:
        flat = F.pad(flat, (0, 0, 0, (-rows) % 16))
    qx = quantize_static(flat, attn._q38_x_scale)
    proj = [_fp8_mm(attn, n, qx, attn._q38_x_scale)[:rows].reshape(bsz, seq, -1)
            for n in ('q', 'k', 'v')]
    q, gate = proj[0].view(bsz, seq, -1, attn.head_dim*2).chunk(2, dim=-1)
    q = attn.q_norm(q).transpose(1, 2)
    k = attn.k_norm(proj[1].view(bsz, seq, -1, attn.head_dim)).transpose(1, 2)
    v = proj[2].view(bsz, seq, -1, attn.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))
    o = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling, is_causal=True, enable_gqa=True)
    o = o.transpose(1, 2).reshape(bsz, seq, -1)*torch.sigmoid(gate.reshape(bsz, seq, -1))
    return _quantized_projection(attn, 'o', o, attn._q38_out_scale)


def install_fp8_projections(model, calibration, scope='all'):
    if model.training or model.config.mhc_enabled or model.config.qsa_mode != 'full':
        raise ValueError('FP8 projections require the current eval baseline')
    if scope not in ('gdn', 'all'):
        raise ValueError('Projection scope must be gdn or all')
    validate_projection_calibration(model, calibration)
    for i, layer in enumerate(model.model.layers):
        values = calibration['projections'][str(i)]
        if hasattr(layer, 'linear_attn'):
            mod, xkey, okey = layer.linear_attn, 'gdn_x', 'gdn_out'
            selected = [(n, getattr(mod, 'in_proj_'+n)) for n in ('qkv', 'z')]
            selected.append(('out', mod.out_proj))
        elif scope == 'all':
            mod, xkey, okey = layer.self_attn, 'attn_x', 'attn_out'
            selected = [(n, getattr(mod, n+'_proj')) for n in ('q', 'k', 'v', 'o')]
            mod._q38_fp8_attention = True
        else:
            continue
        for name, projection in selected:
            weight = projection.weight.detach()
            scale = weight.float().abs().amax().clamp_min(1e-12)/448
            mod.register_buffer('_q38_'+name+'_weight', quantize_static(weight, scale), persistent=False)
            mod.register_buffer('_q38_'+name+'_wscale', scale, persistent=False)
        for name, key in (('x', xkey), ('out', okey)):
            mod.register_buffer('_q38_'+name+'_scale', torch.tensor(values[key],
                dtype=torch.float32, device=weight.device), persistent=False)
        if hasattr(layer, 'linear_attn'):
            mod.forward = types.MethodType(_fp8_gdn, mod)
