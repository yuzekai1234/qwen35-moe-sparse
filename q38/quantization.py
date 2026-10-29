"""Inference-only W8A8 linear layers, without extra quantization dependencies.

INT8 uses symmetric per-output-channel weights and dynamic per-token activations.
FP8 uses E4M3FN with per-tensor weights and dynamic per-tensor activations, which
works with Ada's tensorwise torch._scaled_mm path. Accumulation/scaling and the
rest of the model retain higher precision. No kernel silently falls back.
"""
import torch
from torch import nn
import torch.nn.functional as F


PRECISIONS = ("bf16", "int8", "fp8")


def quantize_tensor(value, precision, dim=None):
    """Return quantized values and FP32 dequantization scales; zeros stay zero."""
    if precision not in ("int8", "fp8"):
        raise ValueError(f"Unsupported quantized precision: {precision}")
    limit = 127.0 if precision == "int8" else torch.finfo(torch.float8_e4m3fn).max
    value = value.float()
    maximum = value.abs().amax(dim=dim, keepdim=dim is not None)
    scale = (maximum / limit).clamp_min(torch.finfo(torch.float32).tiny)
    scaled = (value / scale).clamp(-limit, limit)
    if precision == "int8":
        return scaled.round().to(torch.int8), scale
    return scaled.to(torch.float8_e4m3fn), scale


class LowPrecisionLinear(nn.Module):
    """Frozen W8A8 linear. `reference` is Q/DQ + FP32 GEMM for quality checks.

    The reference backend is explicitly reported and has no low-precision GEMM
    speed claim. Native uses INT8->INT32 GEMM or FP8 GEMM with BF16 output.
    """
    def __init__(self, linear, precision, backend="native"):
        super().__init__()
        if precision not in ("int8", "fp8") or backend not in ("native", "reference"):
            raise ValueError("Expected int8/fp8 precision and native/reference backend")
        if backend == "native" and linear.weight.device.type != "cuda":
            raise ValueError("Native low-precision GEMM requires CUDA; use reference for CPU checks")
        if backend == "native" and (linear.in_features % 16 or linear.out_features % 16):
            raise ValueError("Native low-precision linear dimensions must be divisible by 16")
        if backend == "native" and precision == "fp8" and torch.cuda.get_device_capability(linear.weight.device) < (8, 9):
            raise ValueError("Native FP8 requires Ada (SM 8.9) or newer")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.precision, self.backend = precision, backend
        weight, scale = quantize_tensor(linear.weight.detach(), precision,
                                        dim=1 if precision == "int8" else None)
        self.register_buffer("quantized_weight", weight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", linear.bias.detach().clone() if linear.bias is not None else None)
        self.eval()

    @torch.no_grad()
    def forward(self, inputs):
        if self.training:
            raise RuntimeError("LowPrecisionLinear supports inference only")
        shape = inputs.shape
        if not shape or shape[-1] != self.in_features:
            raise ValueError(f"Expected input's last dimension to be {self.in_features}")
        flat = inputs.reshape(-1, self.in_features)
        if flat.shape[0] == 0:
            return inputs.new_empty((*shape[:-1], self.out_features))
        quantized, scale = quantize_tensor(flat, self.precision,
                                          dim=1 if self.precision == "int8" else None)
        if self.backend == "reference":
            output = F.linear(quantized.float() * scale,
                              self.quantized_weight.float() * self.weight_scale)
        elif self.precision == "int8":
            # Pad M as well as requiring aligned K/N, including short/tail batches.
            rows = quantized.shape[0]
            padded_rows = max(32, (rows + 15) // 16 * 16)
            padded = F.pad(quantized, (0, 0, 0, padded_rows - rows))
            accum = torch._int_mm(padded.contiguous(), self.quantized_weight.t())[:rows]
            output = accum.float() * scale * self.weight_scale.t()
        else:
            rows = quantized.shape[0]
            # F.pad on float8 is not implemented on every Torch/CUDA combination.
            if rows % 16:
                padded = torch.zeros(((rows + 15) // 16 * 16, self.in_features),
                                     device=inputs.device, dtype=quantized.dtype)
                padded[:rows] = quantized
                quantized = padded
            output = torch._scaled_mm(
                quantized.contiguous(), self.quantized_weight.t(),
                scale_a=scale, scale_b=self.weight_scale,
                out_dtype=torch.bfloat16, use_fast_accum=False,
            )[:rows]
        if self.bias is not None:
            output = output.float() + self.bias.float()
        return output.to(inputs.dtype).reshape(*shape[:-1], self.out_features)


def apply_precision(model, precision="bf16", backend="native"):
    """Quantize only main-path dense FFNs; preserve mHC/GDN/heads/MTP.

    This intentionally does not blanket-replace nn.Linear: mHC reads raw weights
    through F.linear and explicitly computes in FP32. Reject MoE checkpoints
    rather than silently quantizing only their shared experts.
    """
    if precision not in PRECISIONS or backend not in ("native", "reference"):
        raise ValueError("Unsupported precision or backend")
    if model.training:
        raise ValueError("Call model.eval() before applying inference precision")
    if getattr(model, "_q38_precision_applied", False):
        raise ValueError("Reload the original checkpoint before applying another precision")
    if getattr(model.config, "num_experts", 0):
        raise ValueError("This experiment requires a dense checkpoint (num_experts=0)")
    replacements = []
    if precision != "bf16":
        for index, layer in enumerate(model.model.layers):
            for name in ("gate_proj", "up_proj", "down_proj"):
                linear = getattr(layer.mlp, name)
                if not isinstance(linear, nn.Linear):
                    raise TypeError(f"Expected an unmodified dense FFN: layer {index}, {name}")
                replacements.append((layer.mlp, name, f"model.layers.{index}.mlp.{name}",
                                     LowPrecisionLinear(linear, precision, backend)))
    # Validate/construct every layer before mutating the model.
    for parent, name, _, replacement in replacements:
        setattr(parent, name, replacement)
    model._q38_precision_applied = True
    return dict(
        precision=precision, backend="bf16" if precision == "bf16" else backend,
        scope="main_dense_ffn", quantized_modules=[item[2] for item in replacements],
        quantized_weight_elements=sum(item[3].quantized_weight.numel() for item in replacements),
        weight_scaling=("per_output_channel" if precision == "int8" else "per_tensor")
                       if replacements else None,
        activation_scaling=("dynamic_per_token" if precision == "int8" else "dynamic_per_tensor")
                           if replacements else None,
        format="e4m3fn" if precision == "fp8" else precision,
        untouched="attention/GDN, mHC, norms, embeddings, tied output head, training-only MTP",
    )
