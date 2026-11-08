"""The same inference setup is used by qualification and the submitted entry."""
import os
import torch
from hopper_inference import optimize_for_inference, optimize_mhc_mix


def build_feature_forward(model):
    from paper_runtime_weights import restore_frozen_routing_bias
    frozen_bias_proof=restore_frozen_routing_bias(model)
    from dwa_runtime_precision import restore_frozen_dwa_coefficients
    frozen_dwa_proof=restore_frozen_dwa_coefficients(model)
    if getattr(model.config,'paper_dwa',0) and os.environ.get('Q38_PREQUANT_NORM','1')!='1':
        raise ValueError('DWA requires its qualified prequant depth-mixing trunk')
    precision = os.environ.get("Q38_FP32_MATMUL_PRECISION", "highest")
    if precision not in ("highest", "high"):
        raise ValueError("Q38_FP32_MATMUL_PRECISION must be highest or high")
    torch.set_float32_matmul_precision(precision)
    paths = optimize_for_inference(
        model, experts=os.environ.get("Q38_EXPERTS", "triton"), gdn=True, norms=True)
    expected_gdn = sum(type(module).__name__ == "Qwen3_8MoeGatedDeltaNet" for module in model.modules())
    if paths["gated_deltanet"] != expected_gdn:
        raise RuntimeError(f"GDN fast-path installation incomplete: {paths}")
    paths["persistent_routing_bias"] = frozen_bias_proof
    paths["persistent_dwa_coefficients"] = frozen_dwa_proof
    paths["mhc_fused"] = optimize_mhc_mix(model)
    expected_mhc = sum(type(module).__name__ == "MHC" and module.n == 4 for module in model.modules())
    if paths["mhc_fused"] != expected_mhc:
        raise RuntimeError(f"mHC fast-path installation incomplete: {paths}")
    paths["gdn_batch_limit"] = int(os.environ.get("Q38_GDN_BATCH_LIMIT", "64"))
    paths["large_flex_slab"] = 128
    paths["float32_matmul_precision"] = precision
    paths["cuda_graph"] = os.environ.get("Q38_CUDA_GRAPH", "0") == "1"
    variant = os.environ.get("Q38_VARIANT", "baseline")
    paths["variant"] = variant
    if variant != "baseline":
        if paths["cuda_graph"]:
            raise ValueError("Quantized qualification requires Q38_CUDA_GRAPH=0")
        from low_precision import install_variant
        paths["variant_details"] = install_variant(model, variant)
        paths["experts"] = getattr(model.config, "_experts_implementation", "dense_ffn") if getattr(model.config, "num_experts", 0) else "dense_ffn"
    if paths["cuda_graph"]:
        from cuda_graph import GraphForward
        forward = GraphForward(model)
    else:
        def forward(inputs):
            return model(input_ids=inputs, use_cache=False, return_dict=True).logits
    from full_compile import install
    forward = install(model, forward, paths)
    return forward, paths


def build_forward(model):
    feature_forward,paths=build_feature_forward(model)
    spec=os.environ.get('Q38_PAPER_SPEC')
    if spec:
        from paper_entry_runtime import wrap_feature_forward
        return wrap_feature_forward(model,feature_forward,paths,spec)
    def raw(inputs):
        return feature_forward(inputs)[0]
    paths['feature_capture']=dict(own_final_norm_fp8_rows=True,head_frame_retention_fix=True)
    return raw,paths
