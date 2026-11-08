"""Process-local compatibility fixes for the pinned evaluation environment."""
import functools
import torch


def install_flex_compat():
    """Validate 64-token sparse blocks against compatible tiles in Torch 2.8.

    Torch 2.8 checks its default tile before applying the caller's BLOCK_M/N.
    Q38 supplies 64x64, but the SM90 default is 128x128. Restrict the default
    candidates for this model's 64-wide BF16 heads; preserve every other case.
    """
    if not torch.__version__.startswith("2.8."):
        return {"torch": torch.__version__, "flex_compat": "not_needed_or_unqualified"}
    from torch._inductor import lowering  # initialize before importing heuristics
    from torch._inductor.template_heuristics import CUDAConfigHeuristic, FlexConfig
    original = CUDAConfigHeuristic.get_flex_attn_fwd_configs
    if not getattr(original, "_q38_64_block_compat", False):
        @functools.wraps(original)
        def compatible(self, head_dim, dtype):
            candidates = original(self, head_dim, dtype)
            if head_dim == 64 and dtype == torch.bfloat16:
                valid = [c for c in candidates if 64 % c.block_m == 0 and 64 % c.block_n == 0]
                return valid or [FlexConfig(64, 64, 3, 4)]
            return candidates
        compatible._q38_64_block_compat = True
        CUDAConfigHeuristic.get_flex_attn_fwd_configs = compatible
    return {"torch": torch.__version__, "flex_compat": "64_token_blocks"}
