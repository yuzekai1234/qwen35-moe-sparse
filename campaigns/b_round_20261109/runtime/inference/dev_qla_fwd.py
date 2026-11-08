# Copyright (c) 2026 The Qwen team, Alibaba Group. MIT License.
# Local wrapper experiment; frozen installed backend sources are unchanged.
import os
import torch
import tilelang
from flash_qla.utils import prepare_chunk_offsets
from flash_qla.ops.gated_delta_rule.chunk.hopper.fused_fwd import tilelang_fused_chunk_gdr_fwd,TARGET_NUM_CTAS
def fused_gdr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    output_o: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    cp_seq_map: torch.LongTensor | None = None,
    raw_cu_seqlens: torch.LongTensor | None = None,
    state_v_first: bool = False,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    chunk_size = a.shape[-1]
    scale = scale or K ** (-0.5)
    assert K == V and K in (64, 128)
    assert chunk_size == 64
    output_final_state = output_final_state or False
    output_h = output_h or False
    output_o = output_o if output_o is not None else True

    if cu_seqlens is None:
        real_batch_size = batch_size
        num_chunks = tilelang.cdiv(num_tokens, chunk_size) if output_h else 0
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        chunk_offsets = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        seqlen_dtype = torch.int32
        is_varlen = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        chunk_offsets, num_chunks = prepare_chunk_offsets(cu_seqlens, chunk_size)
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        num_chunks = num_chunks if output_h else 0
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    if cp_seq_map is None:
        cp_seq_map = torch.empty(
            (real_batch_size,), dtype=seqlen_dtype, device=k.device
        )
        is_cp = False
    else:
        is_cp = True

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (real_batch_size, H, V, K)
            if state_v_first
            else (real_batch_size, H, K, V),
            dtype=torch.float32,
            device=k.device,
        )
    h = torch.empty(
        (batch_size, num_chunks, H, V, K)
        if state_v_first
        else (batch_size, num_chunks, H, K, V),
        dtype=k.dtype,
        device=k.device,
    )
    if raw_cu_seqlens is None:
        raw_cu_seqlens = torch.empty(
            (real_batch_size + 1,), dtype=seqlen_dtype, device=k.device
        )
        final_state = torch.empty(
            (real_batch_size, H, V, K)
            if state_v_first
            else (real_batch_size, H, K, V),
            dtype=torch.float32,
            device=k.device,
        )
    else:
        final_state = torch.empty(
            (raw_cu_seqlens.shape[0] - 1, H, V, K)
            if state_v_first
            else (raw_cu_seqlens.shape[0] - 1, H, K, V),
            dtype=torch.float32,
            device=k.device,
        )
    o = torch.empty_like(v)

    grid_size = real_batch_size * H
    if grid_size >= TARGET_NUM_CTAS:
        block_DV = 128
    elif grid_size * 2 >= TARGET_NUM_CTAS:
        block_DV = 64
    else:
        block_DV = 32

    block_DV = int(os.environ.get("Q38_DEV_CORE_DV",str(min(block_DV,V))))
    assert block_DV in (32,64) and block_DV <= V

    tilelang_fused_chunk_gdr_fwd_kernel = tilelang_fused_chunk_gdr_fwd(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        h_dtype=h.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        store_h=output_h,
        store_o=output_o,
        is_varlen=is_varlen,
        is_cp=is_cp,
        state_v_first=state_v_first,
        block_DV=block_DV,
    )
    tilelang_fused_chunk_gdr_fwd_kernel(
        q,
        k,
        v,
        a,
        g,
        b,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        cp_seq_map,
        raw_cu_seqlens,
        o,
        h,
        final_state,
    )

    if not output_final_state:
        final_state = None
    if not output_h:
        h = None
    if not output_o:
        o = None

    return o, h, final_state
