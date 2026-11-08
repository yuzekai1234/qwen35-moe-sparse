"""Torch 2.8 Hopper grouped GEMM with aligned, nonempty reduction groups.

Padding is zero and removed before downstream math. This avoids zero-group and
unaligned-K backward paths that failed the bounded native qualification.
"""
import torch


def aligned_grouped_mm(x, weight, offs):
    counts = torch.diff(offs, prepend=offs.new_zeros(1))
    padded_counts = ((counts + 7) // 8 * 8).clamp_min(8)
    padded_ends = padded_counts.cumsum(0, dtype=torch.int32)
    group = torch.repeat_interleave(torch.arange(len(counts), device=x.device), counts,
                                    output_size=len(x))
    starts = offs - counts
    padded_starts = padded_ends - padded_counts
    destinations = torch.arange(len(x), device=x.device) + (padded_starts - starts)[group]
    padded = x.new_zeros((int(padded_ends[-1]), x.shape[1]))
    padded = padded.index_copy(0, destinations, x)
    # Torch2.8 rejects the 1024-group boundary. Split only the expert batch;
    # no token, selected expert or reduction is removed.
    group_limit=512
    indices=list(range(group_limit-1,len(counts),group_limit))
    boundaries=padded_ends[indices].tolist() if indices else []
    if len(counts)%group_limit:boundaries.append(int(padded_ends[-1]))
    outputs=[]
    lo=0
    for group_lo,hi in zip(range(0,len(counts),group_limit),boundaries):
        group_hi=min(len(counts),group_lo+group_limit)
        outputs.append(torch._grouped_mm(padded[lo:hi],weight[group_lo:group_hi],offs=padded_ends[group_lo:group_hi]-lo))
        lo=hi
    out=torch.cat(outputs,0)
    return out[destinations]
