"""A shared, stateless sequence-block stream for compute-matched experiments."""
import hashlib

import numpy as np

SAMPLER_VERSION = "pcg64-shuffled-disjoint-sequence-blocks-v1"


class ShuffledBlockSampler:
    """One permutation, no epoch wrapping; position derives from global step.

    A block predicts seq targets using seq+1 adjacent corpus tokens. Adjacent
    source blocks share one context token, but their prediction targets do not
    overlap. All ranks and micro-batch sizes partition the same global stream.
    """

    def __init__(self, length, seq, seed):
        if length <= seq or seq < 1:
            raise ValueError("Corpus must contain at least seq+1 tokens")
        self.seq = seq
        self.blocks = (length - 1) // seq
        self.order = np.random.default_rng(seed).permutation(self.blocks)
        self.seed = seed
        self.sha256 = hashlib.sha256(self.order.astype('<i8', copy=False).tobytes()).hexdigest()

    def offsets(self, position, count):
        if position < 0 or count < 1 or position + count > self.blocks:
            raise ValueError("Training budget exceeds the no-replacement corpus; epoch wrapping is disabled")
        return self.order[position:position + count] * self.seq

    def micro_offsets(self, step, micro, rank, world, micro_bs, batch_tokens):
        if world < 1 or not 0 <= rank < world or micro < 0 or step < 0:
            raise ValueError("Invalid distributed sampler coordinates")
        if batch_tokens % (world * micro_bs * self.seq):
            raise ValueError("Global batch must be divisible by world*micro_bs*seq")
        accum = batch_tokens // (world * micro_bs * self.seq)
        if micro >= accum:
            raise ValueError("Micro-step exceeds gradient accumulation")
        position = step * (batch_tokens // self.seq) + (micro * world + rank) * micro_bs
        return self.offsets(position, micro_bs)

    def metadata(self):
        return dict(version=SAMPLER_VERSION, seed=self.seed, blocks=self.blocks,
                    sequence_length=self.seq, order_sha256=self.sha256, replacement=False)


def sequential_windows(length, seq, offset=0, targets=0):
    """Disjoint target windows, including a shorter final window."""
    if seq < 1 or offset < 0 or offset >= length - 1 or targets < 0:
        raise ValueError("Invalid validation window")
    count = length - offset - 1 if not targets else targets
    if count > length - offset - 1:
        raise ValueError("Requested validation targets exceed the available partition")
    return [(start, min(seq, offset + count - start)) for start in range(offset, offset + count, seq)]
