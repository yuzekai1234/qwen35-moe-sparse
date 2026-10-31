"""What does a retained frame of the chunked forward keep alive?

Some bench processes kept the frame of the first chunked_prefill.forward call alive, and
with it that call's logits and last FP8 head rows. This check forces the same situation:
the compiled trunk is wrapped so that its first call stores its caller's frame (the
frame of forward). Two forwards are run; after each, the logits are deleted and the
allocated CUDA memory is printed. A forward whose frame holds no large tensors returns to
the memory allocated before the forward.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

RETAINED = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    sys.path.insert(0, str(args.eval_dir.resolve()))
    from transformers import AutoModelForCausalLM
    import eval as evaluation
    from runtime import build_forward

    device = torch.device('cuda', 0)
    tokens = np.memmap(args.packed / 'tokens.bin', dtype=np.int32, mode='r')
    is_text = np.memmap(args.packed / 'is_text.bin', dtype=np.bool_, mode='r')
    model = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa', local_files_only=True)
    model.to(device).eval()
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id
    forward, _ = build_forward(model)
    cell = dict(zip(forward.__code__.co_freevars, forward.__closure__))['compiled']
    original = cell.cell_contents

    def retaining(ids):
        if not RETAINED:
            RETAINED.append(sys._getframe(1))
        return original(ids)

    cell.cell_contents = retaining
    inputs, _ = evaluation.packed_batch(tokens, is_text, 0, args.batch, 4096, pad)
    inputs = torch.from_numpy(inputs).to(device)
    with torch.inference_mode():
        for index in range(2):
            before = torch.cuda.memory_allocated(device)
            logits = forward(inputs); del logits
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated(device)
            print(f'forward {index}: allocated before {before / 2**30:.2f} GiB, after del {after / 2**30:.2f} GiB; '
                  f'retained frame locals {sorted(RETAINED[0].f_locals)}', flush=True)


if __name__ == '__main__':
    main()
