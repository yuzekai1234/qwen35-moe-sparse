"""Launch one of the five original baselines, with explicit CLI overrides."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("q35_c1_s1", "q35_c1_s2", "q35_c1_s3", "q35_c2_m", "q35_c2_b")


def command(name, overrides, world):
    saved = json.loads((ROOT / "run_configs" / name / "run_config.json").read_text())
    # Model/run fields only: never pass archived parameter counts or old host paths.
    keys = ("layers hidden heads kv_heads head_dim lin_k_heads lin_v_heads lin_head_dim "
            "experts topk moe_ffn shared_ffn layer_types aux_coef rope_theta partial_rotary "
            "experts_impl seq tokens batch_tokens micro_bs lr min_lr_ratio warmup wd clip seed "
            "log_every eval_every eval_tokens ckpt_every resume max_steps").split()
    values = {key: saved[key] for key in keys}
    values.update(run_dir=str(ROOT / "runs" / name),
                  data=os.environ.get("DATA", str(ROOT / "data/corpus_exaone154k/train.bin")),
                  tokenizer=os.environ.get("TOK", str(ROOT / "tokenizer/exaone_154k")))
    cmd = [sys.executable]
    if world > 1:
        cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}"]
    cmd += [str(ROOT / "code/train_q35.py")]
    for key, value in values.items():
        cmd += [f"--{key}", str(value)]
    return cmd + overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=NAMES)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.nproc < 1:
        parser.error("--nproc must be positive")
    cmd = command(args.preset, overrides, args.nproc)
    import shlex
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
