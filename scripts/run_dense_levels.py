#!/usr/bin/env python3
"""Run the current dense C1-C4 presets sequentially on eight GPUs; dry-run by default."""
import argparse
import copy
import csv
import hashlib
import json
import importlib.metadata
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.q38_isoflops_contract import (
    FAMILY, TOKENIZER_SHA256, VALIDATION_OFFSET, VALIDATION_TARGETS,
    runtime_identity_matches, validate_model_family,
)

LEVELS = ("c1", "c2", "c3", "c4")
FIT_SOURCE = ROOT / "docs/experiments/q38_combined_isoflops_results.json"
EVALUATION_SHA256 = "4689036c6d9b413b03b68f0042509e0f3ea7b4d98cb210da4ef5b157a18c9f5c"
EVALUATION_TOKENS = 120142976
EVALUATION_BYTES = 445910984


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.chmod(0o644)
    temporary.replace(path)


def share_outputs(root):
    """Exporters may force mode 0600, independently of the inherited umask."""
    for directory, _, files in os.walk(root):
        path = Path(directory)
        if not path.is_symlink():
            path.chmod(0o755)
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(0o644)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    require(not dirty, "Source checkout must be clean")
    return commit


def parse_gpus(value):
    devices = value.split(",")
    require(len(devices) == 8 and all(x.isdecimal() for x in devices)
            and len(set(map(int, devices))) == 8, "Exactly eight distinct GPU indices are required")
    return [str(int(x)) for x in devices]


def make_plan(run_root, evaluation_jsonl, gpus, expected_commit):
    require(run_root.is_absolute() and not run_root.exists(), "Use a new absolute --run-root")
    # This compares every canonical preset with the recorded current-graph fit.
    subprocess.run([sys.executable, str(ROOT / "scripts/make_dense_scaling_configs.py"), "--check"],
                   cwd=ROOT, env=child_environment([]), check=True, stdout=subprocess.PIPE, timeout=180)
    fit = read_json(FIT_SOURCE)
    require(fit.get("family") == FAMILY, "Recorded fit has a different model family")
    fit_source = dict(path=str(FIT_SOURCE), sha256=sha256(FIT_SOURCE), training_source_commit=fit["source_commit"])
    jobs = []
    for level in LEVELS:
        path = ROOT / "configs" / "dense_isoflops" / f"{level}.json"
        config = read_json(path)
        validate_model_family(config)
        train = config["train"]
        require(train["seq"] == 4096 and train["max_steps"] == 0, f"{level}: changed training horizon")
        require(train["tokens"] % train["batch_tokens"] == 0, f"{level}: incomplete global batch")
        require(train["batch_tokens"] % (8 * train["seq"] * train["micro_bs"]) == 0,
                f"{level}: batch is not divisible by eight ranks")
        runtime = copy.deepcopy(config)
        runtime["train"].update(run_dir=str(run_root / level / "train"), resume=0, max_steps=0)
        runtime_path = run_root / level / "config.json"
        train_command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                         "--nproc_per_node=8", "-m", "q38.train", "--config", str(runtime_path)]
        evaluation_command = [sys.executable, "-m", "q38.evaluate_single_gpu", "--dataset-path",
                              str(evaluation_jsonl), "--model-path", str(run_root / level / "train" / "hf"),
                              "--batch-size", "16", "--sequence-length", "4096", "--output",
                              str(run_root / level / "evaluation.json")]
        jobs.append(dict(level=level, canonical_config=str(path), canonical_sha256=sha256(path),
                         runtime_config=runtime, runtime_path=str(runtime_path),
                         train_command=train_command, evaluation_command=evaluation_command,
                         expected_steps=train["tokens"] // train["batch_tokens"]))
    return dict(schema_version=1, family=FAMILY, source_commit=expected_commit, run_root=str(run_root),
                gpus=gpus, evaluation_gpu=gpus[0], fit_source=fit_source,
                data_identity=fit["data_identity"], tokenizer=fit["tokenizer"],
                evaluation=dict(path=str(evaluation_jsonl), sha256=EVALUATION_SHA256,
                                scored_tokens=EVALUATION_TOKENS, text_bytes=EVALUATION_BYTES,
                                gpu_count=1, batch_size=16, sequence_length=4096), jobs=jobs)


def check_inputs(plan, *, hash_evaluation=False):
    require(source_identity() == plan["source_commit"], "Source commit changed")
    for name in ("train", "validation"):
        item = plan["data_identity"][name]
        stat = Path(item["path"]).stat()
        require((stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"]),
                f"Data snapshot changed: {name}")
    require(sha256(Path(plan["tokenizer"]) / "tokenizer.json") == TOKENIZER_SHA256, "Tokenizer changed")
    if hash_evaluation:
        before = Path(plan["evaluation"]["path"]).stat()
        require(sha256(plan["evaluation"]["path"]) == EVALUATION_SHA256, "Not the complete reference A dataset")
        after = Path(plan["evaluation"]["path"]).stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "Evaluation dataset changed during hashing")
        plan["evaluation"]["file_bytes"] = after.st_size
        plan["evaluation"]["mtime_ns"] = after.st_mtime_ns
    else:
        stat = Path(plan["evaluation"]["path"]).stat()
        require((stat.st_size, stat.st_mtime_ns) == (plan["evaluation"]["file_bytes"],
                                                  plan["evaluation"]["mtime_ns"]), "Evaluation dataset changed")
    for job in plan["jobs"]:
        require(sha256(job["canonical_config"]) == job["canonical_sha256"], "Canonical preset changed")
        if Path(job["runtime_path"]).exists():
            require(read_json(job["runtime_path"]) == job["runtime_config"], "Runtime configuration changed")


def gpu_snapshot(devices):
    rows = []
    for gpu in devices:
        def query(fields, subject):
            return subprocess.check_output(["nvidia-smi", "--id=" + gpu, "--query-" + subject + "=" + fields,
                                            "--format=csv,noheader,nounits"], text=True, timeout=30).strip()
        values = [x.strip() for x in query("index,uuid,name,memory.used,memory.total,utilization.gpu", "gpu").split(",")]
        pids = query("pid", "compute-apps").splitlines()
        rows.append(dict(index=int(values[0]), uuid=values[1], name=values[2],
                         memory_used_mib=int(values[3]), memory_total_mib=int(values[4]),
                         utilization_percent=int(values[5]), compute_pids=pids))
    return rows


def wait_for_gpus(devices, timeout, record):
    deadline = time.monotonic() + timeout
    while True:
        rows = gpu_snapshot(devices)
        record(rows)
        if all(not row["compute_pids"] and row["memory_used_mib"] < 1024 for row in rows):
            return rows
        require(time.monotonic() < deadline, "GPU availability wait expired; no other processes were stopped")
        time.sleep(min(15, max(0, deadline - time.monotonic())))


def child_environment(devices):
    env = {key: value for key, value in os.environ.items()
           if "proxy" not in key.lower() and key not in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")}
    env.update(CUDA_VISIBLE_DEVICES=",".join(devices), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def environment_record():
    packages = {}
    for name in ("torch", "transformers", "triton", "flash-linear-attention", "liger-kernel", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    names = ("OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR")
    return dict(python=sys.executable, python_version=sys.version, packages=packages,
                environment={key: os.environ[key] for key in names if key in os.environ})


def capture_owned_tree(proc, handles):
    """Keep stable handles while the unreaped parent still establishes ownership."""
    if proc.poll() is not None:
        return
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            tail = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            parents[int(entry.name)] = (int(tail[1]), tail[19])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    owned = {proc.pid}
    while True:
        expanded = owned | {pid for pid, (parent, _) in parents.items() if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    for pid in owned:
        if pid not in parents:
            continue
        identity = (pid, parents[pid][1])
        if identity in handles:
            continue
        handle = None
        try:
            handle = os.pidfd_open(pid)
            entry = Path("/proc") / str(pid)
            tail = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            if entry.stat().st_uid == os.getuid() and tail[19] == identity[1]:
                handles[identity] = handle
                handle = None
        except (FileNotFoundError, ProcessLookupError):
            pass
        finally:
            if handle is not None:
                os.close(handle)


def stop_owned_tree(proc, handles, grace=60):
    """Never signal a shared process group, a recycled PID, or a GPU-wide match."""
    capture_owned_tree(proc, handles)
    for handle in handles.values():
        try:
            signal.pidfd_send_signal(handle, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Exiting torchrun does not imply its rank workers have exited.
    deadline = time.monotonic() + grace
    import select
    descriptors = list(handles.values())
    pending = descriptors[:]
    while pending and time.monotonic() < deadline:
        readable, _, _ = select.select(pending, [], [], min(1, max(0, deadline - time.monotonic())))
        pending = [handle for handle in pending if handle not in readable]
    for handle in pending:
        try:
            signal.pidfd_send_signal(handle, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=30)


def run_command(command, devices, log_path, timeout, heartbeat):
    env = child_environment(devices)
    with log_path.open("x") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        started = time.monotonic()
        handles = {}
        try:
            heartbeat(proc.pid)
            while proc.poll() is None:
                capture_owned_tree(proc, handles)
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Stage exceeded {timeout}s; see {log_path}")
                try:
                    proc.wait(timeout=min(60, remaining))
                except subprocess.TimeoutExpired:
                    heartbeat(proc.pid)
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, command)
        except BaseException:
            stop_owned_tree(proc, handles)
            raise
        finally:
            for handle in handles.values():
                os.close(handle)


def validate_hf(path, config):
    hf = read_json(path / "config.json")
    hf = hf.get("text_config", hf)
    model = config["model"]
    expected = dict(num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0,
                    shared_expert_intermediate_size=0, mhc_enabled=False, qsa_mode="full", mtp=True,
                    hidden_size=model["hidden"], num_hidden_layers=model["layers"],
                    intermediate_size=model["dense_ffn"], ngram_rows=model["ngram_rows"])
    require(all(hf.get(key) == value for key, value in expected.items()), "HF export has a different architecture")
    require(hf.get("layer_types") == config["accounting"]["layer_types"], "HF layer types changed; preserve GDN")
    require(sha256(path / "tokenizer.json") == TOKENIZER_SHA256, "Exported tokenizer changed")
    weights = sorted(path.glob("*.safetensors"))
    require(bool(weights), "HF weights are missing")
    if (path / "model.safetensors.index.json").exists():
        expected_names = set(read_json(path / "model.safetensors.index.json")["weight_map"].values())
        require(expected_names == {p.name for p in weights}, "HF weight shards are incomplete")
    for weight in weights:
        with weight.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            require(0 < size < 100 * 1024 * 1024, "Invalid safetensors header")
            tensors = json.loads(stream.read(size))
        require(all(item["dtype"] == "BF16" for name, item in tensors.items() if name != "__metadata__"),
                "HF must contain BF16 weights, without quantization")
    return {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
            for p in [path / "config.json", path / "tokenizer.json", *weights]}


def validate_training(job, plan):
    config = job["runtime_config"]
    train, accounting = config["train"], config["accounting"]
    root = Path(train["run_dir"])
    rc, summary, final, flops = [read_json(root / filename) for filename in
                               ("run_config.json", "summary.json", "final_eval.json", "flops.json")]
    require(rc["config_sha256"] == sha256(job["runtime_path"]), "Runtime config identity differs")
    validate_model_family(dict(model=rc, train=rc))
    require(all(rc.get(key) == value for key, value in {**config["model"], **train}.items()),
            "Actual training arguments differ from the runtime configuration")
    require(runtime_identity_matches(rc.get("data_identity"), plan["data_identity"])
            and rc.get("tokenizer_sha256") == TOKENIZER_SHA256, "Training data identity differs")
    require(rc["compute"] == config["compute"] and rc["steps"] == job["expected_steps"]
            and rc["accum"] == train["batch_tokens"] // (8 * 4096 * train["micro_bs"]),
            "Training schedule/world differs")
    expected_flops = train["tokens"] * accounting["train_flops_per_token"]
    require(summary.get("world") == 8 and summary.get("completed_budget") is True
            and final.get("completed_budget") is True, "Eight-GPU full training budget did not complete")
    for record in (summary, final):
        require(record.get("tokens_trained") == train["tokens"] and record.get("train_flops") == expected_flops,
                "Training token/FLOPs budget differs")
        require(record.get("flops_accounting") == accounting["accounting_method"], "Training accounting differs")
    require(flops.get("train_flops") == expected_flops and flops.get("train_tokens") == train["tokens"]
            and flops.get("train_flops_per_token") == accounting["train_flops_per_token"], "FLOPs export differs")
    require(final.get("metric") == "pure_cross_entropy" and final.get("offset") == VALIDATION_OFFSET
            and final.get("scored_tokens") == VALIDATION_TARGETS and final.get("tokenizer_sha256") == TOKENIZER_SHA256,
            "Fixed final validation partition/metric differs")
    require(math.isfinite(final["loss"]) and final["loss"] > 0
            and summary["final_eval_loss"] == final["loss"], "Final validation loss is invalid")
    with (root / "training_metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    require(bool(metrics) and int(metrics[0]["step"]) == 0
            and int(metrics[-1]["step"]) == job["expected_steps"] - 1
            and int(metrics[-1]["tokens"]) == train["tokens"], "Training metrics do not reach the full horizon")
    require(all(math.isfinite(float(row[key])) for row in metrics for key in ("loss", "grad_norm")),
            "Non-finite training loss or gradient")
    require((root / "ckpt.pt").stat().st_size > 0, "Final FP32 master checkpoint is missing")
    artifacts = validate_hf(root / "hf", config)
    return dict(tokens_trained=train["tokens"], steps=job["expected_steps"], train_flops=expected_flops,
                final_validation_ce=final["loss"], train_elapsed_s=summary["elapsed_s"], hf_artifacts=artifacts)


def validate_evaluation(job, result):
    expected = dict(gpu_count=1, batch_size=16, sequence_length=4096,
                    scored_tokens=EVALUATION_TOKENS, text_bytes=EVALUATION_BYTES, timing_enabled=True,
                    active_params_excluding_embeddings=job["runtime_config"]["accounting"]["active_params_excluding_embeddings"])
    require(all(result.get(key) == value for key, value in expected.items()), "Full-A evaluation contract differs")
    require(result.get("model_path") == str(Path(job["runtime_config"]["train"]["run_dir"]) / "hf"),
            "Evaluation used a different checkpoint")
    require(all(math.isfinite(result[key]) and result[key] > 0
                for key in ("bpb", "forward_seconds", "evaltime", "total_nll_nats")), "Invalid evaluation metrics")
    calculated = result["forward_seconds"] / ((expected["active_params_excluding_embeddings"] / 1e9) * (EVALUATION_TOKENS / 1e6))
    require(math.isclose(result["evaltime"], calculated, rel_tol=1e-10)
            and math.isclose(result["bpb"], result["total_nll_nats"] / (math.log(2) * EVALUATION_BYTES), rel_tol=1e-10),
            "Evaluation metric formula differs")
    return result


class Campaign:
    def __init__(self, plan, train_timeout, evaluation_timeout, resource_timeout):
        self.plan = plan
        self.root = Path(plan["run_root"])
        self.train_timeout, self.evaluation_timeout = train_timeout, evaluation_timeout
        self.resource_timeout = resource_timeout
        self.state = dict(schema_version=1, family=FAMILY, source_commit=plan["source_commit"],
                          status="running", started_utc=utc(), evaluation=plan["evaluation"],
                          jobs=[dict(level=level, status="pending") for level in LEVELS])

    def save(self):
        self.state["updated_utc"] = utc()
        atomic_json(self.root / "summary.json", self.state)

    def execute(self):
        self.root.mkdir(mode=0o755)  # Exclusive claim: never resume or overwrite an existing campaign.
        current = None
        try:
            self.save()
            check_inputs(self.plan, hash_evaluation=True)
            atomic_json(self.root / "environment.json", environment_record())
            atomic_json(self.root / "plan.json", self.plan)
            for job, current in zip(self.plan["jobs"], self.state["jobs"]):
                level_root = self.root / job["level"]
                level_root.mkdir(mode=0o755)
                atomic_json(job["runtime_path"], job["runtime_config"])
                current.update(started_utc=utc(), run_dir=job["runtime_config"]["train"]["run_dir"],
                               canonical_sha256=job["canonical_sha256"], runtime_sha256=sha256(job["runtime_path"]),
                               train_command=job["train_command"], evaluation_command=job["evaluation_command"])
                for stage, devices, command, timeout in (
                    ("training", self.plan["gpus"], job["train_command"], self.train_timeout),
                    ("evaluation", [self.plan["evaluation_gpu"]], job["evaluation_command"], self.evaluation_timeout),
                ):
                    current["status"] = "waiting_for_" + stage
                    self.save()
                    check_inputs(self.plan)
                    def resource_record(rows):
                        current["resource_snapshot"] = rows
                        self.save()
                    wait_for_gpus(devices, self.resource_timeout, resource_record)
                    current["status"] = stage
                    self.save()
                    def heartbeat(pid):
                        current.update(child_pid=pid, heartbeat_utc=utc())
                        self.save()
                    run_command(command, devices, level_root / (stage + ".log"), timeout, heartbeat)
                    share_outputs(level_root)
                    current.pop("child_pid", None)
                    if stage == "training":
                        current["training"] = validate_training(job, self.plan)
                        atomic_json(level_root / "training-validation.json", current["training"])
                    else:
                        current["evaluation"] = validate_evaluation(job, read_json(level_root / "evaluation.json"))
                current.update(status="completed", finished_utc=utc())
                self.save()
            self.state["status"] = "completed"
        except BaseException as error:
            self.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            if current is not None:
                current.update(status="failed", error=self.state["error"])
            raise
        finally:
            self.state["finished_utc"] = utc()
            self.save()
            share_outputs(self.root)
            atomic_json(self.root / "completion.json", self.state)
        return self.state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evaluation-jsonl", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--train-timeout", type=int, default=12 * 3600, help="Per level, seconds")
    parser.add_argument("--evaluation-timeout", type=int, default=3 * 3600, help="Per level, seconds")
    parser.add_argument("--resource-timeout", type=int, default=1800, help="Per stage GPU wait, seconds")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(min(args.train_timeout, args.evaluation_timeout, args.resource_timeout) > 0, "Timeouts must be positive")
    require(args.evaluation_jsonl.is_absolute(), "Use an absolute --evaluation-jsonl")
    plan = make_plan(args.run_root, args.evaluation_jsonl, parse_gpus(args.gpus), args.expected_commit)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    os.umask(0o022)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Campaign(plan, args.train_timeout, args.evaluation_timeout, args.resource_timeout).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
