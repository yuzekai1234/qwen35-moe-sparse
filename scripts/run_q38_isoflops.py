#!/usr/bin/env python3
"""按 manifest 逐卡运行独立 Q38 训练任务；默认仅显示计划。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.q38_isoflops_contract import (
    BUDGETS, FAMILY, RUNTIME_OVERRIDES, VALIDATION_OFFSET, VALIDATION_TARGETS,
    runtime_identity_matches, validate_config_data_paths, validate_data_identity, validate_fit_grid, validate_model_family,
)
HEARTBEAT = 600
JOB_TIMEOUT = 4 * 3600
QUEUE_TIMEOUT = 6 * 3600
TERM_GRACE = 30


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_snapshot_files(snapshot):
    """Check only known file metadata; this is not a second full-file hash."""
    identity = validate_data_identity(snapshot)
    for name in ('train', 'validation'):
        expected = identity[name]
        stat = Path(expected['path']).stat()
        if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'Data snapshot no longer matches {name}: {expected["path"]}')


def git_source():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            text=True, stdout=subprocess.PIPE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                           text=True, stdout=subprocess.PIPE).stdout.splitlines()
    return commit, dirty


def gpu_idle(gpu):
    def query(args):
        return subprocess.run(["nvidia-smi", *args], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout.strip()
    mem = query(["--id=" + str(gpu), "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    pids = query(["--id=" + str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    cols = [x.strip() for x in mem.split(",")]
    if len(cols) != 3:
        raise RuntimeError(f"GPU {gpu}: 无法解析 nvidia-smi 指标: {mem!r}")
    active_pids = [x.strip() for x in pids.splitlines() if x.strip() and x.strip() != "[Not Supported]"]
    return int(cols[0]) == 0 and not active_pids, {
        "memory_used_mib": int(cols[0]), "memory_total_mib": int(cols[1]),
        "utilization_percent": int(cols[2]), "compute_pids": active_pids,
    }


def latest_metrics(run_dir):
    path = run_dir / "training_metrics.csv"
    if not path.is_file():
        return None
    try:
        with path.open(newline="") as f:
            last = None
            for last in csv.DictReader(f):
                pass
        if last is None:
            return None
        return {key: last.get(key) for key in ("step", "tokens", "loss", "grad_norm")}
    except (OSError, csv.Error):
        return None


def resolved_repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_manifest(path):
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or not isinstance(data.get("jobs"), list) or not data["jobs"]:
        raise ValueError("manifest 必须含 schema_version=1 和非空 jobs 数组")
    if data.get('family') != FAMILY or data.get('fit_budgets') != BUDGETS:
        raise ValueError('manifest must identify the current no-mHC/full-attention BF16 family')
    identity = validate_data_identity(data.get('data_identity'))
    validate_fit_grid((j.get('compute'), j.get('hidden')) for j in data['jobs'])
    top_source_commit = data.get("source_commit")
    if not isinstance(top_source_commit, str) or not top_source_commit:
        raise ValueError('manifest must pin source_commit')
    if not isinstance(data.get('model_files_sha256'), dict) or not data['model_files_sha256']:
        raise ValueError('manifest must bind the model source hashes')
    validation = data.get("validation", {})
    if not isinstance(validation, dict):
        raise ValueError("manifest.validation 必须是对象")
    if validation != dict(offset=VALIDATION_OFFSET, targets=VALIDATION_TARGETS,
                          tokenizer_sha256=identity['tokenizer_sha256'],
                          validation_sha256=identity['validation']['sha256']):
        raise ValueError('manifest final validation partition or identity changed')
    top_validation_sha = validation.get("validation_sha256")
    top_tokenizer_sha = validation.get("tokenizer_sha256")
    ids, jobs = set(), []
    for raw in data["jobs"]:
        required = ("id", "config", "run_dir", "compute", "expected_tokens", "expected_steps", "train_flops_per_token", "Na")
        if any(key not in raw for key in required):
            raise ValueError(f"任务缺少字段: {raw.get('id')}")
        jid = str(raw["id"])
        if jid in ids:
            raise ValueError(f"重复任务 id: {jid}")
        ids.add(jid)
        if top_source_commit and raw.get("source_commit") not in (None, top_source_commit):
            raise ValueError(f"{jid}: job source_commit 与 manifest 顶层值冲突")
        if top_validation_sha and raw.get("validation_sha256") not in (None, top_validation_sha):
            raise ValueError(f"{jid}: job validation_sha256 与 manifest.validation 冲突")
        if top_tokenizer_sha and raw.get("tokenizer_sha256") not in (None, top_tokenizer_sha):
            raise ValueError(f"{jid}: job tokenizer_sha256 与 manifest.validation 冲突")
        if validation and (raw.get("final_eval_offset") != validation.get("offset") or
                            raw.get("final_eval_targets") != validation.get("targets")):
            raise ValueError(f"{jid}: final evaluation partition 与 manifest.validation 不一致")
        cfg = resolved_repo_path(raw["config"])
        run_dir = Path(raw["run_dir"]).expanduser()
        if not run_dir.is_absolute():
            raise ValueError(f"{jid}: run_dir 必须是绝对路径")
        run_dir = run_dir.resolve()
        if not cfg.is_file():
            raise ValueError(f"配置不存在: {cfg}")
        config = json.loads(cfg.read_text())
        validate_model_family(config)
        validate_config_data_paths(config, identity)
        if (config['model'].get('hidden') != raw.get('hidden') or
                BUDGETS.get(raw.get('budget')) != raw['compute']):
            raise ValueError(f'{jid}: configuration shape/budget differs from its grid point')
        train = config.get("train", {})
        accounting = config.get("accounting", {})
        if validation and (train.get("final_eval_offset") != validation.get("offset") or
                            train.get("final_eval_tokens") != validation.get("targets")):
            raise ValueError(f"{jid}: 配置中的 final evaluation partition 与 manifest.validation 不一致")
        config_run_dir = resolved_repo_path(train.get("run_dir", ""))
        if not train.get("run_dir") or config_run_dir != run_dir:
            raise ValueError(f"{jid}: 配置 run_dir 与 manifest 不一致")
        if run_dir.exists():
            raise ValueError(f"拒绝自动重跑：run_dir 已存在: {run_dir}")
        actual_sha = sha256(cfg)
        if "expected_config_sha256" in raw and raw["expected_config_sha256"] != actual_sha:
            raise ValueError(f"{jid}: 配置 SHA-256 与 manifest 不一致")
        expected_tokens = raw["expected_tokens"]
        batch_tokens = train.get("batch_tokens")
        if (config.get("compute") != raw["compute"] or train.get("tokens") != expected_tokens or
                not isinstance(batch_tokens, int) or batch_tokens <= 0 or
                expected_tokens % batch_tokens or expected_tokens // batch_tokens != raw["expected_steps"] or
                accounting.get("train_flops_per_token") != raw["train_flops_per_token"] or
                accounting.get("active_params_excluding_embeddings") != raw["Na"] or
                not accounting.get("accounting_method")):
            raise ValueError(f"{jid}: manifest 与配置中的 compute/tokens/steps/F/Na/accounting 不一致")
        counted_budget = raw['compute']*995//1000
        if (expected_tokens*raw['train_flops_per_token'] > counted_budget or
                (expected_tokens+batch_tokens)*raw['train_flops_per_token'] <= counted_budget or
                expected_tokens > identity['train']['bytes']//4//train['seq']*train['seq']):
            raise ValueError(f'{jid}: incomplete full-batch budget or repeated training corpus')
        if raw.get("accounting_method") and raw["accounting_method"] != accounting["accounting_method"]:
            raise ValueError(f"{jid}: accounting_method 与配置不一致")
        jobs.append({**raw, "id": jid, "source_commit": raw.get("source_commit") or top_source_commit,
                     "data_identity": identity, "model_files_sha256": data.get('model_files_sha256'),
                     "validation_sha256": raw.get("validation_sha256") or top_validation_sha,
                     "tokenizer_sha256": raw.get("tokenizer_sha256") or top_tokenizer_sha,
                     "config_path": cfg, "config_data": config,
                     "run_dir_path": run_dir, "config_sha256": actual_sha,
                     "accounting_method": accounting["accounting_method"]})
    return jobs


class Runner:
    def __init__(self, jobs, gpus, python, out, job_timeout, queue_timeout, heartbeat, grace):
        self.jobs, self.gpus, self.python, self.out = jobs, gpus, python, out
        self.job_timeout, self.queue_timeout, self.heartbeat, self.grace = job_timeout, queue_timeout, heartbeat, grace
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.children = {}
        self.pending_jobs = sorted(jobs, key=lambda j: (-j["compute"], j["Na"], j["id"]))
        self.completed_workers = 0
        self.gpu_states = {gpu: {"status": "pending"} for gpu in gpus}
        self.states = {j["id"]: {"id": j["id"], "status": "pending", "config_sha256": j["config_sha256"],
                                 "run_dir": str(j["run_dir_path"])} for j in jobs}
        self.started = time.monotonic()

    def save(self):
        with self.lock:
            atomic_json(self.out / "summary.json", {"started_utc": self.started_utc, "updated_utc": utc(),
                                                     "jobs": list(self.states.values()),
                                                     "gpus": {str(gpu): state for gpu, state in self.gpu_states.items()}})

    def update(self, jid, **values):
        with self.condition:
            self.states[jid].update(values)
            self.save()
            self.condition.notify_all()

    def heartbeat_job(self, job, gpu):
        try:
            _, stats = gpu_idle(gpu)
            values = {"gpu_stats": stats}
        except Exception as e:
            values = {"gpu_query_error": str(e)}
        values.update(heartbeat_utc=utc(), latest_metrics=latest_metrics(job["run_dir_path"]))
        self.update(job["id"], **values)

    def worker(self, gpu):
        idle, stats = gpu_idle(gpu)
        with self.condition:
            self.gpu_states[gpu] = {"status": "idle" if idle else "busy", "stats": stats, "checked_utc": utc()}
            self.save()
            self.condition.notify_all()
        if not idle:
            return

        while not self.stop.is_set() and time.monotonic() - self.started < self.queue_timeout:
            idle, stats = gpu_idle(gpu)
            with self.condition:
                self.gpu_states[gpu] = {"status": "idle" if idle else "busy", "stats": stats, "checked_utc": utc()}
                if not idle or not self.pending_jobs or self.stop.is_set():
                    self.save()
                    self.condition.notify_all()
                    return
                job = self.pending_jobs.pop(0)
                self.save()
            jid = job["id"]
            proc = None
            try:
                config_raw = job['config_path'].read_bytes()
                if hashlib.sha256(config_raw).hexdigest() != job['config_sha256']:
                    raise ValueError('Configuration changed before launch')
                validate_config_data_paths(json.loads(config_raw), job['data_identity'])
                run_dir = job["run_dir_path"]
                run_dir.mkdir(parents=True, exist_ok=False)
                log = run_dir / "train.log"
                env = os.environ.copy()
                env.pop("TORCHDYNAMO_SUPPRESS_ERRORS", None)
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                env.update({"PYTHONPATH": str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                            "TRITON_LIBCUDA_PATH": str(ROOT / "libcuda"),
                            "LIBRARY_PATH": str(ROOT / "libcuda") + (os.pathsep + env["LIBRARY_PATH"] if env.get("LIBRARY_PATH") else ""),
                            "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "4",
                            "CUDA_VISIBLE_DEVICES": str(gpu)})
                argv = [str(self.python), "-m", "q38.train", "--config", str(job["config_path"])]
                commit, dirty = git_source()
                pin = job.get("source_commit")
                if pin and (commit != pin or dirty):
                    self.update(jid, status="failed", error=f"源码版本不符合 manifest：HEAD={commit}, dirty={bool(dirty)}",
                                finished_utc=utc(), gpu_id=gpu, gpu_stats=stats)
                    continue
                if self.stop.is_set() or time.monotonic() - self.started >= self.queue_timeout:
                    state = "not_started_deadline" if time.monotonic() - self.started >= self.queue_timeout else "not_started_stopped"
                    self.update(jid, status=state, finished_utc=utc(), gpu_id=gpu, gpu_stats=stats)
                    continue
                with log.open("ab", buffering=0) as lf:
                    check_snapshot_files(job['data_identity'])
                    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                    with self.condition:
                        self.children[jid] = proc
                        should_stop = self.stop.is_set()
                    if should_stop:
                        self.terminate(proc)
                        code = proc.wait()
                        self.update(jid, status="timed_out", started_utc=utc(), finished_utc=utc(),
                                    exitcode=code, timeout=True, gpu_id=gpu, gpu_stats=stats)
                        continue
                    self.update(jid, status="running", started_utc=utc(), pid=proc.pid, pgid=proc.pid, gpu_id=gpu,
                                gpu_stats=stats, argv=argv, source_commit=commit, source_dirty_summary=dirty,
                                config_sha256=job["config_sha256"],
                                environment={**{k: env[k] for k in ("PYTHONPATH", "TRITON_LIBCUDA_PATH", "LIBRARY_PATH",
                                    "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS",
                                    "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")},
                                    "TORCHDYNAMO_SUPPRESS_ERRORS_unset": "TORCHDYNAMO_SUPPRESS_ERRORS" not in env},
                                log=str(log))
                    deadline = time.monotonic() + self.job_timeout
                    timed_out = False
                    while proc.poll() is None:
                        remaining = min(self.heartbeat, deadline - time.monotonic(),
                                        self.queue_timeout - (time.monotonic() - self.started))
                        if remaining <= 0 or self.stop.is_set():
                            timed_out = True
                            self.terminate(proc)
                            break
                        try:
                            proc.wait(timeout=remaining)
                        except subprocess.TimeoutExpired:
                            if proc.poll() is None:
                                self.heartbeat_job(job, gpu)
                    code = proc.wait()
                status = "success" if self.valid_success(job, code) else ("timed_out" if timed_out else "incomplete")
                self.update(jid, status=status, finished_utc=utc(), exitcode=code, timeout=timed_out,
                            latest_metrics=latest_metrics(job["run_dir_path"]))
            except Exception as e:
                if proc is not None:
                    self.terminate(proc)
                self.update(jid, status="failed", error=f"{type(e).__name__}: {e}", finished_utc=utc())
            finally:
                if proc is not None:
                    with self.lock:
                        self.children.pop(jid, None)

    def worker_entry(self, gpu):
        try:
            self.worker(gpu)
        except Exception as e:
            with self.condition:
                self.gpu_states[gpu] = {"status": "failed", "error": f"{type(e).__name__}: {e}", "finished_utc": utc()}
                self.save()
                self.condition.notify_all()
        finally:
            with self.condition:
                self.completed_workers += 1
                self.condition.notify_all()

    def terminate(self, proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=self.grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def valid_success(self, job, code):
        if code != 0:
            return False
        rd = job["run_dir_path"]
        config = job["config_data"]
        train = config["train"]
        accounting = config["accounting"]
        try:
            summary = json.loads((rd / "summary.json").read_text())
            ev = json.loads((rd / "final_eval.json").read_text())
            rc = json.loads((rd / "run_config.json").read_text())
            flops = json.loads((rd / "flops.json").read_text())
            expected_flops = job["train_flops_per_token"] * job["expected_tokens"]
            expected_scored = train.get("final_eval_tokens")
            expected_offset = train.get("final_eval_offset", 0)
            expected_tokenizer = job.get("tokenizer_sha256")
            fields_match = (rc.get("run_dir") and Path(rc["run_dir"]).resolve() == rd.resolve() and
                            rc.get("compute") == config["compute"] and rc.get("tokens") == train["tokens"] and
                            rc.get("batch_tokens") == train["batch_tokens"] and rc.get("steps") == job["expected_steps"] and
                            rc.get("seq") == train.get("seq") and
                            rc.get("final_eval_tokens") == train.get("final_eval_tokens") and
                            rc.get("final_eval_offset") == train.get("final_eval_offset", 0) and
                            rc.get("flops_per_token") == accounting["train_flops_per_token"] and
                            rc.get("config_sha256") == job["config_sha256"] and
                            rc.get("flops_accounting") == accounting["accounting_method"] and
                            all(rc.get(key) == value for key, value in config["model"].items()) and
                            all(rc.get(key) == value for key,value in train.items() if key not in RUNTIME_OVERRIDES) and
                            rc.get('model_code', {}).get('files') == job['model_files_sha256'])
            tokenizer_match = (not expected_tokenizer or
                               (rc.get("tokenizer_sha256") == expected_tokenizer and ev.get("tokenizer_sha256") == expected_tokenizer))
            tokenizer_match = tokenizer_match and rc.get("tokenizer_sha256") == ev.get("tokenizer_sha256")
            validation_sha = job.get("validation_sha256")
            validation_match = (validation_sha == job['data_identity']['validation']['sha256'] and
                                runtime_identity_matches(rc.get('data_identity'), job['data_identity']))
            return (summary.get("completed_budget") is True and summary.get("tokens_trained") == job["expected_tokens"] and
                    summary.get("train_flops") == expected_flops and ev.get("tokens_trained") == job["expected_tokens"] and
                    ev.get("train_flops") == expected_flops and ev.get("completed_budget") is True and
                    ev.get("metric") == "pure_cross_entropy" and ev.get("offset") == expected_offset and
                    ev.get("scored_tokens") == expected_scored and isinstance(ev.get("loss"), (int, float)) and
                    math.isfinite(ev["loss"]) and fields_match and tokenizer_match and validation_match and
                    flops.get("active_params_excluding_embeddings") == job["Na"] and
                    flops.get("train_flops_per_token") == job["train_flops_per_token"] and
                    flops.get("accounting_method") == accounting["accounting_method"])
        except (OSError, ValueError, TypeError):
            return False

    def run(self):
        self.started_utc = utc()
        self.save()
        workers = []
        for gpu in self.gpus:
            thread = threading.Thread(target=self.worker_entry, args=(gpu,), daemon=True)
            thread.start()
            workers.append(thread)
        stop_handled = False
        while True:
            with self.condition:
                if self.completed_workers == len(workers):
                    break
                remaining = self.queue_timeout - (time.monotonic() - self.started)
                if remaining <= 0:
                    self.stop.set()
                stopping = self.stop.is_set()
                children = list(self.children.values()) if stopping and not stop_handled else []
                if stopping and not stop_handled:
                    stop_handled = True
                if not children:
                    self.condition.wait(timeout=max(0, remaining) if not stopping else None)
                    continue
            for proc in children:
                self.terminate(proc)
        for thread in workers:
            thread.join()
        for job in self.jobs:
            if self.states[job["id"]]["status"] == "pending":
                status = "not_started_deadline" if time.monotonic() - self.started >= self.queue_timeout else "not_started_no_available_gpu"
                self.update(job["id"], status=status, finished_utc=utc())
        self.save()
        return all(state["status"] == "success" for state in self.states.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gpus", required=True, help="逗号分隔的物理 GPU 编号")
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--job-timeout", type=int, default=JOB_TIMEOUT)
    parser.add_argument("--queue-timeout", type=int, default=QUEUE_TIMEOUT)
    parser.add_argument("--heartbeat", type=int, default=HEARTBEAT)
    parser.add_argument("--term-grace", type=int, default=TERM_GRACE)
    args = parser.parse_args()
    if any(value <= 0 for value in (args.job_timeout, args.queue_timeout, args.heartbeat, args.term_grace)):
        parser.error("job-timeout、queue-timeout、heartbeat 和 term-grace 必须为正数")
    jobs = load_manifest(args.manifest.resolve())
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or any(value < 0 for value in gpus):
        parser.error("GPU 列表必须是非负且不重复的编号")
    py = args.python.expanduser().absolute()
    if not py.is_file():
        raise ValueError(f"Python 不存在: {py}")
    out = args.output.resolve()
    print(json.dumps({"manifest": str(args.manifest.resolve()), "gpus": gpus, "python": str(py),
                      "jobs": [{"id": j["id"], "config": str(j["config_path"]), "run_dir": str(j["run_dir_path"])} for j in jobs],
                      "execute": args.execute}, indent=2, ensure_ascii=False))
    if not args.execute:
        return 0
    if out.exists():
        raise ValueError(f"拒绝覆盖已有控制目录: {out}")
    commit, dirty = git_source()
    pinned = {job["source_commit"] for job in jobs if job.get("source_commit")}
    if len(pinned) > 1 or dirty or (pinned and commit not in pinned):
        raise ValueError(f"源码版本不符合 manifest：HEAD={commit}, dirty={bool(dirty)}, source_commit={sorted(pinned)}")
    check_snapshot_files(jobs[0]['data_identity'])
    out.mkdir(parents=True)
    runner = Runner(jobs, gpus, py, out, args.job_timeout, args.queue_timeout, args.heartbeat, args.term_grace)

    def request_stop(_signum, _frame):
        runner.stop.set()
        with runner.condition:
            runner.condition.notify_all()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return 0 if runner.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
