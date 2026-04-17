from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.experiment_catalog import ALL_ACTIVE_DATASETS, FOCUSED_MODELS
from geomatric.experiment_paths import DEFAULT_EXPERIMENT_VERSION, ensure_version_manifest, log_dir, normalize_version, record_dir

PYTHON = "/home/xuelin/miniconda3/envs/pyg/bin/python"
FOLDS = [0, 1, 2, 3, 4]
MISSING_OPERATORS = ["GATConv", "SAGEConv", "GINConv"]
EXTERNAL_BASELINES: List[Tuple[str, str]] = [
    ("GraphSAGEBaseline", "SAGEConv"),
    ("GINBaseline", "GINConv"),
    ("JKNetBaseline", "GCNConv"),
    ("APPNPBaseline", "GCNConv"),
]

BASELINE_PROTOCOLS: Dict[str, Dict[str, object]] = {
    "PlainGNN": {
        "name": "GCNConv",
        "ep": 200,
        "patience": 60,
        "lr": 0.005,
        "weight_decay": 1e-4,
        "drop": 0.5,
        "dim": 64,
        "h_layer": 4,
    },
    "NodeResGNN": {
        "name": "GCNConv",
        "ep": 200,
        "patience": 60,
        "lr": 0.005,
        "weight_decay": 1e-4,
        "drop": 0.5,
        "dim": 64,
        "h_layer": 4,
    },
    "GraphResGNN": {
        "name": "GCNConv",
        "ep": 200,
        "patience": 60,
        "lr": 0.005,
        "weight_decay": 1e-4,
        "drop": 0.5,
        "dim": 64,
        "h_layer": 4,
    },
    "GraphSAGEBaseline": {
        "name": "SAGEConv",
        "ep": 200,
        "patience": 60,
        "lr": 0.005,
        "weight_decay": 1e-4,
        "drop": 0.5,
        "dim": 64,
        "h_layer": 4,
    },
    "GINBaseline": {
        "name": "GINConv",
        "ep": 200,
        "patience": 60,
        "lr": 0.005,
        "weight_decay": 1e-4,
        "drop": 0.5,
        "dim": 64,
        "h_layer": 4,
    },
    "JKNetBaseline": {
        "name": "GCNConv",
        "ep": 200,
        "patience": 60,
        "lr": 0.005,
        "weight_decay": 1e-4,
        "drop": 0.5,
        "dim": 64,
        "h_layer": 4,
    },
    "APPNPBaseline": {
        "name": "GCNConv",
        "ep": 200,
        "patience": 60,
        "lr": 0.005,
        "weight_decay": 1e-4,
        "drop": 0.5,
        "dim": 64,
        "h_layer": 4,
    },
}

CROSS_PROTOCOLS: Dict[Tuple[str, str], Dict[str, object]] = {
    ("PROTEINS", "NodeCrossGNN"): {
        "name": "GCNConv",
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.2,
        "dim": 64,
        "h_layer": 4,
    },
    ("PROTEINS", "GraphCrossGNN"): {
        "name": "GCNConv",
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "dim": 64,
        "h_layer": 4,
    },
    ("DD", "NodeCrossGNN"): {
        "name": "GCNConv",
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "dim": 64,
        "h_layer": 3,
    },
    ("DD", "GraphCrossGNN"): {
        "name": "GCNConv",
        "ep": 240,
        "patience": 80,
        "lr": 0.002,
        "weight_decay": 5e-5,
        "drop": 0.2,
        "dim": 64,
        "h_layer": 4,
    },
}

DEFAULT_CROSS_PROTOCOL = {
    "name": "GCNConv",
    "ep": 240,
    "patience": 80,
    "lr": 0.003,
    "weight_decay": 5e-5,
    "drop": 0.3,
    "dim": 64,
    "h_layer": 4,
}

COMMON_ARGS = {
    "batch_size": 128,
    "grad_clip": 2.0,
    "lr_factor": 0.5,
    "lr_patience": 15,
    "min_lr": 1e-5,
}


@dataclass(frozen=True)
class Job:
    category: str
    dataset: str
    model: str
    operator: str
    fold: int

    @property
    def key(self) -> Tuple[str, str, str, int]:
        return (self.dataset, self.model, self.operator, self.fold)

    @property
    def slug(self) -> str:
        return f"{self.category}_{self.dataset}_{self.model}_{self.operator}_fold{self.fold}"


@dataclass
class RunningJob:
    job: Job
    batch_size: int
    log_path: Path
    started_at: float
    process: subprocess.Popen


def log_pattern_parse(file_name: str) -> Tuple[str, str, str, int] | None:
    if not file_name.startswith("train_") or "__" not in file_name or not file_name.endswith(".json"):
        return None
    prefix = file_name.split("__", 1)[0]
    parts = prefix.split("_")
    if len(parts) < 5:
        return None
    fold_token = parts[-1]
    operator = parts[-2]
    model = parts[-3]
    dataset = "_".join(parts[1:-3])
    if not fold_token.startswith("fold"):
        return None
    try:
        fold = int(fold_token[4:])
    except ValueError:
        return None
    return (dataset, model, operator, fold)


def completed_log_keys(active_log_dir: Path) -> set[Tuple[str, str, str, int]]:
    seen: set[Tuple[str, str, str, int]] = set()
    for path in active_log_dir.iterdir():
        if not path.is_file():
            continue
        key = log_pattern_parse(path.name)
        if key is not None:
            seen.add(key)
    return seen


def build_protocol(dataset: str, model: str, operator: str) -> Dict[str, object]:
    if model in BASELINE_PROTOCOLS:
        protocol = dict(BASELINE_PROTOCOLS[model])
    elif model in {"NodeCrossGNN", "GraphCrossGNN"}:
        protocol = dict(CROSS_PROTOCOLS.get((dataset, model), DEFAULT_CROSS_PROTOCOL))
    else:
        protocol = dict(BASELINE_PROTOCOLS[model])
    protocol["name"] = operator
    return {**protocol, **COMMON_ARGS}


def all_target_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset in ALL_ACTIVE_DATASETS:
        for model, operator in EXTERNAL_BASELINES:
            for fold in FOLDS:
                jobs.append(Job(category="baseline", dataset=dataset, model=model, operator=operator, fold=fold))
        for model in FOCUSED_MODELS:
            for operator in MISSING_OPERATORS:
                for fold in FOLDS:
                    jobs.append(Job(category="operator", dataset=dataset, model=model, operator=operator, fold=fold))
    return jobs


def missing_jobs(active_log_dir: Path) -> List[Job]:
    seen = completed_log_keys(active_log_dir)
    return [job for job in all_target_jobs() if job.key not in seen]


def write_status_report(active_log_dir: Path, md_report: Path, status_json: Path, version: str) -> Dict[str, object]:
    seen = completed_log_keys(active_log_dir)
    targets = all_target_jobs()
    missing = [job for job in targets if job.key not in seen]

    baseline_targets = [job for job in targets if job.category == "baseline"]
    operator_targets = [job for job in targets if job.category == "operator"]
    baseline_missing = [job for job in missing if job.category == "baseline"]
    operator_missing = [job for job in missing if job.category == "operator"]

    per_dataset_lines: List[str] = []
    for dataset in ALL_ACTIVE_DATASETS:
        ds_baseline_targets = [job for job in baseline_targets if job.dataset == dataset]
        ds_operator_targets = [job for job in operator_targets if job.dataset == dataset]
        ds_baseline_done = sum(job.key in seen for job in ds_baseline_targets)
        ds_operator_done = sum(job.key in seen for job in ds_operator_targets)
        per_dataset_lines.append(
            f"- `{dataset}`: baseline {ds_baseline_done}/{len(ds_baseline_targets)}, "
            f"operators {ds_operator_done}/{len(ds_operator_targets)}"
        )

    model_operator_lines: List[str] = []
    for model in FOCUSED_MODELS:
        parts = []
        for operator in MISSING_OPERATORS:
            target_count = len(ALL_ACTIVE_DATASETS) * len(FOLDS)
            done_count = sum((dataset, model, operator, fold) in seen for dataset in ALL_ACTIVE_DATASETS for fold in FOLDS)
            parts.append(f"{operator} {done_count}/{target_count}")
        model_operator_lines.append(f"- `{model}`: " + ", ".join(parts))

    baseline_lines: List[str] = []
    for model, operator in EXTERNAL_BASELINES:
        target_count = len(ALL_ACTIVE_DATASETS) * len(FOLDS)
        done_count = sum((dataset, model, operator, fold) in seen for dataset in ALL_ACTIVE_DATASETS for fold in FOLDS)
        baseline_lines.append(f"- `{model}` / `{operator}`: {done_count}/{target_count}")

    next_jobs = missing[:40]
    lines = [
        "# Missing Experiment Completion",
        "",
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Version: {version}",
        f"- Target scope: {len(ALL_ACTIVE_DATASETS)} datasets, {len(FOLDS)} folds",
        f"- Baseline target: {len(baseline_targets)}, completed: {len(baseline_targets) - len(baseline_missing)}, missing: {len(baseline_missing)}",
        f"- Operator target: {len(operator_targets)}, completed: {len(operator_targets) - len(operator_missing)}, missing: {len(operator_missing)}",
        f"- Total target: {len(targets)}, completed: {len(targets) - len(missing)}, missing: {len(missing)}",
        "",
        "## Per Dataset",
        *per_dataset_lines,
        "",
        "## Baseline Coverage",
        *baseline_lines,
        "",
        "## Operator Coverage",
        *model_operator_lines,
        "",
        "## Next Pending Jobs",
    ]
    if next_jobs:
        lines.extend(
            f"- `{job.category}` `{job.dataset}` `{job.model}` `{job.operator}` fold `{job.fold}`" for job in next_jobs
        )
    else:
        lines.append("- none")

    md_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_target": len(baseline_targets),
        "baseline_completed": len(baseline_targets) - len(baseline_missing),
        "baseline_missing": len(baseline_missing),
        "operator_target": len(operator_targets),
        "operator_completed": len(operator_targets) - len(operator_missing),
        "operator_missing": len(operator_missing),
        "total_target": len(targets),
        "total_completed": len(targets) - len(missing),
        "total_missing": len(missing),
    }
    status_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def current_gpu_free_gb() -> float:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    first = proc.stdout.strip().splitlines()[0]
    return float(first) / 1024.0


def estimate_job_memory_gb(job: Job) -> float:
    memory_gb = 2.2
    if "Cross" in job.model:
        memory_gb += 0.8
    elif "Graph" in job.model:
        memory_gb += 0.4
    if job.operator == "GATConv":
        memory_gb += 0.8
    elif job.operator in {"SAGEConv", "GINConv"}:
        memory_gb += 0.3
    if job.dataset in {"DD", "Mutagenicity", "AIDS"}:
        memory_gb += 0.5
    return memory_gb


def clamp_batch(value: int) -> int:
    for candidate in [128, 96, 64, 48, 32, 24, 16, 8, 4]:
        if value >= candidate:
            return candidate
    return 4


def initial_batch_size(job: Job, reserve_gb: float, running_jobs: Sequence[RunningJob]) -> int:
    free_gb = current_gpu_free_gb()
    usable_gb = max(0.0, free_gb - reserve_gb)
    batch = 128
    if job.operator == "GATConv":
        batch = 64
    elif "Cross" in job.model:
        batch = 96

    if job.dataset == "DD" and "Cross" in job.model:
        batch = min(batch, 64)

    if job.dataset in {"MUTAG", "ENZYMES", "PROTEINS"}:
        batch += 32
    if job.dataset == "Mutagenicity":
        batch -= 32
    if len(running_jobs) == 0 and usable_gb >= 18:
        batch = int(batch * 1.5)
    elif usable_gb >= 12:
        batch = int(batch * 1.25)
    return clamp_batch(batch)


def build_command(job: Job, batch_size: int, tensorboard: bool, version: str) -> List[str]:
    protocol = build_protocol(job.dataset, job.model, job.operator)
    protocol["batch_size"] = batch_size
    cmd = [
        PYTHON,
        "geomatric/graph_classify_v3.py",
        "--mode",
        "single",
        "--ds",
        job.dataset,
        "--gname",
        job.model,
        "--name",
        job.operator,
        "--fold",
        str(job.fold),
        "--version",
        version,
    ]
    for key, value in protocol.items():
        cmd.extend([f"--{key}", str(value)])
    if tensorboard:
        cmd.append("--tensorboard")
    return cmd


def launch_job(
    job: Job,
    reserve_gb: float,
    running_jobs: Sequence[RunningJob],
    tensorboard: bool,
    version: str,
    runner_log_dir: Path,
) -> RunningJob:
    batch_size = initial_batch_size(job, reserve_gb=reserve_gb, running_jobs=running_jobs)
    cmd = build_command(job, batch_size=batch_size, tensorboard=tensorboard, version=version)
    log_path = runner_log_dir / f"{job.slug}_bs{batch_size}.log"
    handle = log_path.open("w", encoding="utf-8")
    handle.write("CMD: " + " ".join(cmd) + "\n")
    handle.flush()
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningJob(
        job=job,
        batch_size=batch_size,
        log_path=log_path,
        started_at=time.time(),
        process=process,
    )


def should_retry_with_smaller_batch(log_path: Path, batch_size: int) -> bool:
    if batch_size <= 4 or not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    oom_signals = [
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
    ]
    return any(signal in text for signal in oom_signals)


def rerun_with_smaller_batch(job: Job, previous_batch: int, tensorboard: bool, version: str, runner_log_dir: Path) -> RunningJob:
    next_batch = clamp_batch(max(4, previous_batch // 2))
    cmd = build_command(job, batch_size=next_batch, tensorboard=tensorboard, version=version)
    log_path = runner_log_dir / f"{job.slug}_retry_bs{next_batch}.log"
    handle = log_path.open("w", encoding="utf-8")
    handle.write("CMD: " + " ".join(cmd) + "\n")
    handle.flush()
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningJob(
        job=job,
        batch_size=next_batch,
        log_path=log_path,
        started_at=time.time(),
        process=process,
    )


def drain_finished_jobs(running_jobs: List[RunningJob], tensorboard: bool, version: str, runner_log_dir: Path) -> List[Job]:
    retry_jobs: List[RunningJob] = []
    completed_failures: List[Job] = []
    active: List[RunningJob] = []
    for item in running_jobs:
        rc = item.process.poll()
        if rc is None:
            active.append(item)
            continue
        duration = time.time() - item.started_at
        print(
            f"[done] rc={rc} ds={item.job.dataset} model={item.job.model} op={item.job.operator} "
            f"fold={item.job.fold} bs={item.batch_size} time={duration:.1f}s log={item.log_path.name}",
            flush=True,
        )
        if rc != 0 and should_retry_with_smaller_batch(item.log_path, item.batch_size):
            retry = rerun_with_smaller_batch(
                item.job,
                item.batch_size,
                tensorboard=tensorboard,
                version=version,
                runner_log_dir=runner_log_dir,
            )
            print(
                f"[retry] ds={item.job.dataset} model={item.job.model} op={item.job.operator} "
                f"fold={item.job.fold} bs={item.batch_size}->{retry.batch_size}",
                flush=True,
            )
            retry_jobs.append(retry)
        elif rc != 0:
            completed_failures.append(item.job)
    running_jobs[:] = active + retry_jobs
    return completed_failures


def run_scheduler(
    max_parallel: int,
    reserve_gb: float,
    poll_seconds: float,
    tensorboard: bool,
    version: str,
    active_log_dir: Path,
    runner_log_dir: Path,
    md_report: Path,
    status_json: Path,
) -> int:
    pending = missing_jobs(active_log_dir)
    running: List[RunningJob] = []
    failures: List[Job] = []
    print(
        f"pending_jobs={len(pending)} reserve_gb={reserve_gb:.1f} max_parallel={max_parallel} tensorboard={tensorboard}",
        flush=True,
    )
    write_status_report(active_log_dir, md_report, status_json, version)
    while pending or running:
        failures.extend(drain_finished_jobs(running, tensorboard=tensorboard, version=version, runner_log_dir=runner_log_dir))
        write_status_report(active_log_dir, md_report, status_json, version)

        launched = False
        while pending and len(running) < max_parallel:
            next_job = pending[0]
            free_gb = current_gpu_free_gb()
            if free_gb - reserve_gb < estimate_job_memory_gb(next_job):
                break
            running.append(
                launch_job(
                    next_job,
                    reserve_gb=reserve_gb,
                    running_jobs=running,
                    tensorboard=tensorboard,
                    version=version,
                    runner_log_dir=runner_log_dir,
                )
            )
            pending.pop(0)
            launched = True
            time.sleep(2.0)

        if launched:
            continue
        time.sleep(poll_seconds)

    write_status_report(active_log_dir, md_report, status_json, version)
    if failures:
        print(f"failed_jobs={len(failures)}", flush=True)
        for job in failures[:20]:
            print(f"failed {job.category} {job.dataset} {job.model} {job.operator} fold={job.fold}", flush=True)
        return 1
    print("all_missing_jobs_completed", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the missing baseline/operator experiments.")
    parser.add_argument("--max_parallel", type=int, default=6)
    parser.add_argument("--reserve_gb", type=float, default=4.0)
    parser.add_argument("--poll_seconds", type=float, default=20.0)
    parser.add_argument("--report_only", action="store_true")
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    args = parser.parse_args()
    ensure_version_manifest(ROOT)
    version = normalize_version(args.version)
    active_log_dir = log_dir(ROOT, version)
    runner_log_dir = active_log_dir / "missing_jobs"
    md_report = ROOT / "md" / f"missing_experiment_completion_{version}.md"
    status_json = record_dir(ROOT, version) / "missing_experiment_status.json"

    active_log_dir.mkdir(parents=True, exist_ok=True)
    runner_log_dir.mkdir(parents=True, exist_ok=True)
    status_json.parent.mkdir(parents=True, exist_ok=True)

    status = write_status_report(active_log_dir, md_report, status_json, version)
    print(json.dumps(status, indent=2), flush=True)
    if args.report_only:
        return

    raise SystemExit(
        run_scheduler(
            max_parallel=args.max_parallel,
            reserve_gb=args.reserve_gb,
            poll_seconds=args.poll_seconds,
            tensorboard=not args.no_tensorboard,
            version=version,
            active_log_dir=active_log_dir,
            runner_log_dir=runner_log_dir,
            md_report=md_report,
            status_json=status_json,
        )
    )


if __name__ == "__main__":
    main()
