from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.experiment_paths import DEFAULT_EXPERIMENT_VERSION, log_dir, normalize_version


TARGETS: Sequence[Dict[str, object]] = [
    {
        "ds": "AIDS",
        "gname": "GraphResGNN",
        "name": "GINConv",
        "ep": 200,
        "patience": 60,
        "lr": 0.005,
        "weight_decay": 1e-4,
        "drop": 0.5,
        "dim": 64,
        "h_layer": 4,
    },
    {
        "ds": "PROTEINS",
        "gname": "GraphResGNN",
        "name": "SAGEConv",
        "ep": 200,
        "patience": 60,
        "lr": 0.005,
        "weight_decay": 1e-4,
        "drop": 0.5,
        "dim": 64,
        "h_layer": 4,
    },
    {
        "ds": "AIDS",
        "gname": "NodeCrossGNN",
        "name": "GATConv",
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "dim": 64,
        "h_layer": 4,
    },
    {
        "ds": "DD",
        "gname": "NodeCrossGNN",
        "name": "GATConv",
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "dim": 64,
        "h_layer": 3,
    },
]

COMMON_ARGS: Dict[str, object] = {
    "batch_size": 128,
    "grad_clip": 2.0,
    "lr_factor": 0.5,
    "lr_patience": 15,
    "min_lr": 1e-5,
    "gate_init": 0.8,
    "gate_mode": "learnable",
    "fixed_gate_value": 0.8,
    "mode": "single",
    "tensorboard": True,
}

RESIDUAL_SETTINGS: Sequence[Tuple[str, Dict[str, object]]] = [
    ("learnable", {"residual_mode": "learnable"}),
    ("topk_0p25", {"residual_mode": "topk", "topk_ratio": 0.25}),
    ("topk_0p5", {"residual_mode": "topk", "topk_ratio": 0.5}),
    ("sparse_0p02", {"residual_mode": "sparse", "sparse_lambda": 0.02}),
    ("sparse_0p05", {"residual_mode": "sparse", "sparse_lambda": 0.05}),
]


def build_command(config: Dict[str, object], version: str) -> List[str]:
    cmd = [sys.executable, "geomatric/graph_classify_v3.py", "--version", version]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def build_jobs(version: str, folds: Sequence[int]) -> List[Tuple[str, List[str]]]:
    jobs: List[Tuple[str, List[str]]] = []
    for target in TARGETS:
        slug = f"{target['ds']}_{target['gname']}_{target['name']}"
        for setting_name, setting in RESIDUAL_SETTINGS:
            for fold in folds:
                config = {**target, **COMMON_ARGS, **setting}
                config["fold"] = fold
                config["exp_tag"] = f"residual_mode_{target['ds'].lower()}_{target['gname'].lower()}_{target['name'].lower()}_{setting_name}"
                jobs.append((f"{slug}_{setting_name}_fold{fold}", build_command(config, version)))
    return jobs


def result_exists(version: str, tag: str, cmd: Sequence[str]) -> bool:
    version_logs = log_dir(ROOT, normalize_version(version))
    args = list(cmd)
    ds = args[args.index("--ds") + 1]
    gname = args[args.index("--gname") + 1]
    name = args[args.index("--name") + 1]
    fold = args[args.index("--fold") + 1]
    exp_tag = args[args.index("--exp_tag") + 1]
    pattern = f"train_{ds}_{gname}_{name}_fold{fold}_{exp_tag}__*.json"
    return any(version_logs.glob(pattern))


def filter_missing_jobs(version: str, jobs: Iterable[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    return [job for job in jobs if not result_exists(version, job[0], job[1])]


def run_job(job: Tuple[str, List[str]]) -> Dict[str, object]:
    tag, cmd = job
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "tag": tag,
        "cmd": " ".join(cmd),
        "rc": proc.returncode,
        "time": time.time() - start,
        "tail": "\n".join(proc.stdout.strip().splitlines()[-4:]) if proc.stdout else "",
        "stderr": proc.stderr.strip(),
    }


def run_jobs(jobs: Iterable[Tuple[str, List[str]]], max_workers: int) -> int:
    jobs = list(jobs)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for idx, future in enumerate(cf.as_completed(futures), 1):
            result = future.result()
            status = "OK" if result["rc"] == 0 else "FAIL"
            print(f"[{idx:03d}/{len(jobs)}] {status} {result['tag']} time={result['time']:.1f}s", flush=True)
            if result["tail"]:
                print(result["tail"], flush=True)
            if result["stderr"]:
                print(result["stderr"], file=sys.stderr, flush=True)
            if result["rc"] != 0:
                print(result["cmd"], file=sys.stderr, flush=True)
                return result["rc"]
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learnable/top-k/sparse residual ablations.")
    parser.add_argument("--version", type=str, default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--list_only", action="store_true")
    parser.add_argument("--missing_only", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.version, args.folds)
    if args.missing_only:
        jobs = filter_missing_jobs(args.version, jobs)
    if args.list_only:
        for tag, cmd in jobs:
            print(tag, " ".join(cmd))
        return

    start = time.time()
    rc = run_jobs(jobs, max_workers=args.max_workers)
    print(f"all_done rc={rc} total_time={time.time() - start:.1f}s", flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
