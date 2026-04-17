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

from geomatric.experiment_paths import DEFAULT_EXPERIMENT_VERSION


BASE_CONFIG: Dict[str, object] = {
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
    "batch_size": 256,
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

SENSITIVITY_SWEEPS: Dict[str, Sequence[object]] = {
    "lr": [0.001, 0.002, 0.003, 0.005],
    "drop": [0.2, 0.3, 0.5, 0.6],
    "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4],
    "h_layer": [2, 3, 4, 5],
    "dim": [32, 64, 128],
    "gate_init": [0.2, 0.5, 0.8, 0.95],
}

GATE_ABLATION_VALUES = [0.0, 0.5, 1.0]


def format_tag_value(value: object) -> str:
    return str(value).replace(".", "p")


def build_command(config: Dict[str, object], version: str) -> List[str]:
    cmd = [sys.executable, "geomatric/graph_classify_v3.py", "--version", version]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def sensitivity_jobs(version: str) -> List[Tuple[str, List[str]]]:
    jobs: List[Tuple[str, List[str]]] = []
    seen = set()
    for sweep_name, values in SENSITIVITY_SWEEPS.items():
        for value in values:
            config = dict(BASE_CONFIG)
            config[sweep_name] = value
            config["fold"] = 0
            config["exp_tag"] = f"aids_supp_sens_{sweep_name}_{format_tag_value(value)}"
            signature = tuple((key, config[key]) for key in sorted(config))
            if signature in seen:
                continue
            seen.add(signature)
            jobs.append((config["exp_tag"], build_command(config, version)))
    return jobs


def gate_ablation_jobs(version: str) -> List[Tuple[str, List[str]]]:
    jobs: List[Tuple[str, List[str]]] = []
    for fold in range(5):
        learnable_cfg = dict(BASE_CONFIG)
        learnable_cfg["fold"] = fold
        learnable_cfg["exp_tag"] = "aids_supp_gate_learnable"
        jobs.append((f"learnable_fold{fold}", build_command(learnable_cfg, version)))

        for fixed_value in GATE_ABLATION_VALUES:
            fixed_cfg = dict(BASE_CONFIG)
            fixed_cfg["fold"] = fold
            fixed_cfg["gate_mode"] = "fixed"
            fixed_cfg["fixed_gate_value"] = fixed_value
            fixed_cfg["exp_tag"] = f"aids_supp_gate_fixed_{format_tag_value(fixed_value)}"
            jobs.append((f"fixed_{format_tag_value(fixed_value)}_fold{fold}", build_command(fixed_cfg, version)))
    return jobs


def iter_jobs(which: str, version: str) -> List[Tuple[str, List[str]]]:
    jobs: List[Tuple[str, List[str]]] = []
    if which in {"sensitivity", "all"}:
        jobs.extend(sensitivity_jobs(version))
    if which in {"gate_ablation", "all"}:
        jobs.extend(gate_ablation_jobs(version))
    return jobs


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
    parser = argparse.ArgumentParser(description="Run AIDS supplementary parameter and gate-ablation experiments.")
    parser.add_argument("--which", choices=["sensitivity", "gate_ablation", "all"], default="all")
    parser.add_argument("--version", type=str, default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--list_only", action="store_true")
    args = parser.parse_args()

    jobs = iter_jobs(args.which, args.version)
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
