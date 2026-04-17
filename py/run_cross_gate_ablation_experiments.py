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


DEFAULT_CROSS_PROTOCOL: Dict[str, object] = {
    "ep": 240,
    "patience": 80,
    "lr": 0.003,
    "weight_decay": 5e-5,
    "drop": 0.3,
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

SPECIAL_PROTOCOLS: Dict[Tuple[str, str], Dict[str, object]] = {
    ("DD", "NodeCrossGNN"): {
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "dim": 64,
        "h_layer": 3,
    },
}

TARGETS: Sequence[Dict[str, str]] = [
    {"ds": "AIDS", "gname": "NodeCrossGNN", "name": "GATConv"},
    {"ds": "DD", "gname": "NodeCrossGNN", "name": "GATConv"},
    {"ds": "DD", "gname": "NodeCrossGNN", "name": "GINConv"},
    {"ds": "ENZYMES", "gname": "NodeCrossGNN", "name": "GATConv"},
    {"ds": "MUTAG", "gname": "GraphCrossGNN", "name": "GATConv"},
    {"ds": "Mutagenicity", "gname": "NodeCrossGNN", "name": "GATConv"},
]

GATE_SETTINGS: Sequence[Tuple[str, Dict[str, object]]] = [
    ("learnable", {"gate_mode": "learnable"}),
    ("fixed_0p0", {"gate_mode": "fixed", "fixed_gate_value": 0.0}),
    ("fixed_0p5", {"gate_mode": "fixed", "fixed_gate_value": 0.5}),
    ("fixed_1p0", {"gate_mode": "fixed", "fixed_gate_value": 1.0}),
]


def build_config(target: Dict[str, str], gate_tag: str, gate_overrides: Dict[str, object], fold: int) -> Dict[str, object]:
    config = dict(DEFAULT_CROSS_PROTOCOL)
    config.update(SPECIAL_PROTOCOLS.get((target["ds"], target["gname"]), {}))
    config.update(target)
    config.update(gate_overrides)
    config["fold"] = fold
    config["exp_tag"] = (
        f"cross_gate_{target['ds'].lower()}_{target['gname'].lower()}_{target['name'].lower()}_{gate_tag}"
    )
    return config


def build_command(config: Dict[str, object], version: str) -> List[str]:
    cmd = [sys.executable, "geomatric/graph_classify_v3.py", "--version", version]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def iter_jobs(version: str) -> List[Tuple[str, List[str]]]:
    jobs: List[Tuple[str, List[str]]] = []
    for target in TARGETS:
        slug = f"{target['ds']}_{target['gname']}_{target['name']}"
        for gate_tag, gate_overrides in GATE_SETTINGS:
            for fold in range(5):
                config = build_config(target, gate_tag, gate_overrides, fold)
                jobs.append((f"{slug}_{gate_tag}_fold{fold}", build_command(config, version)))
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
    parser = argparse.ArgumentParser(description="Run gate ablations on cross-winning V2 configurations.")
    parser.add_argument("--version", type=str, default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--list_only", action="store_true")
    args = parser.parse_args()

    jobs = iter_jobs(args.version)
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
