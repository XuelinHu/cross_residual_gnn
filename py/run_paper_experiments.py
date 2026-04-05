from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
from typing import Dict, List, Tuple


FOCUSED_MAIN_DATASETS = ["MUTAG", "PROTEINS", "DD", "MSRC_9"]
TOPIC_DATASETS = ["PROTEINS", "DD", "ENZYMES"]
EXTENDED_DATASETS = ["AIDS", "Mutagenicity"]
ALL_DATASETS = ["MUTAG", "PROTEINS", "DD", "ENZYMES", "MSRC_9", "AIDS", "Mutagenicity"]

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
    "batch_size": 32,
    "grad_clip": 2.0,
    "lr_factor": 0.5,
    "lr_patience": 15,
    "min_lr": 1e-5,
}


def build_protocol(dataset: str, model: str) -> Dict[str, object]:
    if model in BASELINE_PROTOCOLS:
        return {**BASELINE_PROTOCOLS[model], **COMMON_ARGS}
    return {**CROSS_PROTOCOLS.get((dataset, model), DEFAULT_CROSS_PROTOCOL), **COMMON_ARGS}


def build_command(dataset: str, model: str, fold: int) -> List[str]:
    protocol = build_protocol(dataset, model)
    cmd = [
        sys.executable,
        "geomatric/graph_classify_v3.py",
        "--mode",
        "single",
        "--ds",
        dataset,
        "--gname",
        model,
        "--fold",
        str(fold),
    ]
    for key, value in protocol.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def build_jobs(datasets: List[str], models: List[str], folds: List[int]) -> List[Tuple[str, str, int, List[str]]]:
    jobs: List[Tuple[str, str, int, List[str]]] = []
    for dataset in datasets:
        for model in models:
            for fold in folds:
                jobs.append((dataset, model, fold, build_command(dataset, model, fold)))
    return jobs


def run_job(job: Tuple[str, str, int, List[str]]) -> Dict[str, object]:
    dataset, model, fold, cmd = job
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    return {
        "dataset": dataset,
        "model": model,
        "fold": fold,
        "cmd": " ".join(cmd),
        "rc": proc.returncode,
        "duration": duration,
        "tail": "\n".join(proc.stdout.strip().splitlines()[-4:]) if proc.stdout else "",
        "stderr": proc.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready V3 experiments.")
    parser.add_argument(
        "--dataset_group",
        choices=["main", "topic", "extended", "all"],
        default="main",
        help="Dataset bundle to execute.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["PlainGNN", "NodeResGNN", "NodeCrossGNN", "GraphResGNN", "GraphCrossGNN"],
        help="Model list to execute.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging for every run.")
    args = parser.parse_args()

    if args.dataset_group == "main":
        datasets = FOCUSED_MAIN_DATASETS
    elif args.dataset_group == "topic":
        datasets = TOPIC_DATASETS
    elif args.dataset_group == "extended":
        datasets = EXTENDED_DATASETS
    else:
        datasets = ALL_DATASETS

    jobs = build_jobs(datasets, args.models, args.folds)
    if args.tensorboard:
        for index, job in enumerate(jobs):
            dataset, model, fold, cmd = job
            jobs[index] = (dataset, model, fold, [*cmd, "--tensorboard"])
    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for index, future in enumerate(cf.as_completed(futures), 1):
            result = future.result()
            status = "OK" if result["rc"] == 0 else "FAIL"
            print(
                f"[{index:03d}/{len(jobs)}] {status} "
                f"ds={result['dataset']} model={result['model']} fold={result['fold']} "
                f"time={result['duration']:.1f}s",
                flush=True,
            )
            if result["tail"]:
                print(result["tail"], flush=True)
            if result["stderr"]:
                print(result["stderr"], file=sys.stderr, flush=True)
            if result["rc"] != 0:
                print(result["cmd"], file=sys.stderr, flush=True)
                sys.exit(result["rc"])

    print(f"all_done total_time={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
