from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
from typing import Dict, List, Tuple


DATASETS = ["PROTEINS", "DD", "ENZYMES"]
MODELS = ["NodeCrossGNN", "GraphCrossGNN"]

BASE_CONFIGS: Dict[Tuple[str, str], Dict[str, object]] = {
    ("PROTEINS", "NodeCrossGNN"): {
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.2,
        "dim": 64,
        "h_layer": 4,
    },
    ("PROTEINS", "GraphCrossGNN"): {
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "dim": 64,
        "h_layer": 4,
    },
    ("DD", "NodeCrossGNN"): {
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "dim": 64,
        "h_layer": 3,
    },
    ("DD", "GraphCrossGNN"): {
        "ep": 240,
        "patience": 80,
        "lr": 0.002,
        "weight_decay": 5e-5,
        "drop": 0.2,
        "dim": 64,
        "h_layer": 4,
    },
    ("ENZYMES", "NodeCrossGNN"): {
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "dim": 64,
        "h_layer": 4,
    },
    ("ENZYMES", "GraphCrossGNN"): {
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "dim": 64,
        "h_layer": 4,
    },
}

COMMON_ARGS = {
    "name": "GCNConv",
    "batch_size": 32,
    "grad_clip": 2.0,
    "lr_factor": 0.5,
    "lr_patience": 15,
    "min_lr": 1e-5,
    "mode": "single",
}

SWEEP_VALUES = {
    "h_layer": [3, 4, 5],
    "drop": [0.2, 0.3, 0.5],
    "lr": [0.002, 0.003, 0.005],
}


def build_jobs(fold: int) -> List[Tuple[str, str, str, object, List[str]]]:
    jobs: List[Tuple[str, str, str, object, List[str]]] = []
    seen = set()
    for dataset in DATASETS:
        for model in MODELS:
            base = dict(BASE_CONFIGS[(dataset, model)])
            for sweep_name, values in SWEEP_VALUES.items():
                for value in values:
                    config = dict(base)
                    config[sweep_name] = value
                    signature = (dataset, model, fold, config["ep"], config["patience"], config["lr"], config["weight_decay"], config["drop"], config["dim"], config["h_layer"])
                    if signature in seen:
                        continue
                    seen.add(signature)
                    cmd = [
                        sys.executable,
                        "geomatric/graph_classify_v3.py",
                    ]
                    tag_value = str(value).replace(".", "p")
                    full_cfg = {
                        **COMMON_ARGS,
                        **config,
                        "ds": dataset,
                        "gname": model,
                        "fold": fold,
                        "exp_tag": f"sensitivity_{sweep_name}_{tag_value}",
                    }
                    for key, val in full_cfg.items():
                        cmd.extend([f"--{key}", str(val)])
                    jobs.append((dataset, model, sweep_name, value, cmd))
    return jobs


def run_job(job: Tuple[str, str, str, object, List[str]]) -> Dict[str, object]:
    dataset, model, sweep_name, value, cmd = job
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "dataset": dataset,
        "model": model,
        "sweep": sweep_name,
        "value": value,
        "rc": proc.returncode,
        "time": time.time() - start,
        "tail": "\n".join(proc.stdout.strip().splitlines()[-4:]) if proc.stdout else "",
        "stderr": proc.stderr.strip(),
        "cmd": " ".join(cmd),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fold-level sensitivity scans for cross-residual models.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=6)
    args = parser.parse_args()

    jobs = build_jobs(args.fold)
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for idx, future in enumerate(cf.as_completed(futures), 1):
            result = future.result()
            status = "OK" if result["rc"] == 0 else "FAIL"
            print(
                f"[{idx:03d}/{len(jobs)}] {status} ds={result['dataset']} model={result['model']} "
                f"{result['sweep']}={result['value']} time={result['time']:.1f}s",
                flush=True,
            )
            if result["tail"]:
                print(result["tail"], flush=True)
            if result["stderr"]:
                print(result["stderr"], file=sys.stderr, flush=True)
            if result["rc"] != 0:
                print(result["cmd"], file=sys.stderr, flush=True)
                sys.exit(result["rc"])
    print(f"all_done total_time={time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
