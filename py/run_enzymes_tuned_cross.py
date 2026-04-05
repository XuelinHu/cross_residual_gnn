from __future__ import annotations

import concurrent.futures as cf
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/xuelin/miniconda3/envs/pyg/bin/python"


def build_jobs() -> list[list[str]]:
    jobs: list[list[str]] = []
    for fold in range(5):
        jobs.append(
            [
                PYTHON,
                "geomatric/graph_classify_v3.py",
                "--mode",
                "single",
                "--name",
                "GCNConv",
                "--batch_size",
                "32",
                "--grad_clip",
                "2.0",
                "--lr_factor",
                "0.5",
                "--lr_patience",
                "15",
                "--min_lr",
                "1e-5",
                "--ds",
                "ENZYMES",
                "--gname",
                "NodeCrossGNN",
                "--fold",
                str(fold),
                "--ep",
                "240",
                "--patience",
                "80",
                "--lr",
                "0.003",
                "--weight_decay",
                "5e-5",
                "--drop",
                "0.2",
                "--dim",
                "64",
                "--h_layer",
                "3",
                "--exp_tag",
                "enzymes_tuned",
            ]
        )
        jobs.append(
            [
                PYTHON,
                "geomatric/graph_classify_v3.py",
                "--mode",
                "single",
                "--name",
                "GCNConv",
                "--batch_size",
                "32",
                "--grad_clip",
                "2.0",
                "--lr_factor",
                "0.5",
                "--lr_patience",
                "15",
                "--min_lr",
                "1e-5",
                "--ds",
                "ENZYMES",
                "--gname",
                "GraphCrossGNN",
                "--fold",
                str(fold),
                "--ep",
                "240",
                "--patience",
                "80",
                "--lr",
                "0.002",
                "--weight_decay",
                "5e-5",
                "--drop",
                "0.2",
                "--dim",
                "64",
                "--h_layer",
                "3",
                "--exp_tag",
                "enzymes_tuned",
            ]
        )
    return jobs


def run_job(cmd: list[str]) -> tuple[int, float, str, str]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join(proc.stdout.strip().splitlines()[-4:]) if proc.stdout else ""
    return proc.returncode, time.time() - start, " ".join(cmd), tail or proc.stderr.strip()


def main() -> None:
    jobs = build_jobs()
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_job, cmd) for cmd in jobs]
        for idx, future in enumerate(cf.as_completed(futures), 1):
            rc, elapsed, cmd, tail = future.result()
            print(f"[{idx:02d}/{len(jobs)}] rc={rc} time={elapsed:.1f}s cmd={cmd}", flush=True)
            if tail:
                print(tail, flush=True)
            if rc != 0:
                sys.exit(rc)
    print(f"all_done total_time={time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
