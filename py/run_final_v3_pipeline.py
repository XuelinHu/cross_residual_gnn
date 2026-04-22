from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


STEPS = {
    "consolidate": [sys.executable, "py/consolidate_final_v3.py"],
    "summarize": [sys.executable, "py/summarize_paper_experiments.py", "--version", "LATEST"],
    "reports": [sys.executable, "py/generate_all_result_reports.py", "--version", "LATEST"],
    "figures": [sys.executable, "py/generate_suite_analysis_figures.py", "--version", "LATEST"],
    "sensitivity": [sys.executable, "py/generate_sensitivity_reports.py"],
}


def run_step(name: str) -> int:
    cmd = STEPS[name]
    print(f"[run] {name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Latest paper-facing workflow entry.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["consolidate", "summarize", "reports"],
        choices=list(STEPS.keys()),
        help="Workflow steps to execute.",
    )
    args = parser.parse_args()

    for step in args.steps:
        rc = run_step(step)
        if rc != 0:
            raise SystemExit(rc)

    print("[done] latest workflow completed", flush=True)


if __name__ == "__main__":
    main()
