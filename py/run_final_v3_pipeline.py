from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


STEPS = {
    "consolidate": [sys.executable, "py/consolidate_final_v3.py"],
    "summarize": [sys.executable, "py/summarize_paper_experiments.py", "--version", "V3"],
    "reports": [sys.executable, "py/generate_all_result_reports.py", "--version", "V3"],
    "figures": [sys.executable, "py/generate_suite_analysis_figures.py", "--version", "V3"],
    "sensitivity": [sys.executable, "py/generate_sensitivity_reports.py"],
}


def run_step(name: str) -> int:
    cmd = STEPS[name]
    print(f"[run] {name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Final paper-facing V3 workflow entry.")
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

    print("[done] final V3 workflow completed", flush=True)


if __name__ == "__main__":
    main()
