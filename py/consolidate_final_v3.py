from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy_matching(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in src_dir.glob(pattern):
        dst = dst_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


def count_matching(base: Path, pattern: str) -> int:
    return sum(1 for _ in base.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate accepted V2/V3 experiment artifacts into final V3.")
    parser.add_argument("--src_main", default="V2")
    parser.add_argument("--src_new", default="V3")
    parser.add_argument("--dst", default="V3")
    args = parser.parse_args()

    logs_src_main = ROOT / "logs" / args.src_main
    records_src_main = ROOT / "records" / args.src_main
    logs_src_new = ROOT / "logs" / args.src_new
    records_src_new = ROOT / "records" / args.src_new
    logs_dst = ROOT / "logs" / args.dst
    records_dst = ROOT / "records" / args.dst

    copied = Counter()

    # Accepted formal benchmark and supplementary studies from V2.
    copied["logs_from_main"] += copy_matching(logs_src_main, logs_dst, "train_*__*.json")
    copied["records_from_main"] += copy_matching(records_src_main, records_dst, "*")

    # Preserve all existing V3 new residual artifacts.
    copied["logs_from_new"] += copy_matching(logs_src_new, logs_dst, "train_*__*.json")
    copied["records_from_new"] += copy_matching(records_src_new, records_dst, "*")

    summary = {
        "src_main": args.src_main,
        "src_new": args.src_new,
        "dst": args.dst,
        "copied": dict(copied),
        "dst_counts": {
            "train_json": count_matching(logs_dst, "train_*__*.json"),
            "records_files": count_matching(records_dst, "*"),
        },
    }
    out = records_dst / "v3_consolidation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
