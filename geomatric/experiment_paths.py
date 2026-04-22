from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


DEFAULT_EXPERIMENT_VERSION = "LATEST"
ARCHIVED_EXPERIMENT_VERSIONS = ("V1", "V2", "V3")


def normalize_version(version: str | None) -> str:
    if version is None:
        return DEFAULT_EXPERIMENT_VERSION
    text = version.strip()
    return text.upper() if text else DEFAULT_EXPERIMENT_VERSION


def versioned_dir(project_root: Path, category: str, version: str) -> Path:
    return project_root / category / normalize_version(version)


def log_dir(project_root: Path, version: str) -> Path:
    return versioned_dir(project_root, "logs", version)


def record_dir(project_root: Path, version: str) -> Path:
    return versioned_dir(project_root, "records", version)


def run_dir(project_root: Path, version: str) -> Path:
    return versioned_dir(project_root, "runs", version)


def version_manifest_path(project_root: Path) -> Path:
    return project_root / "records" / "experiment_versions.json"


def manifest_payload(project_root: Path) -> Dict[str, object]:
    versions = {
        "V1": {
            "description": "Archived pre-versioned experiment outputs moved from top-level runtime directories.",
            "logs_dir": str(log_dir(project_root, "V1").relative_to(project_root)),
            "records_dir": str(record_dir(project_root, "V1").relative_to(project_root)),
            "runs_dir": str(run_dir(project_root, "V1").relative_to(project_root)),
        },
        "V2": {
            "description": "Archived formal reruns with supplementary gate ablations and cross-gate studies.",
            "logs_dir": str(log_dir(project_root, "V2").relative_to(project_root)),
            "records_dir": str(record_dir(project_root, "V2").relative_to(project_root)),
            "runs_dir": str(run_dir(project_root, "V2").relative_to(project_root)),
        },
        "V3": {
            "description": "Archived pre-latest consolidation stage kept for historical comparison.",
            "logs_dir": str(log_dir(project_root, "V3").relative_to(project_root)),
            "records_dir": str(record_dir(project_root, "V3").relative_to(project_root)),
            "runs_dir": str(run_dir(project_root, "V3").relative_to(project_root)),
        },
        DEFAULT_EXPERIMENT_VERSION: {
            "description": "Current paper-facing latest version for training, reports, and consolidated artifacts.",
            "logs_dir": str(log_dir(project_root, DEFAULT_EXPERIMENT_VERSION).relative_to(project_root)),
            "records_dir": str(record_dir(project_root, DEFAULT_EXPERIMENT_VERSION).relative_to(project_root)),
            "runs_dir": str(run_dir(project_root, DEFAULT_EXPERIMENT_VERSION).relative_to(project_root)),
        },
    }
    return {
        "default_version": DEFAULT_EXPERIMENT_VERSION,
        "versions": versions,
    }


def ensure_version_manifest(project_root: Path) -> Path:
    manifest = version_manifest_path(project_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_payload(project_root), indent=2), encoding="utf-8")
    return manifest
