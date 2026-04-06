from __future__ import annotations

from typing import Dict, List


MAIN_BIOLOGICAL_DATASETS: List[str] = ["PROTEINS", "DD", "ENZYMES"]
SUPPLEMENTARY_DATASETS: List[str] = ["MUTAG", "AIDS", "Mutagenicity"]
ALL_ACTIVE_DATASETS: List[str] = [*MAIN_BIOLOGICAL_DATASETS, *SUPPLEMENTARY_DATASETS]

FOCUSED_MODELS: List[str] = [
    "PlainGNN",
    "NodeResGNN",
    "NodeCrossGNN",
    "GraphResGNN",
    "GraphCrossGNN",
]

MODEL_DISPLAY: Dict[str, str] = {
    "PlainGNN": "Plain",
    "NodeResGNN": "NodeRes",
    "NodeCrossGNN": "NodeCross",
    "GraphResGNN": "GraphRes",
    "GraphCrossGNN": "GraphCross",
}

DATASET_METADATA: Dict[str, Dict[str, str]] = {
    "PROTEINS": {
        "source": "PyG TUDataset",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
        "role": "main biological benchmark",
        "family": "tu",
    },
    "DD": {
        "source": "PyG TUDataset",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
        "role": "main biological benchmark",
        "family": "tu",
    },
    "ENZYMES": {
        "source": "PyG TUDataset",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
        "role": "main biological benchmark",
        "family": "tu",
    },
    "MUTAG": {
        "source": "PyG TUDataset",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
        "role": "supplementary robustness dataset",
        "family": "tu",
    },
    "AIDS": {
        "source": "PyG TUDataset",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
        "role": "supplementary robustness dataset",
        "family": "tu",
    },
    "Mutagenicity": {
        "source": "PyG TUDataset",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
        "role": "supplementary robustness dataset",
        "family": "tu",
    },
}


def dataset_family(dataset_name: str) -> str:
    metadata = DATASET_METADATA.get(dataset_name)
    if metadata is not None:
        return metadata["family"]
    if dataset_name.startswith("ogbg-"):
        return "ogb_graphprop"
    return "tu"
