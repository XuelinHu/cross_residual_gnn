from __future__ import annotations

from typing import Dict, List


# 论文当前聚焦的主实验数据集，偏向生物图分类任务。
MAIN_BIOLOGICAL_DATASETS: List[str] = ["PROTEINS", "DD", "ENZYMES"]
# 作为补充稳健性实验的数据集。
SUPPLEMENTARY_DATASETS: List[str] = ["MUTAG", "AIDS", "Mutagenicity"]
ALL_ACTIVE_DATASETS: List[str] = [*MAIN_BIOLOGICAL_DATASETS, *SUPPLEMENTARY_DATASETS]

# 当前图分类主线对比的模型集合。
FOCUSED_MODELS: List[str] = [
    "PlainGNN",
    "NodeResGNN",
    "NodeCrossGNN",
    "GraphResGNN",
    "GraphCrossGNN",
]

# 当前最终版全口径实验包含的算子集合。
ACTIVE_OPERATORS: List[str] = [
    "GCNConv",
    "GATConv",
    "SAGEConv",
    "GINConv",
]

# 外部 baseline 与对应固定算子。
EXTERNAL_BASELINES: List[tuple[str, str]] = [
    ("GraphSAGEBaseline", "SAGEConv"),
    ("GINBaseline", "GINConv"),
    ("JKNetBaseline", "GCNConv"),
    ("APPNPBaseline", "GCNConv"),
]

MODEL_DISPLAY: Dict[str, str] = {
    "PlainGNN": "Plain",
    "NodeResGNN": "NodeRes",
    "NodeCrossGNN": "NodeCross",
    "GraphResGNN": "GraphRes",
    "GraphCrossGNN": "GraphCross",
}

# 数据集静态元信息，供训练脚本和分析脚本共用。
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
    """返回数据集所属家族。

    默认规则：
    - 已登记的数据集按 `DATASET_METADATA` 返回
    - 名称以 `ogbg-` 开头时视作 OGB graph property prediction
    - 其他情况默认按 TU 数据集处理
    """

    metadata = DATASET_METADATA.get(dataset_name)
    if metadata is not None:
        return metadata["family"]
    if dataset_name.startswith("ogbg-"):
        return "ogb_graphprop"
    return "tu"
