"""图分类 V3 主训练脚本。

该文件承担四类职责：
1. 解析训练参数，并规范化模型别名与默认超参数。
2. 加载 TU / OGB 图数据集，完成分层划分与统计导出。
3. 定义 Plain / Residual / Cross 以及外部 baseline 模型。
4. 执行单配置训练、评估、日志落盘和预设实验套件。
"""

import argparse
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    APPNP,
    GATConv,
    GCNConv,
    GINConv,
    JumpingKnowledge,
    SAGEConv,
    TransformerConv,
    global_mean_pool,
)

try:
    from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
except ImportError:
    Evaluator = None
    PygGraphPropPredDataset = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geomatric.experiment_catalog import dataset_family
from geomatric.experiment_paths import (
    DEFAULT_EXPERIMENT_VERSION,
    ensure_version_manifest,
    log_dir,
    normalize_version,
    record_dir,
    run_dir,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = PROJECT_ROOT / "data"
SEPARATOR = "__"

# 本文提出的自定义模型家族。
FAMILY_MODELS = [
    "PlainGNN",
    "NodeResGNN",
    "NodeCrossGNN",
    "GraphCondGNN",
    "GraphResGNN",
    "GraphCrossGNN",
]

# 论文中的外部基线模型。
EXTERNAL_BASELINES = [
    "GraphSAGEBaseline",
    "GINBaseline",
    "JKNetBaseline",
    "APPNPBaseline",
]

# 当前主线实验允许的消息传递算子。
AVAILABLE_OPERATORS = [
    "GCNConv",
    "GATConv",
    "SAGEConv",
    "GINConv",
]


def gate_logit_from_probability(probability: float) -> float:
    """把 [0,1] 区间内的初值映射到 logit 空间，便于用 sigmoid 学习门限。"""

    clipped = min(max(probability, 1e-4), 1.0 - 1e-4)
    return float(np.log(clipped / (1.0 - clipped)))


class LearnableGate(nn.Module):
    """可学习门限。

    内部参数存储在 logit 空间，前向时通过 sigmoid 投影到 (0,1)。
    """

    def __init__(self, init_probability: float = 1.0):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(gate_logit_from_probability(init_probability), dtype=torch.float))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    关键默认值：
    - `--name=GCNConv`：默认图卷积算子
    - `--gname=PlainGNN`：默认基线结构
    - `--ds=MUTAG`：默认数据集
    - `--ep=500`：默认训练 500 轮
    - `--lr=1e-2`、`--weight_decay=1e-2`：默认优化器超参数
    - `--dim=64`、`--h_layer=2`：默认隐藏维度与隐藏层数
    - `--drop=0.6`：默认 dropout
    """

    parser = argparse.ArgumentParser(
        description="V3 图分类训练入口，包含残差/交叉结构与外部基线。"
    )
    parser.add_argument("--name", type=str, default="GCNConv", help="消息传递算子，默认 GCNConv。")
    parser.add_argument("--gname", type=str, default="PlainGNN", help="模型结构名称，默认 PlainGNN。")
    parser.add_argument("--ds", type=str, default="MUTAG", help="数据集名称，可为 TU 或 OGB 图属性预测数据集。")
    parser.add_argument("--ep", type=int, default=500, help="训练轮数，默认 500。")
    parser.add_argument("--lr", type=float, default=1e-2, help="学习率，默认 1e-2。")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--drop", type=float, default=0.6)
    parser.add_argument("--dim", type=int, default=64, help="隐藏层维度，默认 64。")
    parser.add_argument("--h_layer", type=int, default=2, help="隐藏图卷积层数，默认 2。")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_EXPERIMENT_VERSION,
        help="实验输出版本目录，默认写入 V2；旧结果统一归档在 V1。",
    )
    parser.add_argument("--fold", type=int, default=0, help="五折交叉验证中的 fold 编号，默认 0。")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="训练集内部再切分验证集的比例，默认 0.1。")
    parser.add_argument("--patience", type=int, default=20, help="基于验证损失的早停耐心轮数，默认 20。")
    parser.add_argument("--min_delta", type=float, default=0.0, help="验证损失至少改善多少才重置早停计数。")
    parser.add_argument("--grad_clip", type=float, default=2.0, help="梯度裁剪阈值；小于等于 0 表示关闭。")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="学习率衰减因子，默认 0.5。")
    parser.add_argument("--lr_patience", type=int, default=15, help="学习率调度器耐心轮数，默认 15。")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="学习率下界，默认 1e-5。")
    parser.add_argument(
        "--gate_init",
        type=float,
        default=0.8,
        help="残差/交叉门限的初始值，训练时会作为可学习参数更新，取值建议在 0 到 1 之间。",
    )
    parser.add_argument("--jk_mode", type=str, default="cat", choices=["cat", "max", "lstm"])
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "suite", "stats"],
        help="single 训练单配置；suite 执行预设实验组；stats 仅导出数据集统计。",
    )
    parser.add_argument(
        "--suite_name",
        type=str,
        default="external_baselines",
        choices=["external_baselines", "node_residual", "graph_residual", "depth_ablation"],
    )
    parser.add_argument("--debug", action="store_true", help="Skip file writes.")
    parser.add_argument("--tensorboard", action="store_true", help="Write TensorBoard logs if available.")
    parser.add_argument("--exp_tag", type=str, default="", help="Optional tag appended to output filenames.")
    parser.add_argument("--report_dataset_stats", action="store_true", help="Print dataset statistics before training.")
    parser.add_argument("--save_dataset_stats", action="store_true", help="Save dataset statistics to records.")
    return parser.parse_args()


def canonical_args(args: argparse.Namespace) -> argparse.Namespace:
    """规范化算子名，避免命名差异导致当前实现不一致。"""

    if args.name == "GraphSAGE":
        args.name = "SAGEConv"
    if args.name == "GIN":
        args.name = "GINConv"
    return args


def ensure_dirs() -> None:
    """确保训练输出所需的目录存在。"""

    for path in [
        DATA_ROOT,
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "records",
        PROJECT_ROOT / "runs",
        log_dir(PROJECT_ROOT, DEFAULT_EXPERIMENT_VERSION),
        record_dir(PROJECT_ROOT, DEFAULT_EXPERIMENT_VERSION),
        run_dir(PROJECT_ROOT, DEFAULT_EXPERIMENT_VERSION),
        log_dir(PROJECT_ROOT, "V1"),
        record_dir(PROJECT_ROOT, "V1"),
        run_dir(PROJECT_ROOT, "V1"),
    ]:
        path.mkdir(parents=True, exist_ok=True)
    ensure_version_manifest(PROJECT_ROOT)


def set_seed(seed: int) -> None:
    """统一设置 CPU 与 CUDA 的随机种子，便于结果复现。"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    """生成用于文件名的时间戳。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_metrics(**metrics: object) -> str:
    """把一组指标格式化为便于日志打印的 `key=value` 字符串。"""

    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.5f}")
        else:
            parts.append(f"{key}={value}")
    return "\t".join(parts)


def with_exp_tag(prefix: str, exp_tag: str) -> str:
    """在输出前缀后拼接实验标签，便于区分多轮实验。"""

    if not exp_tag:
        return prefix
    return f"{prefix}_{exp_tag}"


def save_json(payload: Dict[str, object], prefix: str, debug: bool, output_dir: Path) -> Optional[Path]:
    """把结构化实验结果写入日志目录；debug 模式下只返回目标路径。"""

    file_path = output_dir / f"{prefix}{SEPARATOR}{timestamp()}.json"
    if debug:
        print(f"[debug] skip save_json -> {file_path}")
        return file_path
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return file_path


def save_lines(lines: List[str], prefix: str, debug: bool, output_dir: Path) -> Optional[Path]:
    """把实验摘要按文本行写入记录目录。"""

    file_path = output_dir / f"{prefix}{SEPARATOR}{timestamp()}.txt"
    if debug:
        print(f"[debug] skip save_lines -> {file_path}")
        return file_path
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return file_path


def infer_input_dim(dataset: object) -> int:
    """推断图节点特征维度；无特征数据集默认补成 1 维。"""

    return dataset.num_features if dataset.num_features > 0 else 1


def prepare_batch(data: torch.Tensor) -> torch.Tensor:
    """把 batch 补齐到可训练格式，并移动到当前设备。"""

    if data.x is None:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
    return data.to(DEVICE)


def load_dataset(dataset_name: str) -> object:
    """按数据集家族加载 TU 或 OGB 图分类数据集。"""

    family = dataset_family(dataset_name)
    if family == "tu":
        dataset = TUDataset(root=str(DATA_ROOT / "TUDataset"), name=dataset_name)
        return dataset.shuffle()
    if family == "ogb_graphprop":
        if PygGraphPropPredDataset is None:
            raise ImportError(
                "ogb is required for ogbg datasets. Install it with `pip install ogb`."
            )
        return PygGraphPropPredDataset(name=dataset_name, root=str(DATA_ROOT / "OGB"))
    raise ValueError(f"Unsupported dataset family for {dataset_name}.")


def graph_target(graph: torch.Tensor) -> torch.Tensor:
    """把图标签压平成分类训练所需的一维 LongTensor。"""

    return graph.y.view(-1).long()


def dataset_labels(dataset: Iterable[torch.Tensor]) -> List[int]:
    """提取一组图样本的标签，用于分层划分。"""

    return [int(graph_target(graph)[0]) for graph in dataset]


def stratified_kfold_indices(labels: List[int], n_splits: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """手工构造分层 K 折索引，保证每个类别尽量均匀分布到各折。"""

    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for indices in label_to_indices.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for offset, index in enumerate(shuffled):
            folds[offset % n_splits].append(index)

    split_pairs: List[Tuple[List[int], List[int]]] = []
    all_indices = set(range(len(labels)))
    for fold_indices in folds:
        test_indices = sorted(fold_indices)
        train_indices = sorted(all_indices - set(test_indices))
        split_pairs.append((train_indices, test_indices))
    return split_pairs


def stratified_train_val_indices(labels: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """在训练集内部继续做一次分层训练/验证切分。"""

    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for indices in label_to_indices.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_ratio))
        if len(shuffled) > 1:
            val_count = max(1, min(len(shuffled) - 1, val_count))
        else:
            val_count = 0
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])
    return sorted(train_indices), sorted(val_indices)


def split_dataset(
    dataset: object,
    fold: int,
    dataset_name: str,
) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Optional[Sequence[torch.Tensor]], Dict[str, object]]:
    """根据数据集家族生成训练/测试/验证划分及其上下文说明。"""

    family = dataset_family(dataset_name)
    if family == "tu":
        graph_list = list(dataset)
        labels = dataset_labels(graph_list)
        folds = stratified_kfold_indices(labels, n_splits=5, seed=0)
        train_indices, test_indices = folds[fold]
        train_dataset = [graph_list[index] for index in train_indices]
        test_dataset = [graph_list[index] for index in test_indices]
        return train_dataset, test_dataset, None, {
            "dataset_family": family,
            "split_protocol": "stratified_5fold_cv",
            "repeat_id": fold,
            "official_split": False,
        }

    split_idx = dataset.get_idx_split()
    train_dataset = Subset(dataset, split_idx["train"].tolist())
    valid_dataset = Subset(dataset, split_idx["valid"].tolist())
    test_dataset = Subset(dataset, split_idx["test"].tolist())
    return train_dataset, test_dataset, valid_dataset, {
        "dataset_family": family,
        "split_protocol": "official_ogb_split",
        "repeat_id": fold,
        "official_split": True,
    }


def split_train_val_dataset(
    train_dataset: Sequence[torch.Tensor],
    val_ratio: float,
    seed: int,
) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    """从训练集里再切一份验证集；不满足条件时直接返回空验证集。"""

    if not train_dataset:
        return [], []
    if val_ratio <= 0.0:
        return train_dataset, []

    if len(train_dataset) <= 1:
        return train_dataset, []

    labels = dataset_labels(train_dataset)
    if len(set(labels)) < 2:
        return train_dataset, []

    train_indices, val_indices = stratified_train_val_indices(labels, val_ratio=val_ratio, seed=seed)
    inner_train_dataset = [train_dataset[index] for index in train_indices]
    val_dataset = [train_dataset[index] for index in val_indices]
    return inner_train_dataset, val_dataset


def build_loader(dataset_slice: Sequence[torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    """把数据切片封装为 PyG DataLoader。"""

    return DataLoader(dataset_slice, batch_size=batch_size, shuffle=shuffle)


def sampled_graph_iter(dataset: object, sample_cap: int = 2048) -> Iterable[torch.Tensor]:
    """为大数据集提供等间隔采样迭代器，避免统计阶段全量遍历过慢。"""

    total = len(dataset)
    if total <= sample_cap:
        for graph in dataset:
            yield graph
        return
    indices = np.linspace(0, total - 1, num=sample_cap, dtype=int)
    for index in indices.tolist():
        yield dataset[index]


def dataset_statistics(dataset: object, dataset_name: str) -> Dict[str, object]:
    """计算图数量、节点数、边数、密度和类别分布等摘要指标。"""

    node_counts: List[int] = []
    edge_counts: List[float] = []
    avg_degrees: List[float] = []
    densities: List[float] = []
    class_hist: Counter = Counter()

    sampled_graphs = 0
    for graph in sampled_graph_iter(dataset):
        num_nodes = int(graph.num_nodes)
        directed_edges = int(graph.edge_index.size(1))
        undirected_edges = directed_edges / 2 if graph.is_undirected() else directed_edges
        density_denominator = max(num_nodes * max(num_nodes - 1, 1) / 2, 1)
        density = undirected_edges / density_denominator if num_nodes > 1 else 0.0
        avg_degree = directed_edges / max(num_nodes, 1)

        node_counts.append(num_nodes)
        edge_counts.append(undirected_edges)
        avg_degrees.append(avg_degree)
        densities.append(density)
        class_hist[int(graph_target(graph)[0])] += 1
        sampled_graphs += 1

    return {
        "dataset": dataset_name,
        "dataset_family": dataset_family(dataset_name),
        "graphs": len(dataset),
        "classes": dataset.num_classes,
        "num_features": infer_input_dim(dataset),
        "avg_nodes": float(np.mean(node_counts)),
        "std_nodes": float(np.std(node_counts)),
        "avg_edges": float(np.mean(edge_counts)),
        "avg_degree": float(np.mean(avg_degrees)),
        "avg_density": float(np.mean(densities)),
        "min_nodes": int(np.min(node_counts)),
        "max_nodes": int(np.max(node_counts)),
        "sampled_graphs": sampled_graphs,
        "class_hist": dict(sorted(class_hist.items())),
    }


def build_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    """为 GIN 算子构造一个两层感知机。"""

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
    )


def build_operator(name: str, in_channels: int, out_channels: int) -> nn.Module:
    """按名称创建消息传递算子。"""

    if name == "GCNConv":
        return GCNConv(in_channels, out_channels)
    if name == "GATConv":
        return GATConv(in_channels, out_channels)
    if name == "TransformerConv":
        return TransformerConv(in_channels, out_channels)
    if name == "SAGEConv":
        return SAGEConv(in_channels, out_channels)
    if name == "GINConv":
        return GINConv(build_mlp(in_channels, out_channels))
    raise ValueError(f"Unsupported operator: {name}")


class PlainBlock(nn.Module):
    """最基础的图分类块。

    结构为：
    输入算子 -> 若干隐藏图卷积层 -> global mean pooling -> 分类器。
    当 `res_graph=True` 时，会在每一层注入图级隐状态。
    """

    def __init__(
        self,
        hidden_channels: int,
        dataset: TUDataset,
        hidden_layers: int,
        operator: str,
        dropout: float,
        res_graph: bool = False,
        gate_init: float = 0.8,
    ):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.input_layer = build_operator(operator, input_dim, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [build_operator(operator, hidden_channels, hidden_channels) for _ in range(hidden_layers)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout
        self.res_graph = res_graph
        self.graph_gate = LearnableGate(init_probability=gate_init)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, graph_hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行单个 PlainBlock 前向传播，并返回 logits 与图嵌入。"""

        x = F.relu(self.input_layer(x, edge_index))
        gate = self.graph_gate()

        for layer in self.hidden_layers:
            # 关键路径：图级状态通过 batch 索引广播回节点表征。
            if self.res_graph and graph_hidden is not None:
                x = x + gate * graph_hidden[batch]
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = global_mean_pool(x, batch)
        if graph_hidden is not None:
            graph_embedding = graph_embedding + gate * graph_hidden
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class NodeResGNN(nn.Module):
    """节点级残差模型。

    当前层输入会显式叠加前一层缓存，从而缓解深层训练中的信息衰减。
    """

    def __init__(
        self,
        hidden_channels: int,
        dataset: TUDataset,
        hidden_layers: int,
        operator: str,
        dropout: float,
        gate_init: float = 0.8,
    ):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.input_layer = build_operator(operator, input_dim, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [build_operator(operator, hidden_channels, hidden_channels) for _ in range(hidden_layers)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout
        self.residual_gate = LearnableGate(init_probability=gate_init)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向时将 `current + previous` 送入下一层，形成节点级残差。"""

        current = F.relu(self.input_layer(x, edge_index))
        previous = torch.zeros_like(current)
        gate = self.residual_gate()

        for layer in self.hidden_layers:
            cached = current
            # 关键路径：上一层缓存 `previous` 作为残差支路参与卷积。
            current = F.relu(layer(current + gate * previous, edge_index))
            current = F.dropout(current, p=self.dropout, training=self.training)
            previous = cached

        graph_embedding = global_mean_pool(current, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class NodeCrossGNN(nn.Module):
    """节点级双分支交叉模型。

    两个分支分别维护自己的历史状态，并在每一步交叉注入对方的上一轮表示。
    """

    def __init__(
        self,
        hidden_channels: int,
        dataset: TUDataset,
        hidden_layers: int,
        operator: str,
        dropout: float,
        gate_init: float = 0.8,
    ):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.input_layer_1 = build_operator(operator, input_dim, hidden_channels)
        self.input_layer_2 = build_operator(operator, input_dim, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [build_operator(operator, hidden_channels, hidden_channels) for _ in range(hidden_layers * 2)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout
        self.cross_gate = LearnableGate(init_probability=gate_init)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """双分支交替更新，并在图池化前把两个分支的结果相加。"""

        branch_1 = F.relu(self.input_layer_1(x, edge_index))
        branch_2 = F.relu(self.input_layer_2(x, edge_index))
        prev_1 = torch.zeros_like(branch_1)
        prev_2 = torch.zeros_like(branch_2)
        gate = self.cross_gate()

        layer_id = 0
        while layer_id < len(self.hidden_layers):
            cache_1 = branch_1
            cache_2 = branch_2
            # 关键路径：branch_1 接收 prev_2，branch_2 接收 prev_1，形成交叉残差。
            branch_1 = F.relu(self.hidden_layers[layer_id](branch_1 + gate * prev_2, edge_index))
            branch_1 = F.dropout(branch_1, p=self.dropout, training=self.training)
            branch_2 = F.relu(self.hidden_layers[layer_id + 1](branch_2 + gate * prev_1, edge_index))
            branch_2 = F.dropout(branch_2, p=self.dropout, training=self.training)
            prev_1 = cache_1
            prev_2 = cache_2
            layer_id += 2

        graph_embedding = global_mean_pool(branch_1 + branch_2, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class GraphCondGNN(nn.Module):
    """图级条件模型。

    第一段 block 先生成图隐状态，第二段 block 再以该图隐状态作为条件继续编码。
    """

    def __init__(
        self,
        hidden_channels: int,
        dataset: TUDataset,
        hidden_layers: int,
        operator: str,
        dropout: float,
        gate_init: float = 0.8,
    ):
        super().__init__()
        self.block_1 = PlainBlock(
            hidden_channels,
            dataset,
            hidden_layers,
            operator,
            dropout,
            res_graph=False,
            gate_init=gate_init,
        )
        self.block_2 = PlainBlock(
            hidden_channels,
            dataset,
            hidden_layers,
            operator,
            dropout,
            res_graph=False,
            gate_init=gate_init,
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """先估计图级上下文，再进行二次编码。"""

        _, graph_hidden = self.block_1(x, edge_index, batch)
        _, graph_embedding = self.block_2(x, edge_index, batch, graph_hidden)
        return self.classifier(graph_embedding), graph_embedding


class GraphResGNN(nn.Module):
    """图级残差模型。

    多个 PlainBlock 串联，后一个 block 接收前一个 block 的图嵌入作为残差条件。
    """

    def __init__(
        self,
        hidden_channels: int,
        dataset: TUDataset,
        hidden_layers: int,
        operator: str,
        dropout: float,
        gate_init: float = 0.8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                PlainBlock(
                    hidden_channels,
                    dataset,
                    hidden_layers,
                    operator,
                    dropout,
                    res_graph=True,
                    gate_init=gate_init,
                )
                for _ in range(3)
            ]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """串联图级 block，并逐步更新图隐状态。"""

        graph_hidden = None
        graph_embedding = None
        for block in self.blocks:
            _, graph_embedding = block(x, edge_index, batch, graph_hidden)
            graph_hidden = graph_embedding
        return self.classifier(graph_embedding), graph_embedding


class GraphCrossGNN(nn.Module):
    """图级双分支交叉模型。

    以 block 为单位维护两条图级状态支路，并在相邻 block 间交叉交换图隐状态。
    """

    def __init__(
        self,
        hidden_channels: int,
        dataset: TUDataset,
        hidden_layers: int,
        operator: str,
        dropout: float,
        gate_init: float = 0.8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                PlainBlock(
                    hidden_channels,
                    dataset,
                    hidden_layers,
                    operator,
                    dropout,
                    res_graph=False,
                    gate_init=gate_init,
                )
                for _ in range(4)
            ]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """每两个 block 为一组做交叉更新，最后融合两条图级状态。"""

        graph_hidden_1 = None
        graph_hidden_2 = None
        block_id = 0

        while block_id < len(self.blocks):
            _, current_1 = self.blocks[block_id](x, edge_index, batch, graph_hidden_1)
            _, current_2 = self.blocks[block_id + 1](x, edge_index, batch, graph_hidden_2)
            # 关键路径：当前块产出的图表示被交换给另一条支路作为下一轮条件。
            graph_hidden_1 = current_2
            graph_hidden_2 = current_1
            block_id += 2

        graph_embedding = graph_hidden_1 + graph_hidden_2
        logits = self.classifier(F.dropout(graph_embedding, p=self.blocks[0].dropout, training=self.training))
        return logits, graph_embedding


class GraphSAGEBaseline(nn.Module):
    """外部基线 1：标准 GraphSAGE 图分类模型。"""

    def __init__(self, hidden_channels: int, dataset: TUDataset, hidden_layers: int, dropout: float):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.input_layer = SAGEConv(input_dim, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(hidden_layers)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """标准 GraphSAGE 前向传播。"""

        x = F.relu(self.input_layer(x, edge_index))
        for layer in self.hidden_layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(x, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class GINBaseline(nn.Module):
    """外部基线 2：GIN 图分类模型。"""

    def __init__(self, hidden_channels: int, dataset: TUDataset, hidden_layers: int, dropout: float):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.input_layer = GINConv(build_mlp(input_dim, hidden_channels))
        self.hidden_layers = nn.ModuleList(
            [GINConv(build_mlp(hidden_channels, hidden_channels)) for _ in range(hidden_layers)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """标准 GIN 前向传播。"""

        x = F.relu(self.input_layer(x, edge_index))
        for layer in self.hidden_layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(x, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class JKNetBaseline(nn.Module):
    """外部基线 3：Jumping Knowledge Network。

    `jk_mode` 默认是 `cat`，会把各层表征拼接后再做线性投影。
    """

    def __init__(self, hidden_channels: int, dataset: TUDataset, hidden_layers: int, dropout: float, operator: str, jk_mode: str):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        base_operator = operator if operator in {"GCNConv", "SAGEConv"} else "GCNConv"
        self.input_layer = build_operator(base_operator, input_dim, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [build_operator(base_operator, hidden_channels, hidden_channels) for _ in range(hidden_layers)]
        )
        self.jump = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=hidden_layers + 1)
        jump_dim = hidden_channels if jk_mode != "cat" else hidden_channels * (hidden_layers + 1)
        self.projection = nn.Linear(jump_dim, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """收集各层输出后通过 JumpingKnowledge 汇聚。"""

        outputs = []
        x = F.relu(self.input_layer(x, edge_index))
        outputs.append(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            outputs.append(x)
        x = self.jump(outputs)
        x = F.relu(self.projection(x))
        graph_embedding = global_mean_pool(x, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class APPNPBaseline(nn.Module):
    """外部基线 4：APPNP。

    默认传播超参数固定为 `K=10`、`alpha=0.1`。
    """

    def __init__(self, hidden_channels: int, dataset: TUDataset, dropout: float):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.lin_in = nn.Linear(input_dim, hidden_channels)
        self.lin_hidden = nn.Linear(hidden_channels, hidden_channels)
        self.propagation = APPNP(K=10, alpha=0.1)
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """先做两层 MLP，再执行 APPNP 传播。"""

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_hidden(x))
        x = self.propagation(x, edge_index)
        graph_embedding = global_mean_pool(x, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


def build_model(args: argparse.Namespace, dataset: TUDataset) -> nn.Module:
    """根据参数构造具体模型实例。"""

    common_kwargs = {
        "hidden_channels": args.dim,
        "dataset": dataset,
        "hidden_layers": args.h_layer,
        "dropout": args.drop,
    }
    gated_kwargs = {**common_kwargs, "gate_init": args.gate_init}

    if args.gname == "PlainGNN":
        return PlainBlock(operator=args.name, res_graph=False, **gated_kwargs)
    if args.gname == "NodeResGNN":
        return NodeResGNN(operator=args.name, **gated_kwargs)
    if args.gname == "NodeCrossGNN":
        return NodeCrossGNN(operator=args.name, **gated_kwargs)
    if args.gname == "GraphCondGNN":
        return GraphCondGNN(operator=args.name, **gated_kwargs)
    if args.gname == "GraphResGNN":
        return GraphResGNN(operator=args.name, **gated_kwargs)
    if args.gname == "GraphCrossGNN":
        return GraphCrossGNN(operator=args.name, **gated_kwargs)
    if args.gname == "GraphSAGEBaseline":
        return GraphSAGEBaseline(**common_kwargs)
    if args.gname == "GINBaseline":
        return GINBaseline(**common_kwargs)
    if args.gname == "JKNetBaseline":
        return JKNetBaseline(operator=args.name, jk_mode=args.jk_mode, **common_kwargs)
    if args.gname == "APPNPBaseline":
        return APPNPBaseline(hidden_channels=args.dim, dataset=dataset, dropout=args.drop)
    raise ValueError(f"Unsupported architecture: {args.gname}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计总参数量、可训练参数量和冻结参数量。"""

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }


def gradient_norm(model: nn.Module) -> float:
    """计算当前参数梯度的 L2 范数，用于训练诊断。"""

    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_value = float(parameter.grad.detach().norm(2).item())
        squared_norm += grad_value * grad_value
    return squared_norm ** 0.5


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """返回张量的均值、标准差和最大值，便于监控数值稳定性。"""

    if tensor.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    detached = tensor.detach()
    return {
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()) if detached.numel() > 1 else 0.0,
        "max": float(detached.max().item()),
    }


def collect_gate_values(model: nn.Module) -> Dict[str, float]:
    """提取当前模型中所有可学习门限的 sigmoid 值。"""

    gate_values: Dict[str, float] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LearnableGate):
            gate_values[module_name] = float(module().detach().item())
    return gate_values


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    """在普通分类任务上评估模型，返回损失与准确率。"""

    model.eval()
    total_correct = 0
    total_loss = 0.0
    total_graphs = 0

    with torch.no_grad():
        for batch_data in loader:
            batch_data = prepare_batch(batch_data)
            targets = graph_target(batch_data)
            logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_correct += int((predictions == targets).sum())
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size

    return {
        "loss": total_loss / max(total_graphs, 1),
        "acc": total_correct / max(total_graphs, 1),
    }


def evaluate_ogb(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    evaluator: Evaluator,
) -> Dict[str, float]:
    """在 OGB 图属性预测数据集上评估模型。"""

    model.eval()
    total_loss = 0.0
    total_graphs = 0
    y_true_parts: List[torch.Tensor] = []
    y_pred_parts: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_data in loader:
            batch_data = prepare_batch(batch_data)
            targets = graph_target(batch_data)
            logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1, keepdim=True)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size
            y_true_parts.append(targets.view(-1, 1).detach().cpu())
            y_pred_parts.append(predictions.detach().cpu())

    y_true = torch.cat(y_true_parts, dim=0).numpy()
    y_pred = torch.cat(y_pred_parts, dim=0).numpy()
    metric_name = evaluator.eval_metric
    metric_value = float(evaluator.eval({"y_true": y_true, "y_pred": y_pred})[metric_name])
    return {
        "loss": total_loss / max(total_graphs, 1),
        "acc": metric_value,
        "metric_name": metric_name,
    }


def train_one_config(args: argparse.Namespace) -> Dict[str, object]:
    """执行单个配置的完整训练流程。

    主要步骤：
    1. 设定随机种子并加载数据集
    2. 生成训练/验证/测试划分
    3. 训练模型并记录早停、学习率与数值诊断
    4. 在最佳权重上做最终测试并导出 JSON
    """

    family = dataset_family(args.ds)
    version = normalize_version(args.version)
    version_log_dir = log_dir(PROJECT_ROOT, version)
    version_run_dir = run_dir(PROJECT_ROOT, version)
    effective_seed = args.seed + args.fold if family == "ogb_graphprop" else args.seed
    set_seed(effective_seed)
    dataset = load_dataset(args.ds)
    stats = dataset_statistics(dataset, args.ds)

    if args.report_dataset_stats:
        print(json.dumps(stats, indent=2))
    if args.save_dataset_stats:
        save_json(stats, prefix=f"dataset_stats_{args.ds}", debug=args.debug, output_dir=version_log_dir)
    if args.mode == "stats":
        return {"dataset_stats": stats}

    train_dataset, test_dataset, official_val_dataset, split_context = split_dataset(dataset, args.fold, args.ds)
    if official_val_dataset is None:
        train_dataset, val_dataset = split_train_val_dataset(train_dataset, args.val_ratio, seed=args.seed + args.fold)
    else:
        val_dataset = official_val_dataset
    train_loader = build_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_loader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None
    test_loader = build_loader(test_dataset, batch_size=args.batch_size, shuffle=False)
    evaluator = Evaluator(name=args.ds) if family == "ogb_graphprop" and Evaluator is not None else None

    model = build_model(args, dataset).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    parameter_stats = count_parameters(model)

    print(
        format_metrics(
            model=args.gname,
            operator=args.name,
            dataset=args.ds,
            version=version,
            fold=args.fold,
            dataset_family=family,
            total_params=parameter_stats["total_params"],
            trainable_params=parameter_stats["trainable_params"],
            gate_init=args.gate_init,
        )
    )

    writer = None
    if args.tensorboard and SummaryWriter is not None:
        tb_prefix = with_exp_tag(
            f"{args.gname}_{args.name}_{args.ds}_{args.dim}_fold{args.fold}_{args.h_layer}",
            args.exp_tag,
        )
        tb_log_dir = version_run_dir / f"{tb_prefix}_{timestamp()}"
        writer = SummaryWriter(str(tb_log_dir))

    history: List[Dict[str, float]] = []
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_test_acc = -1.0
    best_state_dict = deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(args.ep):
        model.train()
        total_correct = 0
        total_loss = 0.0
        total_graphs = 0
        grad_norm_sum = 0.0
        grad_norm_steps = 0
        embedding_abs_mean_sum = 0.0
        embedding_abs_max = 0.0
        embedding_std_sum = 0.0
        logits_abs_mean_sum = 0.0
        logits_abs_max = 0.0

        for batch_data in train_loader:
            batch_data = prepare_batch(batch_data)
            targets = graph_target(batch_data)
            optimizer.zero_grad()
            logits, graph_embedding = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, targets)
            loss.backward()
            batch_grad_norm = gradient_norm(model)
            # 关键步骤：梯度裁剪用于抑制深层/交叉结构训练中的梯度爆炸。
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_correct += int((predictions == targets).sum())
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size
            grad_norm_sum += batch_grad_norm
            grad_norm_steps += 1

            embedding_stats = tensor_stats(graph_embedding.abs())
            logits_stats = tensor_stats(logits.abs())
            embedding_abs_mean_sum += embedding_stats["mean"] * batch_size
            embedding_std_sum += float(graph_embedding.detach().std(unbiased=False).item()) * batch_size
            embedding_abs_max = max(embedding_abs_max, embedding_stats["max"])
            logits_abs_mean_sum += logits_stats["mean"] * batch_size
            logits_abs_max = max(logits_abs_max, logits_stats["max"])

        train_metrics = {
            "loss": total_loss / max(total_graphs, 1),
            "acc": total_correct / max(total_graphs, 1),
        }
        train_diagnostics = {
            "grad_norm": grad_norm_sum / max(grad_norm_steps, 1),
            "embedding_abs_mean": embedding_abs_mean_sum / max(total_graphs, 1),
            "embedding_std": embedding_std_sum / max(total_graphs, 1),
            "embedding_abs_max": embedding_abs_max,
            "logits_abs_mean": logits_abs_mean_sum / max(total_graphs, 1),
            "logits_abs_max": logits_abs_max,
        }
        if val_loader is not None:
            if evaluator is not None:
                val_metrics = evaluate_ogb(model, val_loader, criterion, evaluator)
            else:
                val_metrics = evaluate(model, val_loader, criterion)
        else:
            val_metrics = train_metrics
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        gate_values = collect_gate_values(model)

        # 以验证损失作为早停基准，`min_delta` 控制最小改进幅度。
        improved = val_metrics["loss"] < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "grad_norm": train_diagnostics["grad_norm"],
            "embedding_abs_mean": train_diagnostics["embedding_abs_mean"],
            "embedding_std": train_diagnostics["embedding_std"],
            "embedding_abs_max": train_diagnostics["embedding_abs_max"],
            "logits_abs_mean": train_diagnostics["logits_abs_mean"],
            "logits_abs_max": train_diagnostics["logits_abs_max"],
            "lr": current_lr,
            "patience_counter": patience_counter,
            "gate_values": gate_values,
        }
        history.append(epoch_record)

        if writer is not None:
            writer.add_scalar("loss/train", train_metrics["loss"], epoch)
            if val_loader is not None:
                writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("acc/train", train_metrics["acc"], epoch)
            if val_loader is not None:
                writer.add_scalar("acc/val", val_metrics["acc"], epoch)
            writer.add_scalar("optim/lr", current_lr, epoch)
            writer.add_scalar("optim/grad_norm", train_diagnostics["grad_norm"], epoch)
            writer.add_scalar("embedding/abs_mean", train_diagnostics["embedding_abs_mean"], epoch)
            writer.add_scalar("embedding/std", train_diagnostics["embedding_std"], epoch)
            writer.add_scalar("embedding/abs_max", train_diagnostics["embedding_abs_max"], epoch)
            writer.add_scalar("logits/abs_mean", train_diagnostics["logits_abs_mean"], epoch)
            writer.add_scalar("logits/abs_max", train_diagnostics["logits_abs_max"], epoch)
            for gate_name, gate_value in gate_values.items():
                writer.add_scalar(f"gates/{gate_name}", gate_value, epoch)
            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == args.ep - 1:
                writer.add_histogram("embedding/graph_embedding", graph_embedding.detach().cpu(), epoch)
                writer.add_histogram("logits/logits", logits.detach().cpu(), epoch)

        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == args.ep - 1:
            print(
                f"[epoch {epoch + 1:04d}] "
                + format_metrics(
                    train_loss=train_metrics["loss"],
                    train_acc=train_metrics["acc"],
                    val_loss=val_metrics["loss"],
                    val_acc=val_metrics["acc"],
                    lr=current_lr,
                    patience=patience_counter,
                    **gate_values,
                )
            )

        if patience_counter >= args.patience:
            print(
                f"[early_stop] epoch={epoch + 1} best_epoch={best_epoch + 1} "
                f"best_val_loss={best_val_loss:.5f}"
            )
            break

    if writer is not None:
        writer.close()

    model.load_state_dict(best_state_dict)
    if evaluator is not None:
        test_metrics = evaluate_ogb(model, test_loader, criterion, evaluator)
    else:
        test_metrics = evaluate(model, test_loader, criterion)
    best_test_acc = test_metrics["acc"]

    summary = {
        "config": vars(args),
        "version": version,
        "dataset_stats": stats,
        "split_context": split_context,
        "parameter_stats": parameter_stats,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "test_loss": test_metrics["loss"],
        "eval_metric": test_metrics.get("metric_name", "acc"),
        "history": history,
    }
    save_json(
        summary,
        prefix=with_exp_tag(f"train_{args.ds}_{args.gname}_{args.name}_fold{args.fold}", args.exp_tag),
        debug=args.debug,
        output_dir=version_log_dir,
    )
    return summary


def suite_configs(base_args: argparse.Namespace) -> List[argparse.Namespace]:
    """把一个 suite 名称展开成多组待运行配置。"""

    configs: List[argparse.Namespace] = []

    if base_args.suite_name == "external_baselines":
        targets = [
            ("PlainGNN", base_args.name),
            ("GraphSAGEBaseline", "SAGEConv"),
            ("GINBaseline", "GINConv"),
            ("JKNetBaseline", "GCNConv"),
            ("APPNPBaseline", "GCNConv"),
        ]
    elif base_args.suite_name == "node_residual":
        targets = [
            ("PlainGNN", base_args.name),
            ("NodeResGNN", base_args.name),
            ("NodeCrossGNN", base_args.name),
        ]
    elif base_args.suite_name == "graph_residual":
        targets = [
            ("GraphCondGNN", base_args.name),
            ("GraphResGNN", base_args.name),
            ("GraphCrossGNN", base_args.name),
        ]
    else:
        targets = [(base_args.gname, base_args.name)]

    if base_args.suite_name == "depth_ablation":
        for depth in [1, 2, 3, 4, 5]:
            config = deepcopy(base_args)
            config.gname = "NodeResGNN"
            config.h_layer = depth
            config.mode = "single"
            configs.append(config)
        return configs

    for architecture, operator in targets:
        config = deepcopy(base_args)
        config.gname = architecture
        config.name = operator
        config.mode = "single"
        configs.append(config)
    return configs


def run_suite(args: argparse.Namespace) -> List[Dict[str, object]]:
    """顺序运行一组实验配置，并把结果汇总成简短文本。"""

    results: List[Dict[str, object]] = []
    lines: List[str] = []
    version_record_dir = record_dir(PROJECT_ROOT, normalize_version(args.version))

    for config in suite_configs(args):
        config = canonical_args(config)
        result = train_one_config(config)
        results.append(result)
        if "best_test_acc" in result:
            lines.append(
                format_metrics(
                    suite=args.suite_name,
                    dataset=config.ds,
                    fold=config.fold,
                    model=config.gname,
                    operator=config.name,
                    depth=config.h_layer,
                    best_test_acc=result["best_test_acc"],
                    total_params=result["parameter_stats"]["total_params"],
                )
            )

    save_lines(lines, prefix=f"suite_{args.suite_name}_{args.ds}", debug=args.debug, output_dir=version_record_dir)
    return results


def main() -> None:
    """脚本入口。

    默认模式是 `single`，也就是只训练一个配置；`suite` 模式会跑预设实验组。
    """

    ensure_dirs()
    args = canonical_args(parse_args())
    args.version = normalize_version(args.version)
    for path in [
        log_dir(PROJECT_ROOT, args.version),
        record_dir(PROJECT_ROOT, args.version),
        run_dir(PROJECT_ROOT, args.version),
    ]:
        path.mkdir(parents=True, exist_ok=True)

    if args.gname in FAMILY_MODELS and args.name not in AVAILABLE_OPERATORS:
        raise ValueError(f"{args.name} is not available. Choose from {AVAILABLE_OPERATORS}.")

    start = time.time()
    if args.mode == "suite":
        run_suite(args)
    else:
        train_one_config(args)
    print(f"completed in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
