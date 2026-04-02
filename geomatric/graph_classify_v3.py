import argparse
import json
import os
import platform
import random
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RECORD_ROOT = PROJECT_ROOT / "records"
LOG_ROOT = PROJECT_ROOT / "logs"
RUN_ROOT = PROJECT_ROOT / "runs"
SEPARATOR = "__"

MODEL_ALIASES = {
    "BlockGNN": "PlainGNN",
    "ResBlockGnn": "NodeResGNN",
    "CrossBlockGnn": "NodeCrossGNN",
    "GraphBlockGnn": "GraphCondGNN",
    "ResGraphBlockGnn": "GraphResGNN",
    "CrossGraphBlockGnn": "GraphCrossGNN",
    "GraphSAGE": "GraphSAGEBaseline",
    "GIN": "GINBaseline",
    "JKNet": "JKNetBaseline",
    "APPNP": "APPNPBaseline",
}

FAMILY_MODELS = [
    "PlainGNN",
    "NodeResGNN",
    "NodeCrossGNN",
    "GraphCondGNN",
    "GraphResGNN",
    "GraphCrossGNN",
]

EXTERNAL_BASELINES = [
    "GraphSAGEBaseline",
    "GINBaseline",
    "JKNetBaseline",
    "APPNPBaseline",
]

AVAILABLE_OPERATORS = [
    "GCNConv",
    "GATConv",
    "TransformerConv",
    "SAGEConv",
    "GINConv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V3 graph classification benchmark with stronger external baselines."
    )
    parser.add_argument("--name", type=str, default="GCNConv", help="Message passing operator.")
    parser.add_argument("--gname", type=str, default="PlainGNN", help="Architecture name.")
    parser.add_argument("--ds", type=str, default="MUTAG", help="TUDataset name.")
    parser.add_argument("--ep", type=int, default=500, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--drop", type=float, default=0.6)
    parser.add_argument("--dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--h_layer", type=int, default=2, help="Number of hidden layers.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fold", type=int, default=0, help="Five-fold split index.")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio within the training split.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience on validation loss.")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum validation-loss improvement to reset patience.")
    parser.add_argument("--jk_mode", type=str, default="cat", choices=["cat", "max", "lstm"])
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "suite", "stats"],
        help="single=train one config, suite=run a preset ablation/baseline suite, stats=only export dataset stats",
    )
    parser.add_argument(
        "--suite_name",
        type=str,
        default="external_baselines",
        choices=["external_baselines", "node_residual", "graph_residual", "depth_ablation"],
    )
    parser.add_argument("--debug", action="store_true", help="Skip file writes.")
    parser.add_argument("--tensorboard", action="store_true", help="Write TensorBoard logs if available.")
    parser.add_argument("--report_dataset_stats", action="store_true", help="Print dataset statistics before training.")
    parser.add_argument("--save_dataset_stats", action="store_true", help="Save dataset statistics to records.")
    return parser.parse_args()


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def canonical_args(args: argparse.Namespace) -> argparse.Namespace:
    args.gname = resolve_model_name(args.gname)
    if args.name == "GraphSAGE":
        args.name = "SAGEConv"
    if args.name == "GIN":
        args.name = "GINConv"
    return args


def ensure_dirs() -> None:
    for path in [DATA_ROOT, RECORD_ROOT, LOG_ROOT, RUN_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_metrics(**metrics: object) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.5f}")
        else:
            parts.append(f"{key}={value}")
    return "\t".join(parts)


def save_json(payload: Dict[str, object], prefix: str, debug: bool) -> Optional[Path]:
    file_path = LOG_ROOT / f"{prefix}{SEPARATOR}{timestamp()}.json"
    if debug:
        print(f"[debug] skip save_json -> {file_path}")
        return file_path
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return file_path


def save_lines(lines: List[str], prefix: str, debug: bool) -> Optional[Path]:
    file_path = RECORD_ROOT / f"{prefix}{SEPARATOR}{timestamp()}.txt"
    if debug:
        print(f"[debug] skip save_lines -> {file_path}")
        return file_path
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return file_path


def infer_input_dim(dataset: TUDataset) -> int:
    return dataset.num_features if dataset.num_features > 0 else 1


def prepare_batch(data: torch.Tensor) -> torch.Tensor:
    if data.x is None:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
    return data.to(DEVICE)


def load_dataset(dataset_name: str) -> TUDataset:
    dataset = TUDataset(root=str(DATA_ROOT / "TUDataset"), name=dataset_name)
    return dataset.shuffle()


def split_dataset(dataset: TUDataset, fold: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    total_graphs = len(dataset)
    fold_size = total_graphs // 5
    fold_start = fold * fold_size
    fold_end = total_graphs if fold == 4 else (fold + 1) * fold_size
    test_dataset = list(dataset[fold_start:fold_end])
    train_dataset = list(dataset[:fold_start]) + list(dataset[fold_end:])
    return train_dataset, test_dataset


def split_train_val_dataset(
    train_dataset: List[torch.Tensor],
    val_ratio: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if not train_dataset:
        return [], []
    if val_ratio <= 0.0:
        return train_dataset, []

    val_size = max(1, int(round(len(train_dataset) * val_ratio)))
    val_size = min(val_size, len(train_dataset) - 1) if len(train_dataset) > 1 else 0
    if val_size == 0:
        return train_dataset, []
    val_dataset = list(train_dataset[-val_size:])
    inner_train_dataset = list(train_dataset[:-val_size])
    return inner_train_dataset, val_dataset


def build_loader(dataset_slice: List[torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset_slice, batch_size=batch_size, shuffle=shuffle)


def dataset_statistics(dataset: TUDataset) -> Dict[str, object]:
    node_counts: List[int] = []
    edge_counts: List[float] = []
    avg_degrees: List[float] = []
    densities: List[float] = []
    class_hist: Counter = Counter()

    for graph in dataset:
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
        class_hist[int(graph.y.view(-1)[0])] += 1

    return {
        "dataset": dataset.name,
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
        "class_hist": dict(sorted(class_hist.items())),
    }


def build_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
    )


def build_operator(name: str, in_channels: int, out_channels: int) -> nn.Module:
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
    def __init__(self, hidden_channels: int, dataset: TUDataset, hidden_layers: int, operator: str, dropout: float, res_graph: bool = False):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.input_layer = build_operator(operator, input_dim, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [build_operator(operator, hidden_channels, hidden_channels) for _ in range(hidden_layers)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout
        self.res_graph = res_graph

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, graph_hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.input_layer(x, edge_index))

        for layer in self.hidden_layers:
            if self.res_graph and graph_hidden is not None:
                x = x + graph_hidden[batch]
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = global_mean_pool(x, batch)
        if graph_hidden is not None:
            graph_embedding = graph_embedding + graph_hidden
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class NodeResGNN(nn.Module):
    def __init__(self, hidden_channels: int, dataset: TUDataset, hidden_layers: int, operator: str, dropout: float):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.input_layer = build_operator(operator, input_dim, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [build_operator(operator, hidden_channels, hidden_channels) for _ in range(hidden_layers)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current = F.relu(self.input_layer(x, edge_index))
        previous = torch.zeros_like(current)

        for layer in self.hidden_layers:
            cached = current
            current = F.relu(layer(current + previous, edge_index))
            current = F.dropout(current, p=self.dropout, training=self.training)
            previous = cached

        graph_embedding = global_mean_pool(current, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class NodeCrossGNN(nn.Module):
    def __init__(self, hidden_channels: int, dataset: TUDataset, hidden_layers: int, operator: str, dropout: float):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.input_layer_1 = build_operator(operator, input_dim, hidden_channels)
        self.input_layer_2 = build_operator(operator, input_dim, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [build_operator(operator, hidden_channels, hidden_channels) for _ in range(hidden_layers * 2)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        branch_1 = F.relu(self.input_layer_1(x, edge_index))
        branch_2 = F.relu(self.input_layer_2(x, edge_index))
        prev_1 = torch.zeros_like(branch_1)
        prev_2 = torch.zeros_like(branch_2)

        layer_id = 0
        while layer_id < len(self.hidden_layers):
            cache_1 = branch_1
            cache_2 = branch_2
            branch_1 = F.relu(self.hidden_layers[layer_id](branch_1 + prev_2, edge_index))
            branch_1 = F.dropout(branch_1, p=self.dropout, training=self.training)
            branch_2 = F.relu(self.hidden_layers[layer_id + 1](branch_2 + prev_1, edge_index))
            branch_2 = F.dropout(branch_2, p=self.dropout, training=self.training)
            prev_1 = cache_1
            prev_2 = cache_2
            layer_id += 2

        graph_embedding = global_mean_pool(branch_1 + branch_2, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class GraphCondGNN(nn.Module):
    def __init__(self, hidden_channels: int, dataset: TUDataset, hidden_layers: int, operator: str, dropout: float):
        super().__init__()
        self.block_1 = PlainBlock(hidden_channels, dataset, hidden_layers, operator, dropout, res_graph=False)
        self.block_2 = PlainBlock(hidden_channels, dataset, hidden_layers, operator, dropout, res_graph=False)
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, graph_hidden = self.block_1(x, edge_index, batch)
        _, graph_embedding = self.block_2(x, edge_index, batch, graph_hidden)
        return self.classifier(graph_embedding), graph_embedding


class GraphResGNN(nn.Module):
    def __init__(self, hidden_channels: int, dataset: TUDataset, hidden_layers: int, operator: str, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PlainBlock(hidden_channels, dataset, hidden_layers, operator, dropout, res_graph=True) for _ in range(3)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_hidden = None
        graph_embedding = None
        for block in self.blocks:
            _, graph_embedding = block(x, edge_index, batch, graph_hidden)
            graph_hidden = graph_embedding
        return self.classifier(graph_embedding), graph_embedding


class GraphCrossGNN(nn.Module):
    def __init__(self, hidden_channels: int, dataset: TUDataset, hidden_layers: int, operator: str, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PlainBlock(hidden_channels, dataset, hidden_layers, operator, dropout, res_graph=False) for _ in range(4)]
        )
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_hidden_1 = None
        graph_hidden_2 = None
        block_id = 0

        while block_id < len(self.blocks):
            _, current_1 = self.blocks[block_id](x, edge_index, batch, graph_hidden_1)
            _, current_2 = self.blocks[block_id + 1](x, edge_index, batch, graph_hidden_2)
            graph_hidden_1 = current_2
            graph_hidden_2 = current_1
            block_id += 2

        graph_embedding = graph_hidden_1 + graph_hidden_2
        logits = self.classifier(F.dropout(graph_embedding, p=self.blocks[0].dropout, training=self.training))
        return logits, graph_embedding


class GraphSAGEBaseline(nn.Module):
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
        x = F.relu(self.input_layer(x, edge_index))
        for layer in self.hidden_layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(x, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class GINBaseline(nn.Module):
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
        x = F.relu(self.input_layer(x, edge_index))
        for layer in self.hidden_layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_mean_pool(x, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


class JKNetBaseline(nn.Module):
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
    def __init__(self, hidden_channels: int, dataset: TUDataset, dropout: float):
        super().__init__()
        input_dim = infer_input_dim(dataset)
        self.lin_in = nn.Linear(input_dim, hidden_channels)
        self.lin_hidden = nn.Linear(hidden_channels, hidden_channels)
        self.propagation = APPNP(K=10, alpha=0.1)
        self.classifier = nn.Linear(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_hidden(x))
        x = self.propagation(x, edge_index)
        graph_embedding = global_mean_pool(x, batch)
        logits = self.classifier(F.dropout(graph_embedding, p=self.dropout, training=self.training))
        return logits, graph_embedding


def build_model(args: argparse.Namespace, dataset: TUDataset) -> nn.Module:
    common_kwargs = {
        "hidden_channels": args.dim,
        "dataset": dataset,
        "hidden_layers": args.h_layer,
        "dropout": args.drop,
    }

    if args.gname == "PlainGNN":
        return PlainBlock(operator=args.name, res_graph=False, **common_kwargs)
    if args.gname == "NodeResGNN":
        return NodeResGNN(operator=args.name, **common_kwargs)
    if args.gname == "NodeCrossGNN":
        return NodeCrossGNN(operator=args.name, **common_kwargs)
    if args.gname == "GraphCondGNN":
        return GraphCondGNN(operator=args.name, **common_kwargs)
    if args.gname == "GraphResGNN":
        return GraphResGNN(operator=args.name, **common_kwargs)
    if args.gname == "GraphCrossGNN":
        return GraphCrossGNN(operator=args.name, **common_kwargs)
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
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_correct = 0
    total_loss = 0.0
    total_graphs = 0

    with torch.no_grad():
        for batch_data in loader:
            batch_data = prepare_batch(batch_data)
            logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, batch_data.y)
            predictions = logits.argmax(dim=1)
            batch_size = batch_data.y.size(0)
            total_correct += int((predictions == batch_data.y).sum())
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size

    return {
        "loss": total_loss / max(total_graphs, 1),
        "acc": total_correct / max(total_graphs, 1),
    }


def train_one_config(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    dataset = load_dataset(args.ds)
    stats = dataset_statistics(dataset)

    if args.report_dataset_stats:
        print(json.dumps(stats, indent=2))
    if args.save_dataset_stats:
        save_json(stats, prefix=f"dataset_stats_{args.ds}", debug=args.debug)
    if args.mode == "stats":
        return {"dataset_stats": stats}

    train_dataset, test_dataset = split_dataset(dataset, args.fold)
    train_dataset, val_dataset = split_train_val_dataset(train_dataset, args.val_ratio)
    train_loader = build_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_loader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None
    test_loader = build_loader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args, dataset).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    parameter_stats = count_parameters(model)

    print(
        format_metrics(
            model=args.gname,
            operator=args.name,
            dataset=args.ds,
            fold=args.fold,
            total_params=parameter_stats["total_params"],
            trainable_params=parameter_stats["trainable_params"],
        )
    )

    writer = None
    if args.tensorboard and SummaryWriter is not None:
        log_dir = RUN_ROOT / f"{args.gname}_{args.name}_{args.ds}_{args.dim}_{args.h_layer}_{timestamp()}"
        writer = SummaryWriter(str(log_dir))

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

        for batch_data in train_loader:
            batch_data = prepare_batch(batch_data)
            optimizer.zero_grad()
            logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, batch_data.y)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = batch_data.y.size(0)
            total_correct += int((predictions == batch_data.y).sum())
            total_loss += float(loss.item()) * batch_size
            total_graphs += batch_size

        train_metrics = {
            "loss": total_loss / max(total_graphs, 1),
            "acc": total_correct / max(total_graphs, 1),
        }
        val_metrics = evaluate(model, val_loader, criterion) if val_loader is not None else train_metrics

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
            "patience_counter": patience_counter,
        }
        history.append(epoch_record)

        if writer is not None:
            writer.add_scalar("loss/train", train_metrics["loss"], epoch)
            if val_loader is not None:
                writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("acc/train", train_metrics["acc"], epoch)
            if val_loader is not None:
                writer.add_scalar("acc/val", val_metrics["acc"], epoch)

        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == args.ep - 1:
            print(
                f"[epoch {epoch + 1:04d}] "
                + format_metrics(
                    train_loss=train_metrics["loss"],
                    train_acc=train_metrics["acc"],
                    val_loss=val_metrics["loss"],
                    val_acc=val_metrics["acc"],
                    patience=patience_counter,
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
    test_metrics = evaluate(model, test_loader, criterion)
    best_test_acc = test_metrics["acc"]

    summary = {
        "config": vars(args),
        "dataset_stats": stats,
        "parameter_stats": parameter_stats,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "test_loss": test_metrics["loss"],
        "history": history,
    }
    save_json(
        summary,
        prefix=f"train_{args.ds}_{args.gname}_{args.name}_fold{args.fold}",
        debug=args.debug,
    )
    return summary


def suite_configs(base_args: argparse.Namespace) -> List[argparse.Namespace]:
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
    results: List[Dict[str, object]] = []
    lines: List[str] = []

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

    save_lines(lines, prefix=f"suite_{args.suite_name}_{args.ds}", debug=args.debug)
    return results


def main() -> None:
    ensure_dirs()
    args = canonical_args(parse_args())

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
