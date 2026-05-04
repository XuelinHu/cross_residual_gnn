"""
Supplementary experiments: hidden-state similarity (CKA/cosine) and
per-layer gradient analysis for all 5 CR-GNN models.

Experiment A: Layer-wise representation similarity
  - Captures intermediate graph embeddings after each depth stage
  - Computes CKA and cosine similarity between adjacent stages
  - Plots depth vs. similarity curves for all 5 models

Experiment B: Per-layer gradient norm statistics
  - Runs one training epoch with gradient hooks
  - Records per-layer gradient L2 norms after each batch
  - Plots layer-wise gradient distribution for all 5 models

Usage:
  python py/run_supplementary_experiments.py --ds PROTEINS --operator GCNConv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.graph_classify_v3 import (
    DEVICE,
    apply_residual_mode,
    build_model,
    load_dataset,
    split_dataset,
    build_loader,
    set_seed,
)

OUTPUT_DIR = ROOT / "figures" / "exp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = ROOT / "records" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

FIVE_MODELS = ["PlainGNN", "NodeResGNN", "NodeCrossGNN", "GraphResGNN", "GraphCrossGNN"]


# ──────────────────────────────────────────────
# Experiment A: Hidden-state similarity
# ──────────────────────────────────────────────

def _global_pool_states(
    states: List[torch.Tensor], batch: torch.Tensor
) -> List[np.ndarray]:
    """Global-mean-pool each node-state tensor to a graph embedding."""
    from torch_geometric.nn import global_mean_pool

    return [
        global_mean_pool(s, batch).detach().cpu().numpy() for s in states
    ]


def capture_node_level_states(
    model: nn.Module,
    data,
    num_layers: int = 4,
) -> List[np.ndarray]:
    """Capture intermediate graph embeddings for node-level models.

    Hooks onto input_layer and each hidden_layers[i] to capture
    post-convolution node states, then pools each to a graph embedding.

    Returns:
        List of [num_graphs, hidden_dim] arrays, one per depth stage.
    """
    activations: List[torch.Tensor] = []

    def make_hook(store: List):
        def hook(module, inp, out):
            store.append(out)
        return hook

    handles = []

    # Hook input layer
    if hasattr(model, "input_layer"):
        handles.append(model.input_layer.register_forward_hook(make_hook(activations)))
    elif hasattr(model, "input_layer_1"):
        handles.append(model.input_layer_1.register_forward_hook(make_hook(activations)))

    # Hook hidden layers
    if hasattr(model, "hidden_layers"):
        for layer in model.hidden_layers:
            handles.append(layer.register_forward_hook(make_hook(activations)))

    # Run forward
    model.eval()
    with torch.no_grad():
        _ = model(data.x, data.edge_index, data.batch)

    for h in handles:
        h.remove()

    return _global_pool_states(activations, data.batch)


def capture_graph_level_states(
    model: nn.Module,
    data,
) -> List[np.ndarray]:
    """Capture per-layer graph embeddings for graph-level models.

    Instead of only hooking block-level outputs (which yields only 2-3
    graph embeddings regardless of depth), we manually step through every
    convolution layer inside every block and global-mean-pool node states
    after each layer.  This produces (N_blocks * (1+L)) graph embeddings,
    whose count scales with h_layer.

    Returns:
        List of graph embedding arrays [num_graphs, hidden_dim], one per
        internal convolution layer across all blocks.
    """
    from torch_geometric.nn import global_mean_pool

    graph_embs: List[torch.Tensor] = []
    x_raw = data.x
    edge_index = data.edge_index
    batch = data.batch

    model.eval()
    with torch.no_grad():
        # Determine whether this is GraphCrossGNN (paired cross-exchange)
        is_cross = hasattr(model, "blocks") and len(model.blocks) == 4

        if not is_cross:
            # ── GraphResGNN (3 blocks, sequential) ──
            graph_hidden = None
            for block in model.blocks:
                # ─ input_layer ─
                x = F.relu(block.input_layer(x_raw, edge_index))
                graph_embs.append(
                    global_mean_pool(x, batch).detach()
                )
                gate = block.graph_gate()
                # ─ hidden_layers ─
                for layer in block.hidden_layers:
                    if block.res_graph and graph_hidden is not None:
                        residual = apply_residual_mode(
                            graph_hidden[batch],
                            residual_mode=block.residual_mode,
                            topk_ratio=block.topk_ratio,
                            sparse_lambda=block.sparse_lambda,
                        )
                        x = x + gate * residual
                    x = F.relu(layer(x, edge_index))
                    x = F.dropout(x, p=block.dropout, training=False)
                    graph_embs.append(
                        global_mean_pool(x, batch).detach()
                    )
                # ─ end-of-block graph_embedding (for next block's graph_hidden) ─
                graph_embedding = global_mean_pool(x, batch)
                if graph_hidden is not None:
                    residual = apply_residual_mode(
                        graph_hidden,
                        residual_mode=block.residual_mode,
                        topk_ratio=block.topk_ratio,
                        sparse_lambda=block.sparse_lambda,
                    )
                    graph_embedding = graph_embedding + gate * residual
                graph_hidden = graph_embedding
        else:
            # ── GraphCrossGNN (4 blocks, paired cross-exchange) ──
            # Same internal logic but blocks swap graph_hidden in pairs
            g1: Optional[torch.Tensor] = None
            g2: Optional[torch.Tensor] = None
            blocks_list = list(model.blocks)
            for pair_idx in range(2):
                b1, b2 = blocks_list[2 * pair_idx], blocks_list[2 * pair_idx + 1]
                # ── pair: block b1 uses g1, block b2 uses g2 ──
                for block, g_in in [(b1, g1), (b2, g2)]:
                    x = F.relu(block.input_layer(x_raw, edge_index))
                    graph_embs.append(
                        global_mean_pool(x, batch).detach()
                    )
                    gate = block.graph_gate()
                    for layer in block.hidden_layers:
                        if block.res_graph and g_in is not None:
                            residual = apply_residual_mode(
                                g_in[batch],
                                residual_mode=block.residual_mode,
                                topk_ratio=block.topk_ratio,
                                sparse_lambda=block.sparse_lambda,
                            )
                            x = x + gate * residual
                        x = F.relu(layer(x, edge_index))
                        x = F.dropout(x, p=block.dropout, training=False)
                        graph_embs.append(
                            global_mean_pool(x, batch).detach()
                        )
                    g_out = global_mean_pool(x, batch)
                    if g_in is not None:
                        residual = apply_residual_mode(
                            g_in,
                            residual_mode=block.residual_mode,
                            topk_ratio=block.topk_ratio,
                            sparse_lambda=block.sparse_lambda,
                        )
                        g_out = g_out + gate * residual
                    if block is b1:
                        g1 = g_out
                    else:
                        g2 = g_out
                # ── cross-swap: g1 ↔ g2 between pairs ──
                g1, g2 = g2, g1

    return [g.cpu().numpy() for g in graph_embs]


def capture_states(
    model: nn.Module, data, model_name: str
) -> Tuple[List[np.ndarray], str]:
    """Dispatch to appropriate capture method based on model type."""
    if model_name in ("PlainGNN", "NodeResGNN"):
        states = capture_node_level_states(model, data)
        label = "layer"
    elif model_name == "NodeCrossGNN":
        states = capture_node_level_states(model, data)
        label = "layer_pair"
    elif model_name in ("GraphResGNN", "GraphCrossGNN"):
        states = capture_graph_level_states(model, data)
        label = "layer"  # now per-layer granularity, not just block-level
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return states, label


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices.

    X, Y: [n_samples, dim] — centered internally.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.sum((X @ X.T) * (Y @ Y.T))
    norm_x = np.sqrt(np.sum((X @ X.T) ** 2))
    norm_y = np.sqrt(np.sum((Y @ Y.T) ** 2))
    denom = norm_x * norm_y
    if denom < 1e-12:
        return 1.0
    return float(hsic / denom)


def cosine_similarity_matrix(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean pairwise cosine similarity between rows of X and Y."""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    sims = np.sum(X_norm * Y_norm, axis=1)
    return float(np.mean(sims))


def compute_layer_similarities(
    states: List[np.ndarray],
) -> Dict[str, List[float]]:
    """Compute CKA and cosine similarity between adjacent depth stages."""
    cka_vals = []
    cos_vals = []
    for i in range(len(states) - 1):
        cka_vals.append(linear_cka(states[i], states[i + 1]))
        cos_vals.append(cosine_similarity_matrix(states[i], states[i + 1]))
    return {"cka": cka_vals, "cosine": cos_vals}


def run_experiment_a(args, dataset, test_loader) -> Dict:
    """Run hidden-state similarity analysis for all 5 models."""
    results = {}
    data = next(iter(test_loader))
    data = data.to(DEVICE)

    for model_name in FIVE_MODELS:
        print(f"  Experiment A: capturing states for {model_name}...")
        model_args = argparse.Namespace(
            gname=model_name,
            name=args.operator,
            dim=args.dim,
            h_layer=args.h_layer,
            drop=args.drop,
            gate_init=0.8,
            gate_mode="learnable",
            fixed_gate_value=0.8,
            residual_mode="learnable",
            topk_ratio=0.5,
            sparse_lambda=0.05,
        )
        model = build_model(model_args, dataset).to(DEVICE)
        states, stage_label = capture_states(model, data, model_name)
        sims = compute_layer_similarities(states)
        results[model_name] = {
            "num_stages": len(states),
            "stage_label": stage_label,
            "cka": sims["cka"],
            "cosine": sims["cosine"],
            "state_shapes": [list(s.shape) for s in states],
        }
        print(f"    {len(states)} {stage_label}s, CKA: {[f'{v:.4f}' for v in sims['cka']]}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ──────────────────────────────────────────────
# Experiment B: Per-layer gradient norm
# ──────────────────────────────────────────────

def get_layer_groups(model: nn.Module, model_name: str) -> Dict[str, List[nn.Parameter]]:
    """Group model parameters by logical layer for gradient analysis.

    For node-level models (PlainGNN, NodeResGNN, NodeCrossGNN):
      Groups: input_proj, layer_{1..L} (or layer_pair_{1..P} for NodeCrossGNN),
      classifier.  The layer count is inferred from parameter names at runtime.

    For graph-level models (GraphResGNN, GraphCrossGNN):
      Each block internally has input_layer + hidden_layers.  We aggregate
      per block: block_{1..N}_total, and also report block_{n}_deepest for
      the deepest internal layer of each block.
    """
    groups = defaultdict(list)

    is_graph_level = model_name in ("GraphResGNN", "GraphCrossGNN")
    is_cross_node = model_name == "NodeCrossGNN"

    # Pre-scan to find the deepest hidden-layer index per block for graph-level models
    deepest_per_block: Dict[int, int] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if is_graph_level:
            if "classifier" in name and "blocks" not in name:
                groups["classifier"].append(param)
            elif "blocks." in name:
                parts = name.split("blocks.")
                block_idx = int(parts[1].split(".")[0])
                block_key = f"block_{block_idx + 1}"
                groups[block_key].append(param)
                if "hidden_layers." in name:
                    hl_idx = int(parts[1].split("hidden_layers.")[1].split(".")[0])
                    deepest_per_block[block_idx] = max(deepest_per_block.get(block_idx, -1), hl_idx)
            else:
                groups["other"].append(param)
        elif is_cross_node:
            if "input_layer_1" in name or "input_layer_2" in name:
                groups["input_proj"].append(param)
            elif "hidden_layers." in name:
                idx = int(name.split("hidden_layers.")[1].split(".")[0])
                pair_idx = idx // 2 + 1
                groups[f"pair_{pair_idx}"].append(param)
            elif "classifier" in name:
                groups["classifier"].append(param)
            else:
                groups["other"].append(param)
        else:
            # PlainGNN / NodeResGNN: dynamic layer count
            if "input_layer" in name:
                groups["input_proj"].append(param)
            elif "hidden_layers." in name:
                idx = int(name.split("hidden_layers.")[1].split(".")[0])
                groups[f"layer_{idx + 1}"].append(param)
            elif "classifier" in name:
                groups["classifier"].append(param)
            else:
                groups["other"].append(param)

    # Post-process graph-level: tag deepest internal layer per block
    for block_idx, deepest in deepest_per_block.items():
        # Re-scan parameters to assign deepest
        pass  # handled below

    result = dict(groups)
    # For graph-level, add deepest per block by re-scanning
    if is_graph_level and deepest_per_block:
        for block_idx, deepest_idx in deepest_per_block.items():
            block_key = f"block_{block_idx + 1}_deepest"
            deep_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if f"blocks.{block_idx}.hidden_layers.{deepest_idx}" in name:
                    deep_params.append(param)
            if deep_params:
                result[block_key] = deep_params

    return result


def compute_per_layer_grad_norms(groups: Dict[str, List[nn.Parameter]]) -> Dict[str, float]:
    """Compute L2 gradient norm for each parameter group."""
    norms = {}
    for group_name, params in groups.items():
        squared = 0.0
        count = 0
        for p in params:
            if p.grad is not None:
                squared += float((p.grad.detach() ** 2).sum())
                count += 1
        norms[group_name] = float(np.sqrt(squared)) if count > 0 else 0.0
    return norms


def run_experiment_b(args, full_dataset, train_dataset) -> Dict:
    """Run per-layer gradient norm analysis for all 5 models.

    Args:
        full_dataset: The original TUDataset (needed by build_model for dims).
        train_dataset: The split training list (for the data loader).
    """
    results = {}

    # Use a small subset for gradient analysis (one batch)
    temp_loader = build_loader(train_dataset, batch_size=args.batch_size, shuffle=False)
    batch_data = next(iter(temp_loader))
    batch_data = batch_data.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    for model_name in FIVE_MODELS:
        print(f"  Experiment B: gradient analysis for {model_name}...")
        model_args = argparse.Namespace(
            gname=model_name,
            name=args.operator,
            dim=args.dim,
            h_layer=args.h_layer,
            drop=args.drop,
            gate_init=0.8,
            gate_mode="learnable",
            fixed_gate_value=0.8,
            residual_mode="learnable",
            topk_ratio=0.5,
            sparse_lambda=0.05,
        )
        model = build_model(model_args, full_dataset).to(DEVICE)
        groups = get_layer_groups(model, model_name)

        # Run 50 training steps and record per-layer grad norms
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        grad_history: Dict[str, List[float]] = defaultdict(list)

        for step in range(50):
            model.train()
            optimizer.zero_grad()
            targets = batch_data.y
            logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            norms = compute_per_layer_grad_norms(groups)
            for k, v in norms.items():
                grad_history[k].append(v)

        # Summarize: mean and std of grad norm per layer across steps
        summary = {}
        for layer_name, values in grad_history.items():
            arr = np.array(values)
            summary[layer_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "final": float(arr[-1]),
                "history": [float(v) for v in values],
            }

        results[model_name] = summary
        print(f"    {len(summary)} layer groups recorded")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

MODEL_COLORS = {
    "PlainGNN": "#1f77b4",
    "NodeResGNN": "#ff7f0e",
    "NodeCrossGNN": "#2ca02c",
    "GraphResGNN": "#d62728",
    "GraphCrossGNN": "#9467bd",
}

MODEL_MARKERS = {
    "PlainGNN": "o",
    "NodeResGNN": "s",
    "NodeCrossGNN": "D",
    "GraphResGNN": "^",
    "GraphCrossGNN": "v",
}

MODEL_DISPLAY = {
    "PlainGNN": "PlainGNN (no reuse)",
    "NodeResGNN": "NodeResGNN (node residual)",
    "NodeCrossGNN": "NodeCrossGNN (node cross)",
    "GraphResGNN": "GraphResGNN (graph residual)",
    "GraphCrossGNN": "GraphCrossGNN (graph cross)",
}


def plot_cka_cosine(results_a: Dict, ds: str, op: str) -> None:
    """Plot CKA and cosine similarity vs depth for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_style import apply_paper_style
    apply_paper_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    all_cka_vals = []
    all_cos_vals = []
    for model_name in FIVE_MODELS:
        r = results_a[model_name]
        depths = list(range(1, len(r["cka"]) + 1))
        color = MODEL_COLORS[model_name]
        marker = MODEL_MARKERS[model_name]
        label = MODEL_DISPLAY[model_name]
        all_cka_vals.extend(r["cka"])
        all_cos_vals.extend(r["cosine"])

        ax1.plot(depths, r["cka"], color=color, marker=marker, markersize=9,
                 linewidth=2.5, label=label, markeredgewidth=1.5)
        ax2.plot(depths, r["cosine"], color=color, marker=marker, markersize=9,
                 linewidth=2.5, label=label, markeredgewidth=1.5)

    for ax, title, ylabel, all_vals in [
        (ax1, "CKA similarity between adjacent depth stages", "Linear CKA", all_cka_vals),
        (ax2, "Cosine similarity between adjacent depth stages", "Mean pairwise cosine", all_cos_vals),
    ]:
        ax.set_xlabel("Depth transition (stage i → i+1)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=12, loc="lower right", framealpha=0.9)
        # Adaptive y-axis
        y_margin = 0.06
        ax.set_ylim(max(0.0, min(all_vals) - y_margin), min(1.05, max(all_vals) + y_margin))
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Depth-induced representation similarity — {ds} ({op})",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = OUTPUT_DIR / f"fig_supp_hidden_similarity_{ds}_{op}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_gradient_norms(results_b: Dict, ds: str, op: str) -> None:
    """Plot per-layer gradient norms as bar chart and over steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_style import apply_paper_style
    apply_paper_style()

    # --- Bar chart: mean gradient norm per layer ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, model_name in enumerate(FIVE_MODELS):
        ax = axes[idx]
        r = results_b[model_name]
        # Filter out "other" group (usually near-zero)
        layer_names = [k for k in r.keys() if k != "other"]
        # Sort: input first, then layers/pairs/blocks, then classifier
        def sort_key(k):
            if "input" in k:
                return (0, k)
            if "classifier" in k:
                return (2, k)
            return (1, k)
        layer_names.sort(key=sort_key)
        means = [r[ln]["mean"] for ln in layer_names]
        stds = [r[ln]["std"] for ln in layer_names]

        bars = ax.bar(range(len(layer_names)), means, yerr=stds,
                      color=MODEL_COLORS[model_name], alpha=0.85,
                      edgecolor="black", linewidth=0.5, capsize=3)
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Gradient L2 norm", fontsize=13)
        ax.set_title(MODEL_DISPLAY[model_name], fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        # Adaptive y-axis
        all_bar_vals = [m + s for m, s in zip(means, stds)]
        if all_bar_vals:
            y_max_bar = max(all_bar_vals) * 1.15
            ax.set_ylim(0, y_max_bar)

    # Hide unused subplot (5 models in 2x3 = 6 slots)
    axes[5].set_visible(False)

    fig.suptitle(
        f"Per-layer gradient norm distribution — {ds} ({op})",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    path = OUTPUT_DIR / f"fig_supp_gradient_norms_{ds}_{op}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Line chart: gradient norm evolution over steps (node-level models, deepest layer) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    node_models = ["PlainGNN", "NodeResGNN", "NodeCrossGNN"]
    all_grad_vals = []
    for model_name in node_models:
        r = results_b[model_name]
        # Find deepest layer key
        if model_name == "NodeCrossGNN":
            deep_keys = [k for k in r.keys() if k.startswith("pair_")]
            deep_key = deep_keys[-1] if deep_keys else list(r.keys())[-1]
        else:
            deep_keys = [k for k in r.keys() if k.startswith("layer_")]
            deep_key = deep_keys[-1] if deep_keys else list(r.keys())[-1]
        history = r[deep_key]["history"]
        all_grad_vals.extend(history)
        ax.plot(range(len(history)), history,
                color=MODEL_COLORS[model_name], linewidth=2.5,
                label=f"{MODEL_DISPLAY[model_name]} — {deep_key}", alpha=0.85)

    ax.set_xlabel("Training step", fontsize=14)
    ax.set_ylabel("Gradient L2 norm", fontsize=14)
    ax.set_title(f"Deepest-layer gradient evolution — {ds} ({op})", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    # Adaptive y-axis
    if all_grad_vals:
        margin = (max(all_grad_vals) - min(all_grad_vals)) * 0.12
        ax.set_ylim(max(0, min(all_grad_vals) - margin), max(all_grad_vals) + margin)
    plt.tight_layout()
    path = OUTPUT_DIR / f"fig_supp_gradient_evolution_{ds}_{op}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Multi-layer depth comparison
# ──────────────────────────────────────────────

LAYER_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    6: "#8c564b",
    7: "#e377c2",
    8: "#7f7f7f",
}

LAYER_MARKERS = {
    1: "o",
    2: "s",
    3: "D",
    4: "^",
    5: "v",
    6: "<",
    7: ">",
    8: "P",
}


def plot_layer_depth_comparison(
    all_results: Dict[int, Dict], ds: str, op: str
) -> None:
    """Combined plot: CKA/cosine vs depth across different layer counts (4-8).

    For each model, we plot how CKA and cosine similarity decay with depth
    for each layer count.  The x-axis is the depth-transition index (normalized
    to the maximum depth), and we show one subplot per model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_style import apply_paper_style
    apply_paper_style()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for model_idx, model_name in enumerate(FIVE_MODELS):
        ax = axes[model_idx]
        for h_layer in sorted(all_results.keys()):
            if model_name not in all_results[h_layer]:
                continue
            r = all_results[h_layer][model_name]
            cka_vals = r.get("cka", [])
            depths = list(range(1, len(cka_vals) + 1))
            color = LAYER_COLORS[h_layer]
            marker = LAYER_MARKERS[h_layer]
            ax.plot(depths, cka_vals, color=color, marker=marker,
                    markersize=9, linewidth=2.2,
                    label=f"L={h_layer}", markeredgewidth=1.5)

        ax.set_xlabel("Depth transition (stage i → i+1)", fontsize=14)
        ax.set_ylabel("Linear CKA", fontsize=14)
        ax.set_title(MODEL_DISPLAY[model_name], fontsize=15, fontweight="bold")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        # Adaptive y-axis with small margin
        all_vals = []
        for h_layer in sorted(all_results.keys()):
            if model_name in all_results[h_layer]:
                all_vals.extend(all_results[h_layer][model_name].get("cka", []))
        if all_vals:
            margin = (max(all_vals) - min(all_vals)) * 0.12 + 0.03
            y_min = min(all_vals) - margin
            y_max = min(1.05, max(all_vals) + margin)
            ax.set_ylim(y_min, y_max)

    # Hide unused subplot
    axes[5].set_visible(False)

    fig.suptitle(f"CKA similarity vs depth across layer counts (L=1--8) — {ds} ({op})",
                 fontsize=17, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / f"fig_supp_layer_cka_comparison_{ds}_{op}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Cosine version ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for model_idx, model_name in enumerate(FIVE_MODELS):
        ax = axes[model_idx]
        for h_layer in sorted(all_results.keys()):
            if model_name not in all_results[h_layer]:
                continue
            r = all_results[h_layer][model_name]
            cos_vals = r.get("cosine", [])
            depths = list(range(1, len(cos_vals) + 1))
            color = LAYER_COLORS[h_layer]
            marker = LAYER_MARKERS[h_layer]
            ax.plot(depths, cos_vals, color=color, marker=marker,
                    markersize=9, linewidth=2.2,
                    label=f"L={h_layer}", markeredgewidth=1.5)

        ax.set_xlabel("Depth transition (stage i → i+1)", fontsize=14)
        ax.set_ylabel("Mean pairwise cosine", fontsize=14)
        ax.set_title(MODEL_DISPLAY[model_name], fontsize=15, fontweight="bold")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        all_vals = []
        for h_layer in sorted(all_results.keys()):
            if model_name in all_results[h_layer]:
                all_vals.extend(all_results[h_layer][model_name].get("cosine", []))
        if all_vals:
            margin = (max(all_vals) - min(all_vals)) * 0.12 + 0.03
            y_min = min(all_vals) - margin
            y_max = min(1.05, max(all_vals) + margin)
            ax.set_ylim(y_min, y_max)

    axes[5].set_visible(False)
    fig.suptitle(f"Cosine similarity vs depth across layer counts (L=1--8) — {ds} ({op})",
                 fontsize=17, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / f"fig_supp_layer_cosine_comparison_{ds}_{op}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_layer_gradient_comparison(
    all_grad_results: Dict[int, Dict], ds: str, op: str
) -> None:
    """Combined plot: deepest-layer gradient norm vs layer count for each model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_style import apply_paper_style
    apply_paper_style()

    fig, ax = plt.subplots(figsize=(12, 6))
    layers_sorted = sorted(all_grad_results.keys())

    for model_name in FIVE_MODELS:
        means = []
        for h_layer in layers_sorted:
            r = all_grad_results[h_layer].get(model_name, {})
            if not r:
                means.append(np.nan)
                continue
            # Get the deepest non-classifier, non-input group
            non_aux = [k for k in r.keys()
                       if k not in ("other", "classifier", "input_proj")]
            if not non_aux:
                means.append(np.nan)
                continue
            # Sort to get deepest
            def depth_key(k):
                nums = []
                for part in k.split("_"):
                    try:
                        nums.append(int(part))
                    except ValueError:
                        pass
                return nums if nums else [999]
            deepest_key = sorted(non_aux, key=depth_key)[-1]
            means.append(r[deepest_key]["mean"])

        ax.plot(layers_sorted, means,
                color=MODEL_COLORS[model_name],
                marker=MODEL_MARKERS[model_name],
                markersize=10, linewidth=2.5,
                label=MODEL_DISPLAY[model_name],
                markeredgewidth=1.5)

    ax.set_xlabel("Number of layers (L)", fontsize=14)
    ax.set_ylabel("Deepest-layer gradient L2 norm", fontsize=14)
    ax.set_title(f"Deepest-layer gradient norm vs depth (L=1--8) — {ds} ({op})",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers_sorted)
    # Adaptive y-axis
    all_vals = []
    for h_layer in layers_sorted:
        for model_name in FIVE_MODELS:
            r = all_grad_results[h_layer].get(model_name, {})
            non_aux = [k for k in r.keys()
                       if k not in ("other", "classifier", "input_proj")]
            if not non_aux:
                continue
            def depth_key2(k):
                nums = []
                for part in k.split("_"):
                    try:
                        nums.append(int(part))
                    except ValueError:
                        pass
                return nums if nums else [999]
            dk = sorted(non_aux, key=depth_key2)[-1]
            all_vals.append(r[dk]["mean"])
    if all_vals:
        valid = [v for v in all_vals if not np.isnan(v)]
        if valid:
            margin = (max(valid) - min(valid)) * 0.15
            ax.set_ylim(max(0, min(valid) - margin), max(valid) + margin)

    plt.tight_layout()
    path = OUTPUT_DIR / f"fig_supp_layer_gradient_comparison_{ds}_{op}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_multi_layer_comparison(args, dataset, train_dataset, test_loader):
    """Run experiments A and B for layers 4,5,6,7,8 and produce comparison plots."""
    all_a: Dict[int, Dict] = {}
    all_b: Dict[int, Dict] = {}

    for h_layer in [1, 2, 3, 4, 5, 6, 7, 8]:
        print("\n" + "=" * 60)
        print(f"Layer count: L = {h_layer}")
        print("=" * 60)

        layer_args = argparse.Namespace(
            gname="",  # set per model
            name=args.operator,
            dim=args.dim,
            h_layer=h_layer,
            drop=args.drop,
            gate_init=0.8,
            gate_mode="learnable",
            fixed_gate_value=0.8,
            residual_mode="learnable",
            topk_ratio=0.5,
            sparse_lambda=0.05,
        )

        # Experiment A
        print(f"  Experiment A: Hidden-state similarity (L={h_layer})...")
        data = next(iter(test_loader)).to(DEVICE)
        results_a = {}
        for model_name in FIVE_MODELS:
            print(f"    Capturing states for {model_name}...")
            m_args = argparse.Namespace(**vars(layer_args))
            m_args.gname = model_name
            model = build_model(m_args, dataset).to(DEVICE)
            states, stage_label = capture_states(model, data, model_name)
            sims = compute_layer_similarities(states)
            results_a[model_name] = {
                "num_stages": len(states),
                "stage_label": stage_label,
                "cka": sims["cka"],
                "cosine": sims["cosine"],
                "state_shapes": [list(s.shape) for s in states],
            }
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        all_a[h_layer] = results_a

        # Experiment B
        print(f"  Experiment B: Gradient analysis (L={h_layer})...")
        temp_loader = build_loader(train_dataset, batch_size=args.batch_size, shuffle=False)
        batch_data = next(iter(temp_loader)).to(DEVICE)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        results_b = {}
        for model_name in FIVE_MODELS:
            print(f"    Gradient analysis for {model_name}...")
            m_args = argparse.Namespace(**vars(layer_args))
            m_args.gname = model_name
            model = build_model(m_args, dataset).to(DEVICE)
            groups = get_layer_groups(model, model_name)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            grad_history: Dict[str, List[float]] = defaultdict(list)
            for step in range(50):
                model.train()
                optimizer.zero_grad()
                targets = batch_data.y
                logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                norms = compute_per_layer_grad_norms(groups)
                for k, v in norms.items():
                    grad_history[k].append(v)
            summary = {}
            for layer_name, values in grad_history.items():
                arr = np.array(values)
                summary[layer_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "final": float(arr[-1]),
                    "history": [float(v) for v in values],
                }
            results_b[model_name] = summary
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        all_b[h_layer] = results_b

    # Save combined JSON
    with open(ANALYSIS_DIR / f"layer_comparison_a_{args.ds}_{args.operator}.json", "w") as f:
        json.dump(all_a, f, indent=2)
    with open(ANALYSIS_DIR / f"layer_comparison_b_{args.ds}_{args.operator}.json", "w") as f:
        # Convert nested defaultdict to serializable dict
        all_b_serializable = {}
        for h_layer, models in all_b.items():
            all_b_serializable[str(h_layer)] = models
        json.dump(all_b_serializable, f, indent=2)

    # Produce comparison plots
    print("\n" + "=" * 60)
    print("Generating multi-layer comparison plots...")
    print("=" * 60)
    plot_layer_depth_comparison(all_a, args.ds, args.operator)
    plot_layer_gradient_comparison(all_b, args.ds, args.operator)

    return all_a, all_b


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run supplementary experiments A and B.")
    parser.add_argument("--ds", type=str, default="PROTEINS")
    parser.add_argument("--operator", type=str, default="GCNConv")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--h_layer", type=int, default=4)
    parser.add_argument("--drop", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--multi_layer", action="store_true",
                        help="Run multi-layer comparison (L=4,5,6,7,8) and produce combined plots.")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading dataset: {args.ds}")
    dataset = load_dataset(args.ds)
    train_dataset, test_dataset, _, _ = split_dataset(dataset, args.fold, args.ds)
    test_loader = build_loader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.multi_layer:
        # ── Multi-layer comparison mode ──
        run_multi_layer_comparison(args, dataset, train_dataset, test_loader)
        print("\nDone. Multi-layer comparison results saved to:")
        print(f"  JSON: {ANALYSIS_DIR}")
        print(f"  Figures: {OUTPUT_DIR}")
        return

    # ── Experiment A (single layer) ──
    print("\n" + "=" * 60)
    print("Experiment A: Hidden-state representation similarity")
    print("=" * 60)
    results_a = run_experiment_a(args, dataset, test_loader)

    with open(ANALYSIS_DIR / f"hidden_similarity_{args.ds}_{args.operator}.json", "w") as f:
        json.dump(results_a, f, indent=2)

    plot_cka_cosine(results_a, args.ds, args.operator)

    # ── Experiment B (single layer) ──
    print("\n" + "=" * 60)
    print("Experiment B: Per-layer gradient norm analysis")
    print("=" * 60)
    results_b = run_experiment_b(args, dataset, train_dataset)

    with open(ANALYSIS_DIR / f"gradient_norms_{args.ds}_{args.operator}.json", "w") as f:
        json.dump(results_b, f, indent=2)

    plot_gradient_norms(results_b, args.ds, args.operator)

    print("\nDone. Results saved to:")
    print(f"  JSON: {ANALYSIS_DIR}")
    print(f"  Figures: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
