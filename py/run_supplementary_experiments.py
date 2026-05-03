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
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.graph_classify_v3 import (
    DEVICE,
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
    """Capture intermediate graph embeddings for graph-level models.

    GraphResGNN and GraphCrossGNN internally produce graph_embedding
    at each block. We hook block-level outputs.

    Returns:
        List of graph embedding arrays, one per block stage.
    """
    from torch_geometric.nn import global_mean_pool

    graph_embs: List[torch.Tensor] = []

    def make_graph_hook(store: List):
        def hook(module, inp, out):
            # out is (logits, graph_embedding) from PlainBlock
            if isinstance(out, tuple) and len(out) == 2:
                store.append(out[1].detach())
        return hook

    handles = []
    if hasattr(model, "blocks"):
        for block in model.blocks:
            handles.append(block.register_forward_hook(make_graph_hook(graph_embs)))

    model.eval()
    with torch.no_grad():
        _ = model(data.x, data.edge_index, data.batch)

    for h in handles:
        h.remove()

    # Each graph_emb is already [num_graphs, hidden_dim]
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
        label = "block"
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
      Groups: input_proj, layer_{1..4} (or layer_pair_{1..4} for NodeCrossGNN),
      classifier.

    For graph-level models (GraphResGNN, GraphCrossGNN):
      Each block internally has input_layer + hidden_layers.  We aggregate
      per block: block_{1..N}_total, and also report block_{n}_deepest for
      the deepest internal layer of each block.
    """
    groups = defaultdict(list)

    is_graph_level = model_name in ("GraphResGNN", "GraphCrossGNN")
    is_cross_node = model_name == "NodeCrossGNN"

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if is_graph_level:
            # GraphResGNN (3 blocks), GraphCrossGNN (4 blocks)
            # params named: blocks.0.input_layer.lin.weight, blocks.0.hidden_layers.0.lin.weight, ...
            if "classifier" in name and "blocks" not in name:
                groups["classifier"].append(param)
            elif "blocks." in name:
                # Extract block index
                parts = name.split("blocks.")
                block_idx = int(parts[1].split(".")[0])
                block_key = f"block_{block_idx + 1}"
                groups[block_key].append(param)
                # Also tag deepest internal layer per block
                if f"hidden_layers.{3}" in name or f"hidden_layers.{4}" in name:
                    # h_layer=4, so hidden_layers.3 is the deepest
                    if f"hidden_layers.3" in name:
                        groups[f"block_{block_idx + 1}_deepest"].append(param)
                elif "hidden_layers" not in name and "input_layer" not in name:
                    pass  # classifier inside block
            else:
                groups["other"].append(param)
        elif is_cross_node:
            # NodeCrossGNN: input_layer_1, input_layer_2, hidden_layers (8 total, used in pairs)
            if "input_layer_1" in name:
                groups["input_proj"].append(param)
            elif "input_layer_2" in name:
                groups["input_proj"].append(param)
            elif "hidden_layers." in name:
                idx = int(name.split("hidden_layers.")[1].split(".")[0])
                pair_idx = idx // 2 + 1  # pairs: layers 0-1→pair1, 2-3→pair2, ...
                groups[f"pair_{pair_idx}"].append(param)
            elif "classifier" in name:
                groups["classifier"].append(param)
            else:
                groups["other"].append(param)
        else:
            # PlainGNN / NodeResGNN: input_layer, hidden_layers[0..3], classifier
            if "input_layer" in name:
                groups["input_proj"].append(param)
            elif "hidden_layers.0" in name:
                groups["layer_1"].append(param)
            elif "hidden_layers.1" in name:
                groups["layer_2"].append(param)
            elif "hidden_layers.2" in name:
                groups["layer_3"].append(param)
            elif "hidden_layers.3" in name:
                groups["layer_4"].append(param)
            elif "classifier" in name:
                groups["classifier"].append(param)
            else:
                groups["other"].append(param)

    return dict(groups)


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

    for model_name in FIVE_MODELS:
        r = results_a[model_name]
        depths = list(range(1, len(r["cka"]) + 1))
        color = MODEL_COLORS[model_name]
        marker = MODEL_MARKERS[model_name]
        label = MODEL_DISPLAY[model_name]

        ax1.plot(depths, r["cka"], color=color, marker=marker, markersize=8,
                 linewidth=2, label=label, markeredgewidth=1.5)
        ax2.plot(depths, r["cosine"], color=color, marker=marker, markersize=8,
                 linewidth=2, label=label, markeredgewidth=1.5)

    for ax, title, ylabel in [
        (ax1, "CKA similarity between adjacent depth stages", "Linear CKA"),
        (ax2, "Cosine similarity between adjacent depth stages", "Mean pairwise cosine"),
    ]:
        ax.set_xlabel("Depth transition (stage i → i+1)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Depth-induced representation similarity — {ds} ({op})",
        fontsize=14, fontweight="bold", y=1.01,
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
        ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Gradient L2 norm", fontsize=10)
        ax.set_title(MODEL_DISPLAY[model_name], fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused subplot (5 models in 2x3 = 6 slots)
    axes[5].set_visible(False)

    fig.suptitle(
        f"Per-layer gradient norm distribution — {ds} ({op})",
        fontsize=14, fontweight="bold",
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
        ax.plot(range(len(history)), history,
                color=MODEL_COLORS[model_name], linewidth=2,
                label=f"{MODEL_DISPLAY[model_name]} — {deep_key}", alpha=0.85)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Gradient L2 norm", fontsize=12)
    ax.set_title(f"Deepest-layer gradient evolution — {ds} ({op})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / f"fig_supp_gradient_evolution_{ds}_{op}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


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
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading dataset: {args.ds}")
    dataset = load_dataset(args.ds)
    train_dataset, test_dataset, _, _ = split_dataset(dataset, args.fold, args.ds)
    test_loader = build_loader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Experiment A ──
    print("\n" + "=" * 60)
    print("Experiment A: Hidden-state representation similarity")
    print("=" * 60)
    results_a = run_experiment_a(args, dataset, test_loader)

    with open(ANALYSIS_DIR / f"hidden_similarity_{args.ds}_{args.operator}.json", "w") as f:
        json.dump(results_a, f, indent=2)

    plot_cka_cosine(results_a, args.ds, args.operator)

    # ── Experiment B ──
    print("\n" + "=" * 60)
    print("Experiment B: Per-layer gradient norm analysis")
    print("=" * 60)
    # Use train set for gradient analysis (one batch only)
    results_b = run_experiment_b(args, dataset, train_dataset)

    # Convert defaultdict/list history to serializable format
    with open(ANALYSIS_DIR / f"gradient_norms_{args.ds}_{args.operator}.json", "w") as f:
        json.dump(results_b, f, indent=2)

    plot_gradient_norms(results_b, args.ds, args.operator)

    print("\nDone. Results saved to:")
    print(f"  JSON: {ANALYSIS_DIR}")
    print(f"  Figures: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
