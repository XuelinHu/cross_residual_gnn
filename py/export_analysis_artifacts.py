from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.graph_classify_v3 import (
    DEVICE,
    build_loader,
    build_model,
    dataset_statistics,
    evaluate,
    evaluate_ogb,
    graph_target,
    prepare_batch,
    set_seed,
    split_dataset,
    split_train_val_dataset,
    load_dataset,
)
from geomatric.experiment_catalog import dataset_family

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None


ANALYSIS_DIR = ROOT / "figures" / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper analysis artifacts for one V3 config.")
    parser.add_argument("--ds", required=True)
    parser.add_argument("--gname", required=True)
    parser.add_argument("--name", default="GCNConv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--ep", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--drop", type=float, default=0.3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--h_layer", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=15)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    return parser.parse_args()


def ensure_analysis_dir() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def pca_project(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components].T
    return centered @ basis


def fit_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, List[Dict[str, float]], object, object]:
    family = dataset_family(args.ds)
    set_seed(args.seed + args.fold if family == "ogb_graphprop" else args.seed)
    dataset = load_dataset(args.ds)
    train_dataset, test_dataset, official_val_dataset, _ = split_dataset(dataset, args.fold, args.ds)
    if official_val_dataset is None:
        train_dataset, val_dataset = split_train_val_dataset(train_dataset, args.val_ratio, seed=args.seed + args.fold)
    else:
        val_dataset = official_val_dataset
    train_loader = build_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_loader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None
    test_loader = build_loader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args, dataset).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    evaluator = None
    if family == "ogb_graphprop":
        from ogb.graphproppred import Evaluator
        evaluator = Evaluator(name=args.ds)

    history: List[Dict[str, float]] = []
    best_state_dict = model.state_dict()
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.ep):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_graphs = 0
        for batch_data in train_loader:
            batch_data = prepare_batch(batch_data)
            targets = graph_target(batch_data)
            optimizer.zero_grad()
            logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, targets)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            predictions = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == targets).sum())
            total_graphs += batch_size

        train_loss = total_loss / max(total_graphs, 1)
        train_acc = total_correct / max(total_graphs, 1)
        if val_loader is not None:
            val_metrics = evaluate_ogb(model, val_loader, criterion, evaluator) if evaluator is not None else evaluate(model, val_loader, criterion)
        else:
            val_metrics = {"loss": train_loss, "acc": train_acc}
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "lr": current_lr,
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    model.load_state_dict(best_state_dict)
    model.to(DEVICE)
    return model, history, dataset, test_loader


def collect_test_outputs(model: torch.nn.Module, test_loader) -> Dict[str, np.ndarray]:
    model.eval()
    embeddings: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = prepare_batch(batch_data)
            logits, graph_embedding = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            embeddings.append(graph_embedding.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())
            labels.append(graph_target(batch_data).detach().cpu().numpy())
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    return {
        "embeddings": np.concatenate(embeddings, axis=0),
        "logits": np.concatenate(logits_list, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "preds": np.concatenate(preds, axis=0),
    }


def save_learning_curves(history: List[Dict[str, float]], stem: str) -> None:
    if plt is None:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, [row["lr"] for row in history], color="black")
    axes[2].set_title("Learning Rate")

    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{stem}_curves.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_embedding_heatmap(embeddings: np.ndarray, labels: np.ndarray, stem: str) -> None:
    if plt is None or sns is None:
        return
    order = np.argsort(labels)
    sorted_embeddings = np.abs(embeddings[order])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(sorted_embeddings, cmap="magma", cbar=True, ax=ax)
    ax.set_title("Absolute Graph Embedding Heatmap")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Graph Index (sorted by label)")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{stem}_embedding_heatmap.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_pca_plot(embeddings: np.ndarray, labels: np.ndarray, stem: str) -> None:
    if plt is None:
        return
    projection = pca_project(embeddings, n_components=2)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", s=20, alpha=0.85)
    ax.set_title("Graph Embedding PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*scatter.legend_elements(), title="Class", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{stem}_embedding_pca.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int, stem: str) -> None:
    if plt is None or sns is None:
        return
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for label, pred in zip(labels, preds):
        matrix[int(label), int(pred)] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{stem}_confusion_matrix.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_analysis_dir()
    model, history, dataset, test_loader = fit_model(args)
    outputs = collect_test_outputs(model, test_loader)

    stem = f"{args.ds}_{args.gname}_{args.name}_fold{args.fold}"
    np.savez_compressed(
        ANALYSIS_DIR / f"{stem}_artifacts.npz",
        embeddings=outputs["embeddings"],
        logits=outputs["logits"],
        labels=outputs["labels"],
        preds=outputs["preds"],
    )
    save_learning_curves(history, stem)
    save_embedding_heatmap(outputs["embeddings"], outputs["labels"], stem)
    save_pca_plot(outputs["embeddings"], outputs["labels"], stem)
    save_confusion_matrix(outputs["labels"], outputs["preds"], dataset.num_classes, stem)

    artifact_paths = {
        "raw_npz": str(ANALYSIS_DIR / f"{stem}_artifacts.npz"),
        "curves": str(ANALYSIS_DIR / f"{stem}_curves.png"),
        "embedding_heatmap": str(ANALYSIS_DIR / f"{stem}_embedding_heatmap.png"),
        "embedding_pca": str(ANALYSIS_DIR / f"{stem}_embedding_pca.png"),
        "confusion_matrix": str(ANALYSIS_DIR / f"{stem}_confusion_matrix.png"),
    }
    summary = {
        "config": vars(args),
        "dataset_stats": dataset_statistics(dataset, args.ds),
        "num_test_graphs": int(outputs["labels"].shape[0]),
        "embedding_dim": int(outputs["embeddings"].shape[1]),
        "plotting_available": bool(plt is not None and sns is not None),
        "artifacts": artifact_paths,
    }
    (ANALYSIS_DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
