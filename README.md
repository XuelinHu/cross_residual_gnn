# Cross-Residual Graph Neural Networks

<p align="center">
  <img height="20" src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img height="20" src="https://img.shields.io/badge/PyTorch_Geometric-2.x-blue" />
  <img height="20" src="https://img.shields.io/badge/Python-3.10%2B-green" />
  <img height="20" src="https://img.shields.io/badge/License-GPL_v3.0-purple" />
</p>

Research code for Cross-Residual Graph Neural Networks (CR-GNN), a controlled graph-classification architecture family that compares plain stacking, residual reuse, and cross-residual information exchange under a unified training protocol.

The current repository is centered on the stabilized `v3` graph-classification pipeline in [geomatric/graph_classify_v3.py](/ds1/workspace/ai/cross_residual_gnn/geomatric/graph_classify_v3.py). The paper narrative is framed around protein-oriented graph benchmarks, while the final implementation study also includes a broader full-suite robustness evaluation.

## Overview

The repository now supports five main architectures under a shared `GCNConv` backbone:

- `PlainGNN`
- `NodeResGNN`
- `NodeCrossGNN`
- `GraphResGNN`
- `GraphCrossGNN`

The current codebase also includes:

- stratified 5-fold evaluation
- inner stratified validation split
- `ReduceLROnPlateau`
- gradient clipping
- TensorBoard logging
- JSON result export
- scripts for batch execution, summarization, and report generation

## Repository Layout

```text
cross_residual_gnn/
├── geomatric/
│   ├── graph_classify_v3.py
│   ├── graph_classify_v2.py
│   ├── node_classify.py
│   └── achivement/
├── py/
│   ├── run_paper_experiments.py
│   ├── summarize_paper_experiments.py
│   ├── generate_all_result_reports.py
│   ├── generate_topic_tables.py
│   ├── generate_exp_figures.py
│   └── export_analysis_artifacts.py
├── md/
│   ├── all_results_summary.txt
│   ├── all_exp_tables.tex
│   ├── all_ablation_analysis.md
│   ├── formal_experiment_protocol.md
│   └── paper_gap_checklist.md
├── figures/
│   └── exp/
├── paper/
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   ├── figures/
│   └── sections/
└── README.md
```

## Environment

Recommended environment:

- Python `3.10+`
- PyTorch with CUDA
- PyTorch Geometric
- `tensorboard`
- `matplotlib`, `seaborn`, `pandas`, `openpyxl`

Install the main dependencies with:

```bash
pip install -r requirements.txt
pip install torch-geometric tensorboard
```

If PyTorch Geometric wheels need to be installed manually, follow the official PyG installation instructions for your local PyTorch and CUDA version.

## Main Entry Points

### 1. Single experiment

```bash
python geomatric/graph_classify_v3.py \
  --mode single \
  --ds PROTEINS \
  --gname NodeCrossGNN \
  --name GCNConv \
  --ep 240 \
  --patience 80 \
  --lr 0.003 \
  --weight_decay 5e-5 \
  --drop 0.2 \
  --dim 64 \
  --h_layer 4 \
  --batch_size 32 \
  --grad_clip 2.0 \
  --tensorboard
```

### 2. Full paper experiment batch

```bash
python py/run_paper_experiments.py --dataset_group all --max_workers 6 --tensorboard
```

Supported dataset groups:

- `main`
- `topic`
- `extended`
- `all`

### 3. Summarize latest experiment logs

```bash
python py/summarize_paper_experiments.py --dataset_group all
```

### 4. Generate full-suite report files

```bash
python py/generate_all_result_reports.py
```

This writes:

- [md/all_results_summary.txt](/ds1/workspace/ai/cross_residual_gnn/md/all_results_summary.txt)
- [md/all_exp_tables.tex](/ds1/workspace/ai/cross_residual_gnn/md/all_exp_tables.tex)
- [md/all_ablation_analysis.md](/ds1/workspace/ai/cross_residual_gnn/md/all_ablation_analysis.md)

### 5. TensorBoard

```bash
tensorboard --logdir runs --port 6006
```

When `--tensorboard` is enabled, the training loop logs:

- train / validation loss
- train / validation accuracy
- learning rate
- gradient norm
- embedding statistics
- logit statistics

## Datasets

The currently prepared local dataset suite includes:

- `MUTAG`
- `PROTEINS`
- `DD`
- `ENZYMES`
- `MSRC_9`
- `AIDS`
- `Mutagenicity`

The paper uses:

- `PROTEINS`, `DD`, `ENZYMES` as the protein-oriented main benchmark
- `MUTAG`, `AIDS`, `Mutagenicity`, `MSRC_9` as broader structural validation

## Current Paper-Level Conclusions

Under the stabilized full-suite protocol:

- residual variants win `4/7` datasets
- cross-residual variants win `3/7` datasets
- the best cross model beats the plain baseline on `6/7` datasets

So the current evidence supports a dataset-dependent advantage for cross-residual design, not a universal claim that cross always beats residual reuse.

## Final Full-Suite Winners

- `MUTAG`: `NodeCrossGNN`
- `PROTEINS`: `GraphResGNN`
- `DD`: `NodeResGNN`
- `ENZYMES`: `GraphResGNN`
- `MSRC_9`: `GraphCrossGNN`
- `AIDS`: `GraphResGNN`
- `Mutagenicity`: `NodeCrossGNN`

## Paper Assets

The LaTeX manuscript lives in [paper](/ds1/workspace/ai/cross_residual_gnn/paper). The most relevant generated assets are:

- [paper/main.pdf](/ds1/workspace/ai/cross_residual_gnn/paper/main.pdf)
- [md/all_exp_tables.tex](/ds1/workspace/ai/cross_residual_gnn/md/all_exp_tables.tex)
- [md/all_ablation_analysis.md](/ds1/workspace/ai/cross_residual_gnn/md/all_ablation_analysis.md)
- [figures/exp](/ds1/workspace/ai/cross_residual_gnn/figures/exp)

Compile the paper with:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

## Notes

- Large runtime outputs such as `logs/`, `runs/`, `data/`, and intermediate `records/` are intentionally ignored by Git.
- The codebase still does not implement a true integrative-omics benchmark. The current scope is biomolecular graph representation learning with a protein-oriented main evaluation.
- A final method figure and author metadata are still needed before submission.
