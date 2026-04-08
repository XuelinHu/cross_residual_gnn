# Cross-Residual Graph Neural Networks

<p align="center">
  <img height="20" src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img height="20" src="https://img.shields.io/badge/PyTorch_Geometric-2.x-blue" />
  <img height="20" src="https://img.shields.io/badge/Python-3.10%2B-green" />
  <img height="20" src="https://img.shields.io/badge/License-GPL_v3.0-purple" />
</p>

Research code for Cross-Residual Graph Neural Networks (CR-GNN), a controlled graph-classification architecture family that compares plain stacking, residual reuse, and cross-residual information exchange under a unified training protocol.

The current repository is centered on the stabilized `v3` graph-classification pipeline in [geomatric/graph_classify_v3.py](/ds1/workspace/ai/cross_residual_gnn/geomatric/graph_classify_v3.py). The active Python package is now the [`geomatric`](/ds1/workspace/ai/cross_residual_gnn/geomatric) directory, which also contains the shared dataset catalog, logger, and notification helpers used by the current workflow. The paper narrative is framed around a protein-oriented biological package, while the supplementary experiments retain a broader robustness evaluation. The current scope is still biomolecular graph learning rather than a completed integrative-omics system, but the revised manuscript also strengthens the logical bridge toward plant-related graph inference.

- https://www.frontiersin.org/research-topics/73895/prediction-of-novel-domains-motifs-genes-and-proteins-through-integrative-omics-approaches

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

Older V2-era assets have been moved into [achivement_V2](/ds1/workspace/ai/cross_residual_gnn/achivement_V2) so that the repository root reflects the active `v3` workflow.

## Repository Layout

```text
cross_residual_gnn/
├── achivement_V2/               # Archived V2-era code and notes
├── geomatric/                   # Active package for training and shared helpers
│   ├── __init__.py
│   ├── experiment_catalog.py    # Dataset groups and metadata
│   ├── graph_classify_v3.py     # Main V3 training entry
│   ├── logging_config.py        # Shared logger setup
│   └── dingtalk_util.py         # DingTalk notification helper
├── py/                          # Batch runners and paper/report scripts
│   ├── run_paper_experiments.py
│   ├── summarize_paper_experiments.py
│   ├── generate_all_result_reports.py
│   ├── generate_suite_analysis_figures.py
│   ├── generate_dataset_statistics_report.py
│   ├── run_sensitivity_experiments.py
│   ├── generate_sensitivity_reports.py
│   ├── run_enzymes_tuned_cross.py
│   ├── generate_exp_figures.py
│   └── export_analysis_artifacts.py
├── md/
├── figures/
├── paper/
├── tmp/                         # Local scratch outputs, ignored by Git
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
python -m geomatric.graph_classify_v3 \
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

Direct path execution also works:

```bash
python geomatric/graph_classify_v3.py --mode single --ds PROTEINS
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
python py/generate_suite_analysis_figures.py
```

This writes:

- [md/all_results_summary.txt](/ds1/workspace/ai/cross_residual_gnn/md/all_results_summary.txt)
- [md/all_exp_tables.tex](/ds1/workspace/ai/cross_residual_gnn/md/all_exp_tables.tex)
- [md/all_ablation_analysis.md](/ds1/workspace/ai/cross_residual_gnn/md/all_ablation_analysis.md)
- figures under [paper/figures/exp](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp)

### 5. Run parameter sensitivity scans

```bash
python py/run_sensitivity_experiments.py --fold 0 --max_workers 6
python py/generate_sensitivity_reports.py
```

This writes:

- [md/sensitivity_summary.md](/ds1/workspace/ai/cross_residual_gnn/md/sensitivity_summary.md)
- [md/parameter_sensitivity_analysis.md](/ds1/workspace/ai/cross_residual_gnn/md/parameter_sensitivity_analysis.md)
- sensitivity figures under [paper/figures/exp](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp)

### 6. TensorBoard

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

Main biological package:

- `PROTEINS`
- `DD`
- `ENZYMES`

Supplementary robustness package:

- `MUTAG`
- `AIDS`
- `Mutagenicity`

The revised paper uses:

- `PROTEINS`, `DD`, `ENZYMES` as the main biological benchmark package
- `MUTAG`, `AIDS`, `Mutagenicity` as supplementary structural validation

Generate unified dataset statistics and paper tables with:

```bash
python py/generate_dataset_statistics_report.py
```

This writes:

- [md/dataset_statistics_summary.md](/ds1/workspace/ai/cross_residual_gnn/md/dataset_statistics_summary.md)
- [md/dataset_statistics_tables.tex](/ds1/workspace/ai/cross_residual_gnn/md/dataset_statistics_tables.tex)

## Current Paper-Level Conclusions

Under the current archive:

- residual variants are the strongest default family on the completed topic-facing TU datasets
- cross-residual variants remain selectively strong on selected supplementary datasets

So the current evidence supports a dataset-dependent advantage for cross-residual design, not a universal claim that cross always beats residual reuse.

## Final Full-Suite Winners

- `MUTAG`: `NodeCrossGNN`
- `PROTEINS`: `GraphResGNN`
- `DD`: `NodeResGNN`
- `ENZYMES`: `GraphResGNN`
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

- Large runtime outputs such as `logs/`, `runs/`, `data/`, intermediate `records/`, and local scratch files under `tmp/` are intentionally ignored by Git.
- The `py/` analysis scripts now import shared dataset definitions from `geomatric.experiment_catalog`, so the active package path should be kept intact when moving files.
- The codebase still does not implement a true integrative-omics benchmark. The current scope is biomolecular graph representation learning with a protein-oriented main evaluation and a clearer extension path toward plant-related graph inference.
- The remaining paper-side manual work is mainly author metadata, final reference audit, and venue-specific submission materials.
