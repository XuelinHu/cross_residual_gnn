# Cross-Residual Graph Neural Networks

[中文说明 / Chinese README](./README-CN.md)

<p align="center">
  <img height="20" src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img height="20" src="https://img.shields.io/badge/PyTorch_Geometric-2.x-blue" />
  <img height="20" src="https://img.shields.io/badge/Python-3.10%2B-green" />
  <img height="20" src="https://img.shields.io/badge/License-GPL_v3.0-purple" />
</p>

Research code for Cross-Residual Graph Neural Networks (CR-GNN), a controlled graph-classification architecture family that compares plain stacking, residual reuse, and cross-residual information exchange under a unified training protocol.

The active training pipeline is centered on [geomatric/graph_classify_v3.py](./geomatric/graph_classify_v3.py). Shared dataset definitions, logging helpers, and DingTalk notification utilities now live under the [geomatric/](./geomatric) package.

## Overview

The repository currently supports five main architectures under a shared `GCNConv` backbone:

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
- scripts for batch execution, summarization, plotting, and paper/report generation

Older V2-era assets have been moved into [achivement_V2/](./achivement_V2).

## Current Backfill Status

As of `2026-04-15 10:33:31`, the missing-experiment backfill job has reached:

- total target: `720`
- completed: `677`
- remaining: `43`
- baseline coverage: `120 / 120`
- non-`GCNConv` operator coverage: `557 / 600`

Live status is maintained in:

- [md/missing_experiment_completion.md](./md/missing_experiment_completion.md)
- [records/missing_experiment_status.json](./records/missing_experiment_status.json)
- [logs/missing_experiments_20260414_235657.log](./logs/missing_experiments_20260414_235657.log)

## Repository Layout

```text
cross_residual_gnn/
├── achivement_V2/               # Archived V2-era code and notes
├── data/                        # Local datasets (ignored by Git)
├── figures/                     # Exported experiment figures
├── geomatric/                   # Active Python package
├── logs/                        # JSON experiment snapshots (ignored by Git)
│   └── missing_jobs/            # Per-job stdout/stderr for backfill runs
├── md/                          # Generated markdown / TeX summaries
├── paper/                       # LaTeX manuscript and paper figures
├── py/                          # Batch runners and analysis scripts
├── records/                     # Text summaries and records (ignored by Git)
├── runs/                        # TensorBoard runs (ignored by Git)
├── tmp/                         # Local scratch outputs (ignored by Git)
├── README.md
└── README-CN.md
```

## Environment

Recommended environment:

- Python `3.10+`
- PyTorch
- PyTorch Geometric
- `tensorboard`
- `matplotlib`, `seaborn`, `pandas`, `openpyxl`

Install the main dependencies with:

```bash
pip install -r requirements.txt
pip install torch-geometric tensorboard
```

If PyTorch Geometric wheels need manual installation, follow the official PyG instructions for your local PyTorch and CUDA version.

## Path Index

### Core package paths

- [geomatric/__init__.py](./geomatric/__init__.py): package entry
- [geomatric/graph_classify_v3.py](./geomatric/graph_classify_v3.py): main training / evaluation entry
- [geomatric/experiment_catalog.py](./geomatric/experiment_catalog.py): dataset groups and metadata
- [geomatric/logging_config.py](./geomatric/logging_config.py): shared logger
- [geomatric/dingtalk_util.py](./geomatric/dingtalk_util.py): DingTalk markdown / actionCard notifications

### Run script paths

- [py/run_paper_experiments.py](./py/run_paper_experiments.py): full benchmark batch runner
- [py/run_missing_experiments.py](./py/run_missing_experiments.py): background runner for missing baseline / operator jobs
- [py/run_sensitivity_experiments.py](./py/run_sensitivity_experiments.py): sensitivity sweep runner
- [py/run_enzymes_tuned_cross.py](./py/run_enzymes_tuned_cross.py): tuned ENZYMES cross-model runs
- [py/summarize_paper_experiments.py](./py/summarize_paper_experiments.py): summarize latest logs
- [py/generate_all_result_reports.py](./py/generate_all_result_reports.py): generate full report text / TeX
- [py/generate_suite_analysis_figures.py](./py/generate_suite_analysis_figures.py): generate benchmark figures
- [py/generate_dataset_statistics_report.py](./py/generate_dataset_statistics_report.py): export dataset statistics
- [py/generate_sensitivity_reports.py](./py/generate_sensitivity_reports.py): summarize sensitivity logs and figures
- [py/generate_exp_figures.py](./py/generate_exp_figures.py): generate record-based experiment figures
- [py/generate_method_figure.py](./py/generate_method_figure.py): method figure helper
- [py/generate_topic_tables.py](./py/generate_topic_tables.py): topic-facing TeX tables
- [py/export_analysis_artifacts.py](./py/export_analysis_artifacts.py): per-run analysis artifacts
- [py/organize_references_from_corpus.py](./py/organize_references_from_corpus.py): reference organization helper
- [py/plot_style.py](./py/plot_style.py): plotting style utilities

### Experiment snapshot and output paths

These paths are important because they are the main "snapshots" of runtime results:

- `logs/LATEST/train_<dataset>_<model>_<operator>_fold<k>__<timestamp>.json`
  Current latest log root: [logs/LATEST](./logs/LATEST)
- `logs/LATEST/dataset_stats_<dataset>__<timestamp>.json`
  Dataset-stat log root: [logs/LATEST](./logs/LATEST)
- `records/LATEST/suite_<suite_name>_<dataset>__<timestamp>.txt`
  Current latest record root: [records/LATEST](./records/LATEST)
- `records/LATEST/missing_experiment_status.json`
  Live backfill status snapshot for the latest version: [records/LATEST](./records/LATEST)
- `runs/LATEST/<model>_<operator>_<dataset>_<dim>_fold<k>_<h_layer>_<timestamp>/events.out.tfevents.*`
  TensorBoard root for the latest version: [runs/LATEST](./runs/LATEST)
- `logs/LATEST/missing_jobs/<job_slug>.log`
  Per-job training log root: [logs/LATEST/missing_jobs](./logs/LATEST/missing_jobs)
- `records/experiment_versions.json`
  Unified version index for archived `V1` / `V2` / `V3` and current `LATEST`: [records/experiment_versions.json](./records/experiment_versions.json)
- `figures/analysis/*`
  Analysis artifact root: [figures/analysis/](./figures/analysis)
- `figures/exp/*`
  Exported benchmark figures: [figures/exp/](./figures/exp)
- `paper/figures/exp/*`
  Paper-ready figure copies: [paper/figures/exp/](./paper/figures/exp)

### Generated summary file paths

- [md/all_results_summary.txt](./md/all_results_summary.txt)
- [md/all_exp_tables.tex](./md/all_exp_tables.tex)
- [md/all_exp_tables_appendix.tex](./md/all_exp_tables_appendix.tex)
- [md/all_ablation_analysis.md](./md/all_ablation_analysis.md)
- [md/sensitivity_summary.md](./md/sensitivity_summary.md)
- [md/parameter_sensitivity_analysis.md](./md/parameter_sensitivity_analysis.md)
- [md/dataset_statistics_summary.md](./md/dataset_statistics_summary.md)
- [md/dataset_statistics_summary.json](./md/dataset_statistics_summary.json)
- [md/dataset_statistics_tables.tex](./md/dataset_statistics_tables.tex)
- [md/topic_results_summary.txt](./md/topic_results_summary.txt)
- [md/topic_exp_tables.tex](./md/topic_exp_tables.tex)
- [md/cross_advantage_summary.md](./md/cross_advantage_summary.md)
- [md/formal_experiment_protocol.md](./md/formal_experiment_protocol.md)
- [md/final_implementation_todo.md](./md/final_implementation_todo.md)
- [md/missing_experiment_completion.md](./md/missing_experiment_completion.md)
- [md/paper_gap_checklist.md](./md/paper_gap_checklist.md)
- [md/reference_logic_map.md](./md/reference_logic_map.md)
- [md/frontiers_topic_alignment.md](./md/frontiers_topic_alignment.md)
- [md/topic_aligned_dataset_shortlist.md](./md/topic_aligned_dataset_shortlist.md)
- [md/SESSION_RECORD.md](./md/SESSION_RECORD.md)

### Figure file paths

Current exported experiment figures:

- [figures/exp/fig1_full_suite_results.png](./figures/exp/fig1_full_suite_results.png)
- [figures/exp/fig1_full_suite_results.pdf](./figures/exp/fig1_full_suite_results.pdf)
- [figures/exp/fig2_cross_advantage_heatmap.png](./figures/exp/fig2_cross_advantage_heatmap.png)
- [figures/exp/fig2_cross_advantage_heatmap.pdf](./figures/exp/fig2_cross_advantage_heatmap.pdf)
- [figures/exp/fig3_rank_winner_summary.png](./figures/exp/fig3_rank_winner_summary.png)
- [figures/exp/fig3_rank_winner_summary.pdf](./figures/exp/fig3_rank_winner_summary.pdf)
- [figures/exp/fig4_topic_focus_results.png](./figures/exp/fig4_topic_focus_results.png)
- [figures/exp/fig4_topic_focus_results.pdf](./figures/exp/fig4_topic_focus_results.pdf)
- [figures/exp/fig5_protein_package_summary.png](./figures/exp/fig5_protein_package_summary.png)
- [figures/exp/fig5_protein_package_summary.pdf](./figures/exp/fig5_protein_package_summary.pdf)

Paper figure roots:

- [paper/figures/](./paper/figures)
- [paper/figures/exp/](./paper/figures/exp)
- [paper/figures/cr_gnn_schematic.png](./paper/figures/cr_gnn_schematic.png)
- [paper/figures/task_model_comparison.png](./paper/figures/task_model_comparison.png)

### Dataset paths

The repository does not commit raw datasets. Local dataset roots are:

- [data/](./data): local data root
- `data/TUDataset/`: TU datasets downloaded by `torch_geometric.datasets.TUDataset`
- `data/OGB/`: OGB graph-property datasets downloaded by `ogb.graphproppred.PygGraphPropPredDataset`
- [data/.gitkeep](./data/.gitkeep): placeholder only

Datasets currently grouped in [geomatric/experiment_catalog.py](./geomatric/experiment_catalog.py):

- Main package: `PROTEINS`, `DD`, `ENZYMES`
- Supplementary package: `MUTAG`, `AIDS`, `Mutagenicity`

### Paper file paths

- [paper/main.tex](./paper/main.tex)
- [paper/references.bib](./paper/references.bib)
- [paper/compile.bat](./paper/compile.bat)
- [paper/README.md](./paper/README.md)
- [paper/paper_corpus_merged.json](./paper/paper_corpus_merged.json)
- [paper/sections/01_introduction_peerj.tex](./paper/sections/01_introduction_peerj.tex)
- [paper/sections/02_methods_peerj.tex](./paper/sections/02_methods_peerj.tex)
- [paper/sections/03_results_peerj.tex](./paper/sections/03_results_peerj.tex)
- [paper/sections/04_discussion_peerj.tex](./paper/sections/04_discussion_peerj.tex)
- [paper/sections/05_conclusions_peerj.tex](./paper/sections/05_conclusions_peerj.tex)
- [paper/sections/06_acknowledgments.tex](./paper/sections/06_acknowledgments.tex)
- [paper/sections/07_appendix.tex](./paper/sections/07_appendix.tex)

## Main Commands

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

Direct script path also works:

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

### 4. Backfill missing baseline / operator experiments

```bash
python py/run_missing_experiments.py --report_only
python py/run_missing_experiments.py --max_parallel 8 --reserve_gb 4 --no_tensorboard
```

The current rerun scope excludes `TransformerConv` from the operator backfill set to keep runtime manageable.

This runner is designed for the current `24GB` GPU setup:

- reserves about `4GB` GPU memory
- uses the remaining budget for concurrent jobs
- increases batch size when memory is available
- retries failed jobs with a smaller batch size on OOM

### 5. Generate full-suite report files

```bash
python py/generate_all_result_reports.py
python py/generate_suite_analysis_figures.py
```

### 6. Run parameter sensitivity scans

```bash
python py/run_sensitivity_experiments.py --fold 0 --max_workers 6
python py/generate_sensitivity_reports.py
```

### 7. Export dataset statistics

```bash
python py/generate_dataset_statistics_report.py
```

### 8. TensorBoard

```bash
tensorboard --logdir runs --port 6006
```

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

## Notes

- Large runtime outputs such as `logs/`, `runs/`, `records/`, `data/`, and `tmp/` are intentionally ignored by Git.
- The `py/` analysis scripts import shared dataset definitions from [geomatric/experiment_catalog.py](./geomatric/experiment_catalog.py), so the active package path should be kept intact.
- The codebase does not commit the downloaded dataset content itself; only the expected local data roots are documented here.

 
## format
- https://peerj.com/articles/cs-3773/
- https://peerj.com/articles/cs-3762/
- https://www.overleaf.com/latex/templates/latex-template-for-peerj-journal-and-pre-print-submissions/ptdwfrqxqzbn
- https://peerj.com/about/author-instructions/#instruction-standard-sections-rass
- https://peerj.com/about/policies-and-procedures/#discipline-standards
- https://peerj.com/about/author-instructions/#reference-format
