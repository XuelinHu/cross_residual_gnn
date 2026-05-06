# Cross-Residual Graph Neural Networks

[中文说明 / Chinese README](./README-CN.md)

Research code, experiment assets, and manuscript sources for the CR-GNN study. The repository has been consolidated around the final `V3` pipeline and the current paper-facing manuscript in [`paper/`](./paper).

## Final V3 State

- active training entry: [`geomatric/graph_classify_v3.py`](./geomatric/graph_classify_v3.py)
- final workflow entry: [`py/run_final_v3_pipeline.py`](./py/run_final_v3_pipeline.py)
- final experiment assets: `logs/V3`, `records/V3`, `runs/V3`
- project index: [`md/FINAL_PROJECT_INDEX.md`](./md/FINAL_PROJECT_INDEX.md)
- manuscript entry: [`paper/main.tex`](./paper/main.tex)

The `V3` package is the only version that should be treated as current. Earlier rerun notes and obsolete archive folders are not part of the active workflow anymore.

## Repository Layout

```text
cross_residual_gnn/
├── data/                      # Local datasets, not committed
├── figures/                   # Exported experiment figures
├── geomatric/                 # Active Python package
├── logs/                      # Runtime logs and archived V1/V2/V3 outputs
├── md/                        # Generated summaries, tables, and paper notes
├── paper/                     # Current LaTeX manuscript and paper figures
├── py/                        # Batch runners and analysis/report scripts
├── records/                   # Text summaries and archived V1/V2/V3 records
├── runs/                      # TensorBoard runs
├── README.md
└── README-CN.md
```

## Environment

Per repository instructions, use the Conda environment `pyg` for Python work:

```bash
conda activate pyg
```

Recommended dependencies:

- Python `3.10+`
- PyTorch
- PyTorch Geometric
- `tensorboard`
- `matplotlib`, `seaborn`, `pandas`, `openpyxl`

## Main Commands

Single run:

```bash
conda activate pyg
python -m geomatric.graph_classify_v3 --mode single --ds PROTEINS
```

Final V3 pipeline:

```bash
conda activate pyg
python py/run_final_v3_pipeline.py
python py/run_final_v3_pipeline.py --steps consolidate summarize reports figures
```

Full benchmark batch:

```bash
conda activate pyg
python py/run_paper_experiments.py --dataset_group all --max_workers 6 --tensorboard
```

Summarize and export report artifacts:

```bash
conda activate pyg
python py/summarize_paper_experiments.py --dataset_group all
python py/generate_all_result_reports.py
python py/generate_suite_analysis_figures.py
```

Sensitivity and dataset statistics:

```bash
conda activate pyg
python py/run_sensitivity_experiments.py --fold 0 --max_workers 6
python py/generate_sensitivity_reports.py
python py/generate_dataset_statistics_report.py
```

## Key Files

Code and configuration:

- [`geomatric/graph_classify_v3.py`](./geomatric/graph_classify_v3.py)
- [`geomatric/experiment_catalog.py`](./geomatric/experiment_catalog.py)
- [`geomatric/experiment_paths.py`](./geomatric/experiment_paths.py)
- [`py/run_final_v3_pipeline.py`](./py/run_final_v3_pipeline.py)

Experiment summaries:

- [`md/EXPERIMENT_INDEX.md`](./md/EXPERIMENT_INDEX.md)
- [`md/FINAL_PROJECT_INDEX.md`](./md/FINAL_PROJECT_INDEX.md)
- [`md/V3_residual_summary.md`](./md/V3_residual_summary.md)
- [`md/all_exp_tables_V3.tex`](./md/all_exp_tables_V3.tex)
- [`md/all_results_summary_V3.txt`](./md/all_results_summary_V3.txt)
- [`md/frontiers_topic_alignment.md`](./md/frontiers_topic_alignment.md)

Paper source:

- [`paper/main.tex`](./paper/main.tex)
- [`paper/sections/01_introduction_peerj.tex`](./paper/sections/01_introduction_peerj.tex)
- [`paper/sections/02_methods_peerj.tex`](./paper/sections/02_methods_peerj.tex)
- [`paper/sections/03_results_peerj.tex`](./paper/sections/03_results_peerj.tex)
- [`paper/sections/04_discussion_peerj.tex`](./paper/sections/04_discussion_peerj.tex)
- [`paper/sections/05_conclusions_peerj.tex`](./paper/sections/05_conclusions_peerj.tex)

Paper figures already available:

- [`paper/figures/exp/fig1_full_suite_results.pdf`](./paper/figures/exp/fig1_full_suite_results.pdf)
- [`paper/figures/exp/fig2_cross_advantage_heatmap.pdf`](./paper/figures/exp/fig2_cross_advantage_heatmap.pdf)
- [`paper/figures/exp/fig4_topic_focus_results.pdf`](./paper/figures/exp/fig4_topic_focus_results.pdf)
- [`paper/figures/exp/fig5_protein_package_summary.pdf`](./paper/figures/exp/fig5_protein_package_summary.pdf)

## Paper Status

The current manuscript is positioned as:

- a biomolecular graph-classification methods paper
- with `PROTEINS`, `DD`, and `ENZYMES` as the biological core
- and `MUTAG`, `AIDS`, `Mutagenicity` as supplementary robustness datasets

This is not yet a cleanly finished paper-facing draft. The current source still needs:

- tighter scope alignment around the V3 narrative
- direct alignment between claims and the current tables
- fuller mathematical definitions in the methods section
- reintroduction of the existing main result figures into the body text

For the working assessment and venue-fit notes, start from [`md/frontiers_topic_alignment.md`](./md/frontiers_topic_alignment.md).

## Dataset Grouping

Defined in [`geomatric/experiment_catalog.py`](./geomatric/experiment_catalog.py):

- biological core: `PROTEINS`, `DD`, `ENZYMES`
- supplementary robustness: `MUTAG`, `AIDS`, `Mutagenicity`

## Notes

- Raw datasets are not committed.
- Runtime directories can be large and may contain archived V1/V2/V3 outputs.
- If you are updating paper claims, treat the V3 tables and figures as the source of truth, not older `LATEST`-style notes.
