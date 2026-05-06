# Cross-Residual Graph Neural Networks

[中文说明 / Chinese README](./README-CN.md)

Research code, experiment assets, and manuscript sources for the CR-GNN study. The repository is now organized around the final `V3` pipeline and a submission-facing manuscript package in [`paper/`](./paper).

## Submission Entry

- English manuscript: [`paper/main.tex`](./paper/main.tex)
- Chinese manuscript: [`paper/main_chinese.tex`](./paper/main_chinese.tex)
- English PDF: [`paper/main.pdf`](./paper/main.pdf)
- Chinese PDF: [`paper/main_chinese.pdf`](./paper/main_chinese.pdf)
- submission manifest: [`paper/SUBMISSION_MANIFEST.md`](./paper/SUBMISSION_MANIFEST.md)

If you only care about the paper package, start from `paper/`. If you need to reproduce the final tables and figures, start from the `V3` code and report pipeline below.

## Final V3 State

- active training entry: [`geomatric/graph_classify_v3.py`](./geomatric/graph_classify_v3.py)
- final workflow entry: [`py/run_final_v3_pipeline.py`](./py/run_final_v3_pipeline.py)
- final active summaries: [`md/all_exp_tables.tex`](./md/all_exp_tables.tex), [`md/statistical_tests_main.tex`](./md/statistical_tests_main.tex), [`md/statistical_tests_supp.tex`](./md/statistical_tests_supp.tex)
- final active figures: `figures/exp/*_V3.pdf` and `paper/figures/exp/*_V3.pdf`
- project index: [`md/FINAL_PROJECT_INDEX.md`](./md/FINAL_PROJECT_INDEX.md)

Only `V3` should be treated as current. `V1` and `V2` directories are retained as historical archives for reproducibility, not as active editing targets.

## Repository Layout

```text
cross_residual_gnn/
├── data/                      # Local datasets, not committed
├── figures/                   # Exported experiment figures
├── geomatric/                 # Active Python package
├── logs/                      # Historical runtime logs (archive / reproducibility)
├── md/                        # Final tables plus working notes and legacy summaries
├── paper/                     # Current LaTeX manuscript and paper figures
├── py/                        # Batch runners and analysis / export scripts
├── records/                   # Final analysis JSON and version records
├── runs/                      # TensorBoard archives (historical)
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

Summarize and export final report artifacts:

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
- [`records/experiment_versions.json`](./records/experiment_versions.json)

Paper source:

- [`paper/main.tex`](./paper/main.tex)
- [`paper/main_chinese.tex`](./paper/main_chinese.tex)
- [`paper/sections/01_introduction_peerj.tex`](./paper/sections/01_introduction_peerj.tex)
- [`paper/sections/02_methods_peerj.tex`](./paper/sections/02_methods_peerj.tex)
- [`paper/sections/03_results_peerj.tex`](./paper/sections/03_results_peerj.tex)
- [`paper/sections/04_discussion_peerj.tex`](./paper/sections/04_discussion_peerj.tex)
- [`paper/sections/05_conclusions_peerj.tex`](./paper/sections/05_conclusions_peerj.tex)
- [`paper/sections_cn/`](./paper/sections_cn)

Active paper figures:

- [`paper/figures/exp/fig1_full_suite_results_V3.pdf`](./paper/figures/exp/fig1_full_suite_results_V3.pdf)
- [`paper/figures/exp/fig2_cross_advantage_heatmap_V3.pdf`](./paper/figures/exp/fig2_cross_advantage_heatmap_V3.pdf)
- [`paper/figures/exp/fig4_topic_focus_results_V3.pdf`](./paper/figures/exp/fig4_topic_focus_results_V3.pdf)
- [`paper/figures/exp/fig5_protein_package_summary_V3.pdf`](./paper/figures/exp/fig5_protein_package_summary_V3.pdf)

## Paper Status

The current manuscript has been tightened to a protein-oriented benchmark paper:

- main biological evidence: `PROTEINS` and `DD`
- supporting biological stress test: `ENZYMES`
- supplementary robustness datasets: `MUTAG`, `AIDS`, `Mutagenicity`

Both English and Chinese PDFs compile successfully. Remaining manuscript warnings are formatting-level only, mainly float placement, a few wide tables, and PDF version warnings for two architecture figures.

## Dataset Grouping

Defined in [`geomatric/experiment_catalog.py`](./geomatric/experiment_catalog.py):

- biological core: `PROTEINS`, `DD`, `ENZYMES`
- supplementary robustness: `MUTAG`, `AIDS`, `Mutagenicity`

## Archive Policy

- Raw datasets are not committed.
- `logs/V1`, `logs/V2`, `runs/V1`, and `runs/V2` are archive-only.
- `logs/V3` and `runs/V3` are reproducibility records, not paper source.
- Working notes in `md/` can include legacy or exploratory files; use the active non-suffixed `V3`-aligned tables and the `paper/` manifest as source of truth.

## Notes

- If you update paper claims, use the current `V3` tables and figures, not older `LATEST`-style notes.
- If you prepare a clean handoff, point collaborators directly to [`paper/SUBMISSION_MANIFEST.md`](./paper/SUBMISSION_MANIFEST.md).
