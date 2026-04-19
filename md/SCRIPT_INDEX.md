# Script Index

## Core training and path management

- [graph_classify_v3.py](/ds1/workspace/ai/cross_residual_gnn/geomatric/graph_classify_v3.py)
  - main training entry
  - supports learnable and fixed gate
  - supports `learnable|topk|sparse` residual modes
- [experiment_paths.py](/ds1/workspace/ai/cross_residual_gnn/geomatric/experiment_paths.py)
  - version path management
  - current default version is `V3`

## Main experiment runners

- [run_paper_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/run_paper_experiments.py)
  - full benchmark runner
- [run_missing_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/run_missing_experiments.py)
  - rerun missing jobs
- [summarize_paper_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/summarize_paper_experiments.py)
  - summarize main benchmark

## Supplementary experiment runners

- [run_aids_supplementary_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/run_aids_supplementary_experiments.py)
  - AIDS gate ablation and local parameter scans
- [run_cross_gate_ablation_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/run_cross_gate_ablation_experiments.py)
  - cross + gate ablation
- [run_residual_mode_ablation_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/run_residual_mode_ablation_experiments.py)
  - learnable vs top-k vs sparse residual
- [run_residual_parameter_sweeps.py](/ds1/workspace/ai/cross_residual_gnn/py/run_residual_parameter_sweeps.py)
  - `topk_ratio` / `sparse_lambda` 5-fold sweeps
- [run_sensitivity_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/run_sensitivity_experiments.py)
  - early cross sensitivity scans in `V1`

## Consolidation and reporting

- [consolidate_final_v3.py](/ds1/workspace/ai/cross_residual_gnn/py/consolidate_final_v3.py)
  - merge accepted `V2` artifacts into final `V3`
- [generate_all_result_reports.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_all_result_reports.py)
  - report generation
- [generate_sensitivity_reports.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_sensitivity_reports.py)
  - sensitivity report generation
- [generate_suite_analysis_figures.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_suite_analysis_figures.py)
  - suite-level analysis figures
- [generate_exp_figures.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_exp_figures.py)
  - experiment plots
- [generate_method_figure.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_method_figure.py)
  - method figure assets
- [generate_topic_tables.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_topic_tables.py)
  - topic-aligned tables
- [generate_dataset_statistics_report.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_dataset_statistics_report.py)
  - dataset statistics summaries
- [export_analysis_artifacts.py](/ds1/workspace/ai/cross_residual_gnn/py/export_analysis_artifacts.py)
  - export artifacts for external use

## Utility or one-off scripts

- [run_enzymes_tuned_cross.py](/ds1/workspace/ai/cross_residual_gnn/py/run_enzymes_tuned_cross.py)
  - one-off tuned run for ENZYMES
- [organize_references_from_corpus.py](/ds1/workspace/ai/cross_residual_gnn/py/organize_references_from_corpus.py)
  - reference organization
- [plot_style.py](/ds1/workspace/ai/cross_residual_gnn/py/plot_style.py)
  - shared plotting style

## Recommended scripts to keep central

If you only keep the core paper-facing scripts in mind, focus on:

1. `geomatric/graph_classify_v3.py`
2. `py/run_paper_experiments.py`
3. `py/run_residual_mode_ablation_experiments.py`
4. `py/run_residual_parameter_sweeps.py`
5. `py/consolidate_final_v3.py`
6. `py/generate_all_result_reports.py`
