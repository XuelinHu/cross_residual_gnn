# Paper README

This directory contains the submission-facing manuscript package for the CR-GNN study.

## Main Deliverables

- English source: [`main.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/main.tex)
- Chinese source: [`main_chinese.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/main_chinese.tex)
- English PDF: [`main.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/main.pdf)
- Chinese PDF: [`main_chinese.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/main_chinese.pdf)
- submission manifest: [`SUBMISSION_MANIFEST.md`](/ds1/workspace/ai/cross_residual_gnn/paper/SUBMISSION_MANIFEST.md)
- bibliography: [`references.bib`](/ds1/workspace/ai/cross_residual_gnn/paper/references.bib)

## Source Layout

English sections:

- [`sections/01_introduction_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/01_introduction_peerj.tex)
- [`sections/02_methods_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/02_methods_peerj.tex)
- [`sections/03_results_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/03_results_peerj.tex)
- [`sections/04_discussion_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/04_discussion_peerj.tex)
- [`sections/05_conclusions_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/05_conclusions_peerj.tex)
- [`sections/06_acknowledgments.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/06_acknowledgments.tex)
- [`sections/07_appendix.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/07_appendix.tex)

Chinese sections:

- [`sections_cn/01_introduction.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections_cn/01_introduction.tex)
- [`sections_cn/02_methods.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections_cn/02_methods.tex)
- [`sections_cn/03_results.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections_cn/03_results.tex)
- [`sections_cn/04_discussion.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections_cn/04_discussion.tex)
- [`sections_cn/05_conclusions.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections_cn/05_conclusions.tex)
- [`sections_cn/06_acknowledgments.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections_cn/06_acknowledgments.tex)
- [`sections_cn/07_appendix.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections_cn/07_appendix.tex)

## Current Manuscript Positioning

The paper is now framed as a protein-oriented graph-classification benchmark study:

- main biological evidence: `PROTEINS` and `DD`
- supporting biological stress test: `ENZYMES`
- supplementary robustness datasets: `MUTAG`, `AIDS`, `Mutagenicity`
- main comparison axis: plain, residual, and cross-residual reuse topologies

## Supporting Assets

Active English table inputs from repository root:

- [`../md/all_exp_tables.tex`](/ds1/workspace/ai/cross_residual_gnn/md/all_exp_tables.tex)
- [`../md/all_exp_tables_appendix.tex`](/ds1/workspace/ai/cross_residual_gnn/md/all_exp_tables_appendix.tex)
- [`../md/statistical_tests_main.tex`](/ds1/workspace/ai/cross_residual_gnn/md/statistical_tests_main.tex)
- [`../md/statistical_tests_supp.tex`](/ds1/workspace/ai/cross_residual_gnn/md/statistical_tests_supp.tex)
- [`../md/peerj_gate_tables.tex`](/ds1/workspace/ai/cross_residual_gnn/md/peerj_gate_tables.tex)
- [`../md/peerj_residual_tables.tex`](/ds1/workspace/ai/cross_residual_gnn/md/peerj_residual_tables.tex)

Chinese table inputs:

- [`md_cn/all_exp_tables.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/md_cn/all_exp_tables.tex)
- [`md_cn/all_exp_tables_appendix.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/md_cn/all_exp_tables_appendix.tex)
- [`md_cn/statistical_tests_main.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/md_cn/statistical_tests_main.tex)
- [`md_cn/statistical_tests_supp.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/md_cn/statistical_tests_supp.tex)

Active figure assets:

- [`figures/exp/fig1_full_suite_results_V3.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp/fig1_full_suite_results_V3.pdf)
- [`figures/exp/fig2_cross_advantage_heatmap_V3.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp/fig2_cross_advantage_heatmap_V3.pdf)
- [`figures/exp/fig4_topic_focus_results_V3.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp/fig4_topic_focus_results_V3.pdf)
- [`figures/exp/fig5_protein_package_summary_V3.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp/fig5_protein_package_summary_V3.pdf)
- [`figures/cr_gnn_node_architecture.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/cr_gnn_node_architecture.pdf)
- [`figures/cr_gnn_graph_architecture.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/cr_gnn_graph_architecture.pdf)

Legacy and exploratory figures are intentionally retained in `paper/figures/exp/` for traceability, but the `_V3` versions are the active paper-facing ones.

## Compilation

From this directory:

```bash
latexmk -pdf main.tex
latexmk -xelatex main_chinese.tex
```

## Status

Both PDFs compile successfully. Remaining warnings are non-blocking and mainly concern float placement, wide tables, and PDF version warnings for two architecture figures.
