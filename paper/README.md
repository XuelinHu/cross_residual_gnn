# Paper README

This directory contains the current `V3` manuscript source for the CR-GNN study.

## Primary Files

- main entry: [`main.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/main.tex)
- bibliography: [`references.bib`](/ds1/workspace/ai/cross_residual_gnn/paper/references.bib)
- sections:
  - [`sections/01_introduction_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/01_introduction_peerj.tex)
  - [`sections/02_methods_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/02_methods_peerj.tex)
  - [`sections/03_results_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/03_results_peerj.tex)
  - [`sections/04_discussion_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/04_discussion_peerj.tex)
  - [`sections/05_conclusions_peerj.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/05_conclusions_peerj.tex)
  - [`sections/06_acknowledgments.tex`](/ds1/workspace/ai/cross_residual_gnn/paper/sections/06_acknowledgments.tex)

## Current Positioning

The manuscript is currently framed as a biomolecular graph-classification methods paper:

- biological core: `PROTEINS`, `DD`, `ENZYMES`
- supplementary robustness datasets: `MUTAG`, `AIDS`, `Mutagenicity`
- main method comparison: plain, residual, and cross-residual reuse topologies

## Important Supporting Assets

Main tables and summaries are injected from the repository root:

- [`../md/all_exp_tables.tex`](/ds1/workspace/ai/cross_residual_gnn/md/all_exp_tables.tex)
- [`../md/statistical_tests_main.tex`](/ds1/workspace/ai/cross_residual_gnn/md/statistical_tests_main.tex)
- [`../md/statistical_tests_supp.tex`](/ds1/workspace/ai/cross_residual_gnn/md/statistical_tests_supp.tex)
- [`../md/peerj_gate_tables.tex`](/ds1/workspace/ai/cross_residual_gnn/md/peerj_gate_tables.tex)
- [`../md/peerj_residual_tables.tex`](/ds1/workspace/ai/cross_residual_gnn/md/peerj_residual_tables.tex)

Main figure assets already available:

- [`figures/exp/fig1_full_suite_results.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp/fig1_full_suite_results.pdf)
- [`figures/exp/fig2_cross_advantage_heatmap.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp/fig2_cross_advantage_heatmap.pdf)
- [`figures/exp/fig4_topic_focus_results.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp/fig4_topic_focus_results.pdf)
- [`figures/exp/fig5_protein_package_summary.pdf`](/ds1/workspace/ai/cross_residual_gnn/paper/figures/exp/fig5_protein_package_summary.pdf)

## Compilation

From this directory:

```bash
latexmk -pdf main.tex
```

or:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Current Editing Priorities

1. Align abstract, results, discussion, and conclusion claims with the current V3 tables.
2. Reinsert the existing main result figures into the body text so the benchmark story has visible evidence.
3. Close notation gaps in the methods section, especially for `psi`, `Phi_in`, `Pool`, `B_t`, `T`, and the graph-level cross-residual recurrence.
4. Keep the main storyline centered on the biological core and demote supplementary datasets when making paper-facing claims.
