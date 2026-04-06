# Final Implementation Todo List

## Goal

This document defines the final implementation plan for the topic-facing paper revision.

The revised paper will:

1. keep the core contribution as `Cross-Residual GNN`
2. keep the biological narrative focused on the protein-oriented TU package
3. unify all dataset descriptions and dataset statistics in one place
4. redraw all figures under the style requirements in [figures/paint.md](/ds1/workspace/ai/cross_residual_gnn/figures/paint.md)
5. strengthen the abstract and introduction with a plant-oriented motivation bridge
6. reorganize references from [paper/paper_corpus_merged.json](/ds1/workspace/ai/cross_residual_gnn/paper/paper_corpus_merged.json) in a logical citation order

This revision is still not positioned as a completed integrative-omics paper.
The target positioning is:

- protein-oriented biomolecular graph learning
- and a clearly stated extension path toward plant-related graph inference

## Final Narrative Scope

### Main Datasets In The Paper

The main text should focus on these three datasets:

- `PROTEINS`
- `DD`
- `ENZYMES`

### Supplementary Datasets

These datasets remain useful as robustness evidence, but should not drive the topic-facing narrative:

- `MUTAG`
- `AIDS`
- `Mutagenicity`

### Removed From Main Text

- `MSRC_9`

`MSRC_9` should not appear in the topic-facing main paper narrative.
If needed later, it can stay only in appendix-style material.

## Dataset Presentation Requirement

All datasets must be introduced together in one coherent dataset section instead of being discussed in scattered fragments.

The unified dataset presentation must include:

1. biological role of each dataset
2. task type
3. source
4. split protocol
5. label type
6. node / edge / graph statistics
7. whether the dataset supports the main biological claim or only supplementary robustness

### Required Dataset Tables

At minimum, produce these two tables:

1. Main biological dataset summary table
   - rows: `PROTEINS`, `DD`, `ENZYMES`
   - columns:
     - dataset
     - source
     - task type
     - split protocol
     - number of graphs
     - number of classes
     - feature dimension
     - average nodes
     - average edges
     - role in paper

2. Supplementary robustness dataset table
   - rows: `MUTAG`, `AIDS`, `Mutagenicity`
   - columns:
     - dataset
     - source
     - task type
     - number of graphs
     - number of classes
     - feature dimension
     - average nodes
     - average edges
     - note

## Workstream A: Focused Biological Package

### Objective

Keep the paper and experiment pipeline centered on the focused protein-oriented TU benchmark package.

### Files To Modify

- [geomatric/graph_classify_v3.py](/ds1/workspace/ai/cross_residual_gnn/geomatric/graph_classify_v3.py)
- [py/run_paper_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/run_paper_experiments.py)
- [py/summarize_paper_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/summarize_paper_experiments.py)
- [py/generate_all_result_reports.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_all_result_reports.py)
- [py/generate_suite_analysis_figures.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_suite_analysis_figures.py)
- [README.md](/ds1/workspace/ai/cross_residual_gnn/README.md)
- [requirements.txt](/ds1/workspace/ai/cross_residual_gnn/requirements.txt)

### Implementation Tasks

1. Keep `PROTEINS`, `DD`, and `ENZYMES` as the main biological package.
2. Keep `MUTAG`, `AIDS`, and `Mutagenicity` as supplementary robustness datasets.
3. Remove `MSRC_9` from the topic-facing paper narrative.
4. Ensure report scripts reflect this dataset separation consistently.

## Workstream B: Unified Dataset Statistics

### Objective

Create one reproducible statistics path that can export dataset facts for all active TU datasets.

### Files To Modify

- [geomatric/graph_classify_v3.py](/ds1/workspace/ai/cross_residual_gnn/geomatric/graph_classify_v3.py)
- [py/export_analysis_artifacts.py](/ds1/workspace/ai/cross_residual_gnn/py/export_analysis_artifacts.py)
- [py/generate_all_result_reports.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_all_result_reports.py)

### Implementation Tasks

1. Refactor dataset statistics generation so it works for the active paper datasets through one shared path.
2. Export unified dataset statistics to a structured artifact file.
3. Generate paper-ready LaTeX tables for:
   - main biological datasets
   - supplementary robustness datasets
4. Ensure the dataset section in the paper can rely on generated values rather than hand-maintained numbers.

### Required Output Artifacts

- `md/dataset_statistics_summary.md`
- `md/dataset_statistics_tables.tex`

## Workstream C: Figure System Redesign

### Objective

Redraw all figures under one consistent, paper-grade matplotlib style aligned with [figures/paint.md](/ds1/workspace/ai/cross_residual_gnn/figures/paint.md).

### Files To Modify

- [py/generate_suite_analysis_figures.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_suite_analysis_figures.py)
- [py/generate_sensitivity_reports.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_sensitivity_reports.py)
- [py/generate_exp_figures.py](/ds1/workspace/ai/cross_residual_gnn/py/generate_exp_figures.py)

### Files To Add

- `py/plot_style.py`

### Implementation Tasks

1. Create one shared plotting style module.
2. Enforce:
   - white background
   - Times New Roman or compatible serif fallback
   - paper-scale font sizes
   - colorblind-friendly palette
   - fixed model-to-color mapping
   - fixed model-to-marker mapping
   - fixed model-to-linestyle mapping
3. Export all figures to:
   - PDF
   - high-resolution PNG
4. Remove visually inconsistent defaults currently spread across scripts.

### Figures To Redraw

1. Main results figure
2. Cross-advantage figure
3. Rank / winner summary figure
4. Topic-facing focused results figure
5. Depth sensitivity figure
6. Dropout sensitivity figure
7. Learning-rate sensitivity figure

### New Figures To Add

1. Focused biological comparison
   - `PROTEINS`
   - `DD`
   - `ENZYMES`
2. Protein-related summary figure highlighting:
   - residual vs cross
   - where cross helps
   - where residual remains stronger

## Workstream D: Paper Revision

### Objective

Rewrite the paper around a cleaner topic-facing biological package, while adding a carefully framed plant-oriented motivation bridge.

### Files To Modify

- [paper/sections/abstract.tex](/ds1/workspace/ai/cross_residual_gnn/paper/sections/abstract.tex)
- [paper/sections/introduction.tex](/ds1/workspace/ai/cross_residual_gnn/paper/sections/introduction.tex)
- [paper/sections/datasets.tex](/ds1/workspace/ai/cross_residual_gnn/paper/sections/datasets.tex)
- [paper/sections/experiments.tex](/ds1/workspace/ai/cross_residual_gnn/paper/sections/experiments.tex)
- [paper/sections/conclusion.tex](/ds1/workspace/ai/cross_residual_gnn/paper/sections/conclusion.tex)
- [README.md](/ds1/workspace/ai/cross_residual_gnn/README.md)

### Abstract Tasks

1. Keep the biological evidence package focused on `PROTEINS`, `DD`, and `ENZYMES`.
2. Clarify the paper is protein-oriented, not full integrative omics.
3. Add one sentence that logically connects the method to plant-related graph inference.
4. Keep the plant statement aspirational but defensible.

### Introduction Tasks

1. Add a short systems-biology motivation bridge toward plant graph inference.
2. Explain why information preservation across graph layers matters in:
   - proteins
   - gene/protein interaction systems
   - plant biological networks
3. State clearly that current experiments are still protein-oriented.
4. Position plant relevance as an extension path with real methodological justification, not as a claimed completed validation.

### Dataset Section Tasks

1. Rewrite the dataset section so all datasets are presented together.
3. Add generated dataset statistics tables.
4. Separate:
   - main biological package
   - supplementary robustness package

### Experiments Section Tasks

1. Keep the protocol section centered on the unified TU stratified evaluation path.
2. Keep the main interpretation focused on `PROTEINS`, `DD`, and `ENZYMES`, without any canceled extension dataset references.
3. Remove `MSRC_9` from the main topic-facing result discussion.
4. Reframe the main finding around protein-related datasets first.

### Conclusion Tasks

1. Reaffirm current evidence scope.
2. Add the plant-oriented extension logic.
3. Keep future work grounded in:
   - plant graph inference
   - richer biological signals
   - omics-derived feature integration

## Workstream E: References And Citation Order

### Objective

Re-parse [paper/paper_corpus_merged.json](/ds1/workspace/ai/cross_residual_gnn/paper/paper_corpus_merged.json) and rebuild the reference usage order around the final paper logic.

### Files To Modify

- [paper/references.bib](/ds1/workspace/ai/cross_residual_gnn/paper/references.bib)
- [paper/sections/introduction.tex](/ds1/workspace/ai/cross_residual_gnn/paper/sections/introduction.tex)
- [paper/sections/related_work.tex](/ds1/workspace/ai/cross_residual_gnn/paper/sections/related_work.tex)
- [paper/sections/datasets.tex](/ds1/workspace/ai/cross_residual_gnn/paper/sections/datasets.tex)

### Files To Add

- `py/organize_references_from_corpus.py`
- `md/reference_logic_map.md`

### Implementation Tasks

1. Parse `paper_corpus_merged.json`.
2. Extract the `papers` list into a structured working view.
3. Group references in this final logic order:
   - GNN foundations
   - deep GNN degradation and training stability
   - residual / cross-layer / multi-branch architectures
   - graph classification and benchmark dataset context
   - biomolecular and protein-related graph learning
   - graph datasets and protein-oriented benchmarks
   - plant-oriented systems biology / computational biology motivation
4. Identify missing references needed for:
   - protein-oriented benchmark framing
   - plant-oriented motivation
5. Update `references.bib`.
6. Build a markdown map that records:
   - citation group
   - bib keys
   - intended section usage

### Reference Rule

References must not simply accumulate.
They must appear in a logical narrative order in the paper.

## Workstream F: Analysis Reinforcement

### Objective

Add the lowest-cost analysis items that improve reviewer trust without creating a second large project.

### Files To Modify

- [md/paper_gap_checklist.md](/ds1/workspace/ai/cross_residual_gnn/md/paper_gap_checklist.md)
- [paper/sections/experiments.tex](/ds1/workspace/ai/cross_residual_gnn/paper/sections/experiments.tex)

### Implementation Tasks

1. Add failure-mode discussion for:
   - where cross underperforms residual
   - why graph-level cross is more sensitive
   - why small datasets remain unstable
2. If time permits, add significance checks for the closest topic-facing comparisons.
3. Ensure the completed analysis items are visible in the paper, not only in scratch notes.

## Execution Order

The implementation must proceed in this order:

1. reference reorganization from `paper_corpus_merged.json`
2. unified dataset statistics export
3. result report update
4. figure system redesign and full redraw
5. paper text revision
6. final consistency pass across README, manuscript, generated tables, and figures

## Definition Of Done

The revision is complete only when all of the following are true:

1. The main text centers on `PROTEINS`, `DD`, and `ENZYMES`.
2. All datasets are described together in one clean section with generated statistics tables.
3. All figures are redrawn under the `paint.md` style requirements.
4. Abstract and introduction include a careful plant-oriented motivation bridge.
5. References are reorganized from `paper_corpus_merged.json` and cited in logical order.
6. `MSRC_9` is removed from the topic-facing main paper.
7. The manuscript no longer overclaims integrative omics validation.
