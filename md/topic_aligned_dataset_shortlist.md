# Topic-Aligned Dataset Shortlist

> Note: this shortlist has been normalized to the final paper scope, and only the retained extension directions remain below.

## Purpose

This note ranks candidate datasets by two constraints:

1. how close they are to the Frontiers topic on genes, proteins, motifs, and integrative omics
2. how easily they can be integrated into the current codebase

## Current Code Constraint

The easiest path is still graph classification based on PyG `DataLoader` over graph samples from `TUDataset`.

Current code fit:

- easiest: graph classification datasets with `x`, `edge_index`, and graph-level `y`
- medium effort: graph classification datasets with official train/val/test splits and custom metrics
- higher effort: node classification or multi-label prediction datasets
- highest effort: datasets whose task is sequence design or 3D structure generation rather than classification

## Ranked Recommendations

### Tier 1: Add Immediately

#### 1. ENZYMES

Why it fits:

- protein-related and biologically interpretable
- still in the same TU ecosystem as your current experiments
- very low engineering cost

Why it helps the paper:

- much closer to enzyme or protein-function language than `MUTAG` or `Mutagenicity`
- supports a biomolecular graph narrative without changing the training pipeline

Integration cost:

- minimal
- likely just another `TUDataset` name in the current graph classification pipeline

Recommended use:

- add to the topic-facing main experiments together with `PROTEINS` and `DD`

#### 2. Keep PROTEINS and DD as Main Biological Benchmarks

Why they stay:

- already integrated
- already tested
- already the best biological anchors in the current project

Recommended use:

- these should remain the main datasets for the topic-facing version

### Tier 2: Strong Topic Fit, But Requires Task Expansion

#### 3. PPI (PyG)

Why it fits:

- protein-protein interaction networks
- node features include positional gene sets, motif gene sets, and immunological signatures
- labels are gene ontology sets

Why it helps the paper:

- this is much closer to the topic wording about genes, motifs, and proteins
- gives you a concrete bridge from graph learning into biological function prediction

Integration cost:

- medium to high
- this is node-level multi-label prediction, not graph classification
- requires adapting or rebuilding the node classification pipeline with the cross-residual blocks

Recommended use:

- strongest near-term option if you want to start touching genuine biological function prediction

#### 4. ogbn-proteins

Why it fits:

- protein nodes with biologically meaningful associations
- multi-label protein function prediction
- species-based split evaluates cross-species generalization

Why it helps the paper:

- much closer to protein function inference than TU benchmarks
- supports a stronger biological claim than `PROTEINS` and `DD`

Integration cost:

- high
- large-scale node property prediction
- requires custom evaluation with ROC-AUC
- uses official species split and edge features

Recommended use:

- best upgrade if you want a serious biology-facing benchmark and can afford engineering work

### Tier 4: Long-Term Extensions

#### 6. ProteinMPNNDataset

Why it fits:

- strongly protein-centered
- directly tied to protein structure and sequence design

Why it helps the paper:

- opens a path toward structure-aware protein inference

Integration cost:

- very high
- current task mismatch is substantial
- this is not a simple drop-in classification benchmark for the current V3 pipeline

Recommended use:

- not for the current paper deadline

## Recommended Adoption Order

If you want the best return for the least engineering effort, adopt datasets in this order:

1. `ENZYMES`
2. `PPI`
3. `ogbn-proteins`

## Suggested Topic-Facing Experimental Bundle

### Minimal-Change Bundle

- `PROTEINS`
- `DD`
- `ENZYMES`

This bundle is the easiest way to improve topic fit while keeping your current graph classification code path.

### Stronger Topic-Alignment Bundle

- `PROTEINS`
- `DD`
- `ENZYMES`
- `PPI`

This bundle is stronger semantically because it adds motif and gene-ontology supervision, but it requires node-level work.

## Recommended Decision

For the current codebase, the most pragmatic plan is:

1. add `ENZYMES` now
2. keep `PROTEINS` and `DD` as core datasets
3. if you want to materially improve topic fit, make `PPI` the first node-level extension
4. if you can afford larger-scale engineering work, consider `ogbn-proteins`

## Source Notes

This shortlist is based on:

- TUDataset collection and dataset descriptions for `DD`, `ENZYMES`, and `PROTEINS`
- PyG dataset documentation for `PPI`
- official dataset documentation for `ogbn-proteins`
