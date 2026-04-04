# Frontiers Topic Alignment Notes

## Target Topic

Submission target:

- Frontiers Research Topic: `Prediction of Novel Domains, Motifs, Genes, and Proteins Through Integrative Omics Approaches`
- Topic URL: `https://www.frontiersin.org/research-topics/73895/prediction-of-novel-domains-motifs-genes-and-proteins-through-integrative-omics-approaches`

## Bottom-Line Fit Assessment

The current project is methodologically strong enough for a paper draft, but the present dataset mix is only a partial match for the target topic.

Current work is strongest as:

- a graph representation learning paper
- a graph classification paper
- a cross-residual GNN architecture paper

Current work is weaker as:

- an integrative omics paper
- a novel gene/protein/domain/motif prediction paper
- a plant bioinformatics paper

The main issue is not model quality. The main issue is scope alignment.

## Current Datasets vs Topic Relevance

### Strongest Current Matches

- `PROTEINS`
  - Closest to the topic because it is protein-related.
  - Can support a protein function or protein structure classification narrative.
  - Still not a true integrative omics dataset.

- `DD`
  - Still biologically adjacent because it is protein-related.
  - Usable as a supporting protein graph benchmark.
  - Less directly interpretable than `PROTEINS`.

### Partial or Weak Matches

- `AIDS`
  - Biomedical, but mainly small-molecule or screening oriented.
  - Weak for a gene/protein/domain/motif prediction storyline.

- `Mutagenicity`
  - Toxicity / small-molecule orientation.
  - Weak match to the topic.

- `MUTAG`
  - Chemoinformatics benchmark.
  - Useful as a method stress test, but weak topic alignment.

- `MSRC_9`
  - Not biologically relevant to the target topic.
  - Should not appear in the main paper if this topic is the target venue.

## Recommended Dataset Policy

If the goal is to submit to this specific topic, use the following hierarchy.

### Main Table

- `PROTEINS`
- `DD`

These are the only current datasets that naturally support a protein-centered narrative.

### Supplementary or Appendix Only

- `MUTAG`
- `AIDS`
- `Mutagenicity`

These can be kept as generic robustness benchmarks, but they should not be used to define the biological claim.

### Remove From Topic-Facing Version

- `MSRC_9`

This dataset weakens the paper-topic fit and should be excluded from the topic-facing main text.

## Two Viable Submission Paths

### Path A: Minimal Reframing

Keep the current codebase and most experiments, but reposition the paper as a biomolecular graph learning method.

What changes:

- center the story on `PROTEINS` and `DD`
- move non-protein datasets to appendix
- avoid claiming integrative omics directly
- describe the method as a representation framework for biomolecular graph classification or biomolecular function inference

What this solves:

- makes the paper less generic
- improves topic proximity
- requires the least new engineering

What remains weak:

- still not a true omics integration paper
- still not directly about novel genes, domains, or motifs

### Path B: Stronger Topic Alignment

Extend the work from graph classification into biological function prediction with heterogeneous biological signals.

What changes:

- introduce at least one dataset closer to gene or protein function prediction
- combine graph structure with one more biological signal, such as:
  - sequence-derived embedding
  - protein-protein interaction
  - expression profile
  - domain annotation
  - phenotype or condition labels

What this solves:

- aligns much better with the topic wording
- gives a defensible reason to use the phrase `integrative omics`

What it costs:

- requires new data processing
- likely requires a node classification or heterogeneous graph setting, not only graph classification

## Best Near-Term Strategy

For the current codebase, the most practical route is:

1. keep the core contribution as `Cross-Residual GNN`
2. make `PROTEINS` and `DD` the biological main datasets
3. demote `MUTAG`, `AIDS`, and `Mutagenicity` to supplementary evidence
4. remove `MSRC_9` from the topic-facing main paper
5. add one stronger biological dataset if time allows

This is the smallest change that materially improves venue fit.

## Recommended Biological Narrative

Avoid this framing:

- generic graph classification
- a universal benchmark paper
- cross residual improves everything

Prefer this framing:

- cross-residual graph representation learning for biomolecular graphs
- improving feature propagation for protein-related graph inference
- deeper graph models for biological structure-function prediction

## Claims That Are Safe vs Unsafe

### Safer Claims

- The proposed cross-residual architecture improves information propagation across graph layers.
- The method is effective on protein-related graph benchmarks.
- Cross-residual design shows dataset-dependent benefits, with different cross variants excelling on different biomolecular graph datasets.

### Claims To Avoid Without New Data

- The method solves integrative omics prediction.
- The method predicts novel genes, motifs, or domains.
- The method is specifically validated for plant omics.

## Current Result Storyline

Based on the current experiments, the strongest defensible storyline is:

- Cross-residual connections are the main architectural novelty.
- The best cross-residual form is dataset-dependent.
- `GraphCrossGNN` is strongest on `MUTAG` in current runs.
- tuned `NodeCrossGNN` is stronger than tuned `GraphCrossGNN` on `PROTEINS` and `DD`.
- therefore, the key contribution is not a single fixed cross block, but a cross-residual design principle for biomolecular graph learning.

This storyline is stronger than claiming a single model dominates everywhere.

## Paper Structure Adjustments

### Title Direction

Avoid:

- Cross Residual Graph Neural Networks for General Graph Classification

Prefer:

- Cross-Residual Graph Neural Networks for Biomolecular Graph Representation Learning
- Cross-Residual Message Passing for Protein-Related Graph Classification
- Cross-Residual Graph Networks for Structure-Aware Biomolecular Inference

### Abstract Direction

Abstract should emphasize:

- biological motivation first
- architectural contribution second
- protein-related evaluation third

Suggested abstract logic:

1. Biological function prediction from molecular interaction or structure graphs is hard because deep GNNs oversmooth or lose discriminative signals.
2. We propose a cross-residual graph architecture that injects complementary representations across propagation paths.
3. We evaluate node-level and graph-level cross-residual variants on protein-related graph benchmarks.
4. Results show that cross-residual designs improve robustness or accuracy on selected biomolecular datasets.
5. The architecture provides a practical backbone for future integration with richer omics signals.

### Introduction Direction

Introduction should connect these points:

- need for reliable representation learning in biological graphs
- importance of preserving multi-scale signal
- instability of deeper GNNs on biomolecular tasks
- why cross-residual message passing is more suitable than simple identity skips

## Minimum Additional Experiments To Improve Fit

If time is limited, the highest-value additions are:

1. rerun the final focused comparison using only:
   - `PROTEINS`
   - `DD`
   - `NodeResGNN`
   - `NodeCrossGNN`
   - `GraphResGNN`
   - `GraphCrossGNN`
2. export TensorBoard and artifact plots for those protein-related datasets
3. add embedding heatmaps, confusion matrices, and convergence curves
4. report parameter count and runtime overhead of cross residuals

This does not create omics alignment, but it makes the biological narrative much cleaner.

## High-Value New Dataset Directions

If you are willing to extend the project, prioritize datasets in this order.

### Easiest Extension

- add one more protein-focused graph benchmark
- keep the task as graph classification
- use it to strengthen the biomolecular scope

This is easiest because it does not require rewriting the training objective.

### Better Topic Alignment

- protein-protein interaction data with function labels
- gene interaction network data with expression-derived features
- domain or family prediction tasks with sequence-derived embeddings

This is better aligned because it moves the work toward gene/protein function prediction and multi-signal integration.

## Submission Recommendation

If no new biological dataset is added:

- submit only if the paper is framed conservatively
- avoid overclaiming integrative omics
- treat the work as a biomolecular graph method paper with partial topic overlap

If one stronger biological dataset or one multi-signal dataset is added:

- the fit becomes much more defensible
- the same method contribution can be preserved with a stronger application layer

## Actionable Next Step

The best immediate next step is to prepare a topic-facing experiment package with:

- main datasets: `PROTEINS`, `DD`
- focused models: `NodeResGNN`, `NodeCrossGNN`, `GraphResGNN`, `GraphCrossGNN`
- topic-facing figures from `TensorBoard` and exported artifacts
- a rewritten title, abstract, and introduction centered on biomolecular graph inference
