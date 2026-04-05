# Paper Gap Checklist

## Current State

The manuscript is now materially more consistent than the earlier draft:

- the full-suite results are integrated into the paper
- the main analysis figures and parameter-sensitivity figures are integrated
- the protein-oriented main benchmark is separated from the broader robustness suite
- the final claim is conservative and defensible
- the experiment protocol is fixed and reproducible
- the repository root is aligned with the active V3 workflow

The remaining work is no longer about basic experimentation. It is mostly about submission readiness, presentation quality, and topical fit.

## Must Add Before Submission

### 1. Author Metadata

The title block in [paper/main.tex](/ds1/workspace/ai/cross_residual_gnn/paper/main.tex) still contains placeholders:

- `Your Name`
- institution placeholder
- email placeholder

These must be replaced before submission.

### 2. Reference Audit

The bibliography compiles, but it still needs a final audit:

- remove stale or unused references
- verify all cited works are actually relevant
- check formatting consistency
- confirm the final bibliography reflects the current framing

## Strongly Recommended Additions

### 3. Failure-Mode Discussion

The paper already says cross is dataset-dependent. What is still missing is a short failure-mode paragraph:

- where cross underperforms residual
- why graph-level cross appears more sensitive
- why plain stacking is still competitive on selected datasets

This would improve reviewer trust.

### 4. Optional Significance Check

Mean and standard deviation are already reported, which is acceptable. If time allows, a small significance check would strengthen the closest comparisons:

- `GraphResGNN` vs `NodeCrossGNN` on `PROTEINS`
- `NodeResGNN` vs `NodeCrossGNN` on `DD`
- `GraphResGNN` vs `GraphCrossGNN` on `ENZYMES`

## Topic-Fit Gaps

### 5. Still Not Truly Integrative Omics

The current manuscript is:

- biomolecular graph learning
- protein-oriented in its main benchmark

but it is not yet:

- multi-omics integration
- gene / motif / domain prediction

So the fit to the target Frontiers topic remains partial rather than exact.

### 6. No Strong Biological Feature Integration Yet

The current benchmarks are still structure-centric. The most valuable future extension would be:

- PPI-style node prediction
- ontology supervision
- sequence-derived embeddings
- richer biological annotations

## Nice-to-Have Polishing

### 7. Final Language Tightening Pass

The paper is logically consistent now, but still repeats some framing phrases:

- `protein-oriented`
- `full-suite`
- `topic-facing`

Once the science is frozen, one final language pass would help.

### 8. Float and Layout Review

The LaTeX compiles, but the paper should still get a final layout pass for:

- table float positions
- page breaks around experiments
- appendix readability

## Bottom Line

The main missing items are now:

- final author metadata
- one reference audit
- optional statistical and biology-facing reinforcement

The current blocker is no longer the experiment protocol. The remaining blockers are author metadata, reference quality, topical fit, and final language/layout polish.
