# Paper Final Draft Outline

## 1. Title direction

Use a title that emphasizes mechanism and boundary, not universal best performance.

Good direction:

- Cross-Residual Message Passing with Adaptive Residual Routing for Graph Classification

Avoid:

- wording that claims universal superiority or unconditional SOTA

## 2. Core paper claim

The paper should make a constrained but defensible claim:

- cross-branch historical interaction is useful, but not uniformly dominant
- its gains are operator- and dataset-dependent
- adaptive residual design matters
- sparse residual filtering can outperform standard learnable residual on several representative targets

This is stronger and more credible than forcing a global-best narrative.

## 3. Abstract structure

Paragraph logic:

1. Problem:
   graph classification models often rely on plain residual accumulation, which may not fully exploit cross-branch historical information.
2. Method:
   introduce a cross-residual framework with adaptive gate control and extended residual routing modes.
3. Main result:
   full benchmark on `6 datasets`, `9 methods`, `4 operators`, `5 folds`.
4. Key analysis:
   cross is not universally best, but shows clear advantages on several attention-oriented settings; residual mode design further affects outcome.
5. Conclusion:
   the method reveals when cross interaction and residual filtering are beneficial.

## 4. Introduction structure

### 4.1 Motivation

- standard graph backbones are often improved by residual or dense connections
- but most residual designs are same-branch carryover
- cross-branch historical interaction is less explored
- residual routing strength and sparsity may also affect how useful historical information is

### 4.2 Problem statement

- residual information is not always beneficial in the same way across operators
- cross interaction is not guaranteed to dominate all benchmarks
- the real question is:
  when does cross interaction help, and which residual routing strategy works best?

### 4.3 Contributions

Write contributions conservatively:

1. Propose a cross-residual message passing framework for graph classification.
2. Distinguish node-level and graph-level residual / cross interaction variants.
3. Introduce residual routing extensions:
   `learnable`, `top-k`, `sparse`.
4. Provide a full benchmark and targeted ablations showing operator-dependent and target-dependent behavior.
5. Show that mild sparse residual routing is often favorable on cross-oriented settings, while learnable residual remains strongest on the best graph-level residual line.

## 5. Related work structure

Organize by mechanism:

1. residual and dense graph networks
2. graph attention and operator-specific behavior
3. graph classification backbones and readout strategies
4. sparse or adaptive routing / gating ideas

Do not oversell novelty by claiming nothing related exists.
The key novelty is the combination of:

- cross-branch historical exchange
- explicit residual routing variants
- systematic analysis of when each works

## 6. Method section structure

### 6.1 Base formulation

- define graph classification setup
- define message passing backbone

### 6.2 Residual family

- `PlainGNN`
- `NodeResGNN`
- `GraphResGNN`

Clarify:

- node-level residual keeps layerwise hidden carryover
- graph-level residual injects graph-summary history

### 6.3 Cross family

- `NodeCrossGNN`
- `GraphCrossGNN`

Clarify the real distinction:

- residual: same-path historical reuse
- cross: cross-branch historical exchange

### 6.4 Gate mechanism

- explain learnable gate and fixed gate variants
- make clear that gate controls residual injection strength

### 6.5 Residual routing modes

- `learnable`
- `top-k`
- `sparse`

Describe them simply:

- `learnable`: dense residual with adaptive gate strength
- `top-k`: keep strongest residual channels
- `sparse`: shrink weak residual coefficients before injection

This subsection is important because it turns the paper from a pure architecture paper into a mechanism paper.

## 7. Experiment section structure

### 7.1 Experimental setup

Use the final benchmark facts:

- `6 datasets`
- `9 methods`
- `4 operators`
- `5 folds`
- final source version: `V3`

### 7.2 Main benchmark

Primary table:

- full benchmark comparison

Narrative:

- do not claim cross is globally best
- emphasize that the proposed family is competitive across datasets and operators
- show that graph-level residual variants are very strong

### 7.3 Cross-focused analysis

Use:

- `cross_advantage_summary`
- `cross_residual_delta_table`

Narrative:

- cross gains concentrate on selected settings, especially attention-oriented ones
- this supports an operator-dependent interpretation

### 7.4 Gate ablation

Use:

- `AIDS + GraphResGNN + GINConv`
- cross-gate ablation on cross-winning settings

Narrative:

- adaptive gating is useful
- fixed gates can be competitive but are less robust

### 7.5 Residual-mode ablation

Use the new `V3 residual summary` as a focused subsection.

Narrative:

- there is no universal best residual mode
- `learnable` remains best on the strongest graph-level residual line
- `sparse_0p05` is often best on cross-oriented settings
- `top-k` helps, but optimal ratio depends on the target

### 7.6 Parameter sensitivity

Keep this shorter.

Narrative:

- older single-fold scans provide trend evidence
- new 5-fold residual parameter sweeps provide stronger support for:
  - `topk_ratio`
  - `sparse_lambda`

## 8. Discussion section structure

This section is important for credibility.

State clearly:

- cross is not universally dominant
- residual design is target-dependent
- sparse filtering helps some cross settings, but strong graph-level residual lines may still favor learnable dense residual
- operator dependence is an important empirical finding, not a weakness to hide

## 9. Conclusion structure

Keep the conclusion disciplined:

- summarize mechanism contributions
- summarize benchmark competitiveness
- summarize the key takeaways:
  - cross helps in selected settings
  - adaptive residual routing matters
  - mild sparse residual is often effective

## 10. Supplementary material structure

### A. Full benchmark tables

- full per-dataset, per-operator, per-method results

### B. Gate ablation

- AIDS gate table
- cross + gate tables

### C. Residual-mode and residual-parameter summary

- `learnable / top-k / sparse`
- `topk_ratio / sparse_lambda`

### D. Parameter sensitivity

- older single-fold scans
- explain clearly that these are trend-oriented supplementary studies

### E. Training details

- epoch limits
- batch size
- patience
- optimizer settings
- versioning notes

## 11. Practical writing decision

Yes, the project is already ready for the middle-to-late draft stage.

What should happen next is not more large reruns, but:

1. write the paper around `V3`
2. keep the main claim disciplined
3. convert residual and cross analyses into concise tables and figures
