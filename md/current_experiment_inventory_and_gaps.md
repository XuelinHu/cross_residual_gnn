# Current Experiment Inventory And Gaps

## 1. Current directory and version status

- `achivement_V0`
  - historical archive only
  - no longer part of the active runtime versioning
- `logs/V1`, `records/V1`
  - older formal version
  - includes main results and early sensitivity scans
- `logs/V2`, `records/V2`
  - current complete main benchmark source
  - also contains supplementary gate ablations and cross-gate ablations
- `logs/V3`, `records/V3`
  - new paper-facing extension version
  - currently contains residual-mode ablations only

Current problem:

- `V2` is the only version that currently contains the complete full benchmark matrix
- `V3` has been created, but it has not yet absorbed the accepted `V2` benchmark outputs
- so the repo now has a split state:
  - full benchmark lives in `V2`
  - new residual-mode ablations live in `V3`

## 2. What is complete right now

### 2.1 Main benchmark matrix

The complete main benchmark is in `V2`.

Coverage:

- datasets: `6`
  - `AIDS`
  - `DD`
  - `ENZYMES`
  - `MUTAG`
  - `Mutagenicity`
  - `PROTEINS`
- methods: `9`
  - `PlainGNN`
  - `NodeResGNN`
  - `NodeCrossGNN`
  - `GraphResGNN`
  - `GraphCrossGNN`
  - `GraphSAGEBaseline`
  - `GINBaseline`
  - `JKNetBaseline`
  - `APPNPBaseline`
- operators: `4`
  - `GCNConv`
  - `GATConv`
  - `GINConv`
  - `SAGEConv`
- folds: `5`

Main benchmark size:

- `6 datasets x 9 methods x 4 operators x 5 folds = 720`
- current status in `V2`: `720 / 720`

### 2.2 Supplementary ablations already complete

#### A. Main architecture ablation

Already covered by the main benchmark matrix:

- plain vs residual vs cross
- node-level vs graph-level
- internal methods vs external baselines

#### B. AIDS gate ablation

Completed in `V2`.

Target:

- `AIDS + GraphResGNN + GINConv`

Settings:

- `learnable gate`
- `fixed gate = 0.0`
- `fixed gate = 0.5`
- `fixed gate = 1.0`

Fold coverage:

- all four settings are complete on `5/5` folds

#### C. Cross + gate ablation

Completed in `V2`.

Targets:

- `AIDS + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GINConv`
- `ENZYMES + NodeCrossGNN + GATConv`
- `MUTAG + GraphCrossGNN + GATConv`
- `Mutagenicity + NodeCrossGNN + GATConv`

Settings:

- `learnable gate`
- `fixed gate = 0.0`
- `fixed gate = 0.5`
- `fixed gate = 1.0`

Coverage:

- `6 targets x 4 settings x 5 folds = 120`
- current status in `V2`: `120 / 120`

#### D. Residual-mode ablation

Completed in `V3`.

Targets:

- `AIDS + GraphResGNN + GINConv`
- `PROTEINS + GraphResGNN + SAGEConv`
- `AIDS + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GATConv`

Settings:

- `learnable`
- `topk_0p25`
- `topk_0p5`
- `sparse_0p02`
- `sparse_0p05`

Coverage:

- `4 targets x 5 settings x 5 folds = 100`
- current status in `V3`: `100 / 100`

Note:

- `logs/V3` currently contains `103` files because 3 earlier runs are duplicates
- the unique completed job count is `100`

## 3. Parameter experiments already done

### 3.1 Original sensitivity scans

Stored in `V1`.

Targets:

- datasets:
  - `PROTEINS`
  - `DD`
  - `ENZYMES`
- methods:
  - `NodeCrossGNN`
  - `GraphCrossGNN`
- operator:
  - `GCNConv`

Scanned parameters:

- `lr`
- `drop`
- `h_layer`

Important limitation:

- these runs were executed only on `fold 0`
- they are useful for trend inspection
- they are not a full 5-fold paper-grade sensitivity benchmark

### 3.2 AIDS supplementary parameter scans

Stored in `V2`.

Target:

- `AIDS + GraphResGNN + GINConv`

Scanned parameters:

- `lr`
- `drop`
- `weight_decay`
- `h_layer`
- `dim`
- `gate_init`

Important limitation:

- these runs are also only `fold 0`
- they are useful as local sensitivity evidence
- they should not be treated as a full benchmark-level parameter study

## 4. Current experiment matrix

### 4.1 Main benchmark matrix

| category | status | location |
|---|---:|---|
| 6 datasets | complete | `V2` |
| 9 methods | complete | `V2` |
| 4 operators | complete | `V2` |
| 5 folds | complete | `V2` |
| total 720 jobs | `720/720` | `logs/V2` |

### 4.2 Gate ablation matrix

| category | status | location |
|---|---:|---|
| AIDS gate ablation | `20/20` | `V2` |
| cross + gate ablation | `120/120` | `V2` |

### 4.3 Residual-mode matrix

| category | status | location |
|---|---:|---|
| learnable vs top-k vs sparse | `100/100` | `V3` |

### 4.4 Parameter sensitivity matrix

| category | status | limitation | location |
|---|---:|---|---|
| original cross sensitivity | complete | fold 0 only | `V1` |
| AIDS supplementary sensitivity | complete | fold 0 only | `V2` |

## 5. What is still missing

These are the real gaps, ordered by priority.

### Gap 1. V3 is not yet a true final consolidated version

Right now:

- `V2` has the complete full benchmark
- `V3` has only the new residual-mode ablation

So if you want a true final paper-facing version, `V3` still needs:

- accepted main benchmark results from `V2`
- accepted supplementary gate ablations from `V2`
- accepted cross-gate ablations from `V2`
- a clean manifest/index that points to `V3` as the single source of truth

This is a version consolidation gap, not a modeling gap.

### Gap 2. Parameter studies are not yet full 5-fold studies

Both parameter experiment families currently use single-fold evidence.

Still missing if you want paper-grade parameter tables:

- 5-fold `topk_ratio` sweep
- 5-fold `sparse_lambda` sweep
- possibly 5-fold `gate_init` sweep on the final best line

### Gap 3. Residual-mode coverage is still selective

Current residual-mode ablation covers only 4 representative targets.

Still missing if you want stronger claims:

- residual-mode comparison on more `GraphCrossGNN` settings
- residual-mode comparison on a `GCNConv` line
- residual-mode comparison on a `GINConv` cross-winning line beyond `DD`

This is not strictly required for a first paper draft, but it matters if you want to claim broad robustness.

### Gap 4. No explicit top-k or sparse residual parameter sweep report yet

You already ran:

- `topk_0p25`
- `topk_0p5`
- `sparse_0p02`
- `sparse_0p05`

But there is not yet a dedicated summarized report saying:

- which residual mode wins per target
- whether top-k or sparse beats learnable
- how sensitive the result is to `topk_ratio` and `sparse_lambda`

This is a reporting gap rather than a training gap.

## 6. Recommended next steps

### Must do

1. Consolidate accepted `V2` outputs into `V3`
2. Generate a single `V3` inventory report
3. Summarize the residual-mode ablation results into one table

### Should do

1. Add a small 5-fold parameter sweep for:
   - `topk_ratio`
   - `sparse_lambda`
2. Decide whether the final paper story centers on:
   - `cross + gate`
   - or `residual mode design`

### Optional

1. Extend residual-mode ablation to more datasets and operators
2. Clean duplicate files in `logs/V3`
