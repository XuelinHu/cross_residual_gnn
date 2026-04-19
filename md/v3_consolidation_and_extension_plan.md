# V3 Consolidation And Extension Plan

## Version cleanup

Current version naming after the recent repository changes:

- `achivement_V0`: archived historical materials that were previously stored under `achivement_V2`
- `logs/V1`, `records/V1`, `runs/V1`: first formal versioned runtime outputs
- `logs/V2`, `records/V2`, `runs/V2`: second formal versioned runtime outputs and supplementary reruns

Recommended final consolidation target:

- keep `achivement_V0` as frozen legacy archive only
- keep `V1` and `V2` as reproducibility sources
- aggregate all accepted formal results into a new active version `V3`

`V3` should become the only version used for any new paper-facing tables, supplementary ablations, and final reruns.

## Ablations already completed

### 1. Main architecture ablation

Already covered in the full suite:

- `PlainGNN`
- `NodeResGNN`
- `NodeCrossGNN`
- `GraphResGNN`
- `GraphCrossGNN`
- external baselines:
  - `GraphSAGEBaseline`
  - `GINBaseline`
  - `JKNetBaseline`
  - `APPNPBaseline`

This already covers:

- plain vs residual vs cross
- node-level vs graph-level
- custom family vs external baselines

Primary summary artifact:

- [all_ablation_analysis.md](/ds1/workspace/ai/cross_residual_gnn/md/all_ablation_analysis.md)

### 2. Depth ablation

Already supported and executed:

- `NodeResGNN` depth sweep through `h_layer`

### 3. Learnable gate ablation on the best dataset line

Already completed on:

- `AIDS + GraphResGNN + GINConv`

Compared settings:

- `learnable gate`
- `fixed gate = 0.0`
- `fixed gate = 0.5`
- `fixed gate = 1.0`

Primary artifact:

- [aids_supplementary_experiments_V2.md](/ds1/workspace/ai/cross_residual_gnn/md/aids_supplementary_experiments_V2.md)

### 4. Cross + gate ablation

Already completed on cross-winning settings:

- `AIDS + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GINConv`
- `ENZYMES + NodeCrossGNN + GATConv`
- `MUTAG + GraphCrossGNN + GATConv`
- `Mutagenicity + NodeCrossGNN + GATConv`

Compared settings:

- `learnable gate`
- `fixed gate = 0.0`
- `fixed gate = 0.5`
- `fixed gate = 1.0`

Primary artifact:

- [cross_gate_ablation_plan_V2.md](/ds1/workspace/ai/cross_residual_gnn/md/cross_gate_ablation_plan_V2.md)

## Parameter experiments already completed

### 1. Original sensitivity scans

Already scanned:

- `lr`
- `drop`
- `h_layer`

Primary artifacts:

- [parameter_sensitivity_analysis.md](/ds1/workspace/ai/cross_residual_gnn/md/parameter_sensitivity_analysis.md)
- [sensitivity_summary.md](/ds1/workspace/ai/cross_residual_gnn/md/sensitivity_summary.md)

### 2. AIDS supplementary parameter scans

Already scanned on `AIDS + GraphResGNN + GINConv`:

- `lr`
- `drop`
- `weight_decay`
- `h_layer`
- `dim`
- `gate_init`

Primary artifact:

- [aids_supplementary_experiments_V2.md](/ds1/workspace/ai/cross_residual_gnn/md/aids_supplementary_experiments_V2.md)

## Gaps not yet covered

Not yet systematically covered:

- `batch_size`
- scheduler settings: `lr_factor`, `lr_patience`
- `grad_clip`
- residual-mechanism-specific parameters
  - `topk_ratio`
  - `sparse_lambda`

## New residual mechanisms to add

The current repository already supports:

- `learnable residual` via learnable gate strength

The next extension adds:

### 1. `top-k residual`

Definition in the implementation:

- keep only the largest-magnitude residual channels along the last feature dimension
- mask out the remaining residual channels before residual injection

Controlling parameter:

- `topk_ratio`

### 2. `sparse residual`

Definition in the implementation:

- apply `softshrink` to the residual tensor before injection
- small residual coefficients are driven to zero, forming a differentiable sparse residual path

Controlling parameter:

- `sparse_lambda`

## Implementation status

The main training entry has now been extended with:

- `--residual_mode learnable|topk|sparse`
- `--topk_ratio`
- `--sparse_lambda`

The new modes currently affect:

- `NodeResGNN`
- `NodeCrossGNN`
- graph-level residual injections inside `PlainBlock`
- therefore also `GraphResGNN`
- and graph-hidden exchange inside `GraphCrossGNN`

## Next execution plan

### Step 1. Version consolidation

- create a clean `V3` manifest entry
- define `V3` as the active paper-facing version
- write a migration note describing how `V1` and `V2` feed into `V3`

### Step 2. Residual-mode ablation

Run a dedicated residual-mechanism comparison with:

- `learnable`
- `topk`
- `sparse`

Suggested targets:

- strongest residual line:
  - `AIDS + GraphResGNN + GINConv`
- strongest topic-facing residual line:
  - `PROTEINS + GraphResGNN + SAGEConv`
- strongest cross line:
  - `AIDS + NodeCrossGNN + GATConv`
- hardest large sparse case:
  - `DD + NodeCrossGNN + GATConv`

### Step 3. Residual-mode parameter sweeps

Add:

- `topk_ratio`: e.g. `0.25, 0.5, 0.75`
- `sparse_lambda`: e.g. `0.02, 0.05, 0.1`

## What I will do next after confirmation

1. Add `V3` to the version manifest and switch future paper-facing runs to `V3`
2. Add a dedicated residual-mode ablation runner
3. Generate a compact report that separates:
   - already completed ablations
   - already completed parameter sweeps
   - newly added residual-mode experiments
