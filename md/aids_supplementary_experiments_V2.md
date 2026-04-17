# AIDS Supplementary Experiments for V2

## Dataset selection

`AIDS` is the strongest dataset in the current `V2` full suite.

- best result: `GraphResGNN + GINConv = 0.9410`
- operator-wise best on `AIDS`
  - `GCNConv`: `GraphResGNN = 0.9090`
  - `GATConv`: `NodeCrossGNN = 0.9200`
  - `GINConv`: `GraphResGNN = 0.9410`
  - `SAGEConv`: `GraphResGNN = 0.9145`

This makes `AIDS` the cleanest candidate for supplementary parameter analysis and targeted ablation.

## Existing ablations already covered by the main suite

At fixed `AIDS + GINConv`, the current `V2` full run already gives a complete architecture comparison:

| model | mean acc | std |
|---|---:|---:|
| `GraphResGNN` | `0.9410` | `0.0398` |
| `NodeCrossGNN` | `0.9305` | `0.0173` |
| `GINBaseline` | `0.9145` | `0.0415` |
| `NodeResGNN` | `0.8955` | `0.0487` |
| `PlainGNN` | `0.8935` | `0.0471` |
| `GraphCrossGNN` | `0.8865` | `0.0421` |

So the paper-level model ablation table on the best dataset is already available and does not need rerunning.

## Missing supplementary experiments

The remaining gaps are local sensitivity and gate-specific ablation around the best configuration:

### 1. Parameter sensitivity on the best line

Target line:

- dataset: `AIDS`
- model: `GraphResGNN`
- operator: `GINConv`
- base config: `lr=0.005`, `weight_decay=1e-4`, `drop=0.5`, `dim=64`, `h_layer=4`, `batch_size=256`, `gate_init=0.8`

Recommended sweeps:

- `lr`: `0.001, 0.002, 0.003, 0.005`
- `drop`: `0.2, 0.3, 0.5, 0.6`
- `weight_decay`: `1e-5, 5e-5, 1e-4, 5e-4`
- `h_layer`: `2, 3, 4, 5`
- `dim`: `32, 64, 128`
- `gate_init`: `0.2, 0.5, 0.8, 0.95`

Execution policy:

- first run a `fold-0` scan to locate sensitive dimensions
- then promote the top settings to a full `5-fold` confirmation round

### 2. Learnable-gate ablation

The current paper claim now includes learnable gates, but the repository had no direct control experiment for that claim. The minimum supplementary ablation is:

- `gate_mode=learnable`
- `gate_mode=fixed, fixed_gate_value=0.0`
- `gate_mode=fixed, fixed_gate_value=0.5`
- `gate_mode=fixed, fixed_gate_value=1.0`

Target line:

- dataset: `AIDS`
- model: `GraphResGNN`
- operator: `GINConv`
- full `5-fold`

Interpretation:

- `0.0`: remove graph residual contribution entirely
- `0.5`: keep a weak static residual path
- `1.0`: force full-strength residual injection
- `learnable`: check whether adaptive gating is actually better than any fixed coefficient

## Execution status

The script for these supplementary experiments is:

- [run_aids_supplementary_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/run_aids_supplementary_experiments.py)

Suggested commands:

```bash
python py/run_aids_supplementary_experiments.py --which sensitivity --version V2 --max_workers 4
python py/run_aids_supplementary_experiments.py --which gate_ablation --version V2 --max_workers 4
```

All outputs stay inside the existing versioned `V2` tree and are tagged with `aids_supp_*`, so they can be separated from the main full-suite benchmark logs later.
