# Cross Gate Ablation Plan for V2

## Goal

The repository already contains cross models with gates in the main benchmark, but it previously lacked a direct gate ablation on the strongest cross settings.

This supplementary batch adds a dedicated comparison:

- `learnable gate`
- `fixed gate = 0.0`
- `fixed gate = 0.5`
- `fixed gate = 1.0`

on the configurations where `cross` is already the best architecture in `V2`.

## Target configurations

The selected targets are the `dataset + operator + model` combinations where a cross model wins the main suite:

| dataset | model | operator |
|---|---|---|
| `AIDS` | `NodeCrossGNN` | `GATConv` |
| `DD` | `NodeCrossGNN` | `GATConv` |
| `DD` | `NodeCrossGNN` | `GINConv` |
| `ENZYMES` | `NodeCrossGNN` | `GATConv` |
| `MUTAG` | `GraphCrossGNN` | `GATConv` |
| `Mutagenicity` | `NodeCrossGNN` | `GATConv` |

Each target is evaluated with full `5-fold` runs under all four gate settings.

## Execution

Runner:

- [run_cross_gate_ablation_experiments.py](/ds1/workspace/ai/cross_residual_gnn/py/run_cross_gate_ablation_experiments.py)

Command:

```bash
python py/run_cross_gate_ablation_experiments.py --version V2 --max_workers 4
```

Total jobs:

- `6 targets × 4 gate settings × 5 folds = 120`

## Interpretation

This batch is meant to answer a more specific question than the generic gate study on `AIDS + GraphResGNN + GINConv`:

`When cross is already the winning architecture, does a learnable gate still outperform fixed cross injection strength?`

That result is the one needed if the paper wants to make a stronger argument specifically about `cross + gate`, rather than about gated residual models in general.
