# V3 Residual Summary

This summary merges the completed `V3` residual studies:

- `residual_mode_*`
- `residual_param_*`

All numbers below are 5-fold mean test accuracy after deduplicating repeated reruns and keeping the latest result for the same `(target, setting, fold)`.

## 1. Full setting table

| Target | learnable | topk_0p25 | topk_0p5 | topk_0p75 | sparse_0p02 | sparse_0p05 | sparse_0p1 | best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `AIDS + GraphResGNN + GINConv` | `0.9500` | `0.9075` | `0.9330` | `0.9240` | `0.9470` | `0.9250` | `0.8315` | `learnable` |
| `PROTEINS + GraphResGNN + SAGEConv` | `0.7178` | `0.7241` | `0.7151` | `0.7214` | `0.7160` | `0.7259` | `0.7088` | `sparse_0p05` |
| `AIDS + NodeCrossGNN + GATConv` | `0.8970` | `0.9160` | `0.8960` | `0.8960` | `0.9060` | `0.9180` | `0.9025` | `sparse_0p05` |
| `DD + NodeCrossGNN + GATConv` | `0.7080` | `0.7062` | `0.7029` | `0.7164` | `0.7147` | `0.7181` | `0.7105` | `sparse_0p05` |

## 2. Family winner table

This table compares the best point inside each family.

| Target | best learnable | best top-k | best sparse | overall winner |
|---|---|---|---|---|
| `AIDS + GraphResGNN + GINConv` | `learnable = 0.9500` | `topk_0p5 = 0.9330` | `sparse_0p02 = 0.9470` | `learnable` |
| `PROTEINS + GraphResGNN + SAGEConv` | `learnable = 0.7178` | `topk_0p25 = 0.7241` | `sparse_0p05 = 0.7259` | `sparse` |
| `AIDS + NodeCrossGNN + GATConv` | `learnable = 0.8970` | `topk_0p25 = 0.9160` | `sparse_0p05 = 0.9180` | `sparse` |
| `DD + NodeCrossGNN + GATConv` | `learnable = 0.7080` | `topk_0p75 = 0.7164` | `sparse_0p05 = 0.7181` | `sparse` |

## 3. Main takeaways

- `sparse` is the strongest family on `3 / 4` representative targets.
- `sparse_0p05` is the most consistently strong sparse setting.
- `learnable` remains the best choice on the strongest residual line:
  - `AIDS + GraphResGNN + GINConv`
- `top-k` can help, but its best ratio is target-dependent:
  - `0.5` is best for `AIDS + GraphResGNN + GINConv`
  - `0.25` is best for `PROTEINS + GraphResGNN + SAGEConv`
  - `0.75` is best for `DD + NodeCrossGNN + GATConv`
- aggressive sparsity is harmful on the strong residual line:
  - `AIDS + GraphResGNN + GINConv + sparse_0p1 = 0.8315`

## 4. Paper-facing interpretation

The cleanest claim supported by the current results is:

- `learnable residual` is still the strongest default choice for the strongest graph-level residual line
- but `sparse residual`, especially `sparse_0p05`, is a competitive and often stronger alternative on cross-oriented settings

So the residual story should not be written as:

- one universal winner across all settings

It should be written as:

- residual design is target-dependent
- graph-level strong residual lines favor adaptive learnable residual strength
- cross-heavy settings often benefit from mild sparse residual filtering
