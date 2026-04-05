# Parameter Sensitivity Analysis

## Scope

This note summarizes the focused parameter scans run for the cross-residual models on the topic-facing datasets:

- `PROTEINS`
- `DD`
- `ENZYMES`

Scanned models:

- `NodeCrossGNN`
- `GraphCrossGNN`

Scanned parameters on fold `0`:

- `h_layer`: `3, 4, 5`
- `drop`: `0.2, 0.3, 0.5`
- `lr`: `0.002, 0.003, 0.005`

## Main Sensitivity Signals

### PROTEINS

- `NodeCrossGNN` is most sensitive to depth. With the default `drop=0.2` and `lr=0.003`, `h_layer=3` is best (`0.69955`), while `h=4` and `h=5` both drop to `0.67713`.
- `GraphCrossGNN` benefits from lighter regularization. With `h=4`, reducing dropout from `0.3` to `0.2` lifts fold-0 accuracy from `0.69955` to `0.70404`.
- The learning-rate scan is comparatively mild on PROTEINS, especially for `GraphCrossGNN`, where `lr=0.002` and `lr=0.003` are effectively tied.

### DD

- `DD` is the most optimization-sensitive dataset in the scan.
- `NodeCrossGNN` improves clearly with a larger learning rate: `lr=0.005` reaches `0.74262`, higher than both `0.002` (`0.73418`) and the previous default `0.003` (`0.71308`).
- `GraphCrossGNN` is strongly depth-sensitive: `h=3` reaches `0.74684`, while `h=4` falls to `0.67511`.
- For both cross models on DD, weaker dropout is not always better; the fold-0 optimum appears around `drop=0.3~0.5`, not at the lowest regularization setting.

### ENZYMES

- ENZYMES shows a different failure mode: many runs achieve the best validation loss at epoch `1`, which indicates that the validation split is extremely small and noisy relative to the task difficulty.
- Fold-0 scans slightly prefer shallower and lighter-regularized cross models:
  - `NodeCrossGNN`: `h=3`, `drop=0.2`
  - `GraphCrossGNN`: `h=3`, `drop=0.2`, `lr=0.002`
- However, these fold-0 improvements do not reliably transfer to 5-fold means.

## ENZYMES Tuned Follow-up

To check whether the fold-0 sensitivity signals were real, we ran a full 5-fold follow-up with the best-looking ENZYMES settings:

- `NodeCrossGNN`: `h=3`, `drop=0.2`, `lr=0.003`
- `GraphCrossGNN`: `h=3`, `drop=0.2`, `lr=0.002`

Results:

- `NodeCrossGNN tuned`: `0.27667 ± 0.06980`
- `NodeCrossGNN original full-suite`: `0.30000 ± 0.05798`
- `GraphCrossGNN tuned`: `0.29167 ± 0.06749`
- `GraphCrossGNN original full-suite`: `0.30167 ± 0.07498`

Therefore, the tuned follow-up does **not** improve over the original ENZYMES full-suite configuration. The main paper should keep the original ENZYMES benchmark numbers and use the sensitivity analysis only as an explanation of optimization instability.

## Interpretation

- `PROTEINS`: cross models are moderately sensitive, especially to depth for `NodeCrossGNN` and dropout for `GraphCrossGNN`.
- `DD`: cross models have enough capacity to benefit from more aggressive optimization and carefully chosen depth.
- `ENZYMES`: the limiting factor is not just hyperparameter choice; the task is also constrained by noisy validation feedback under small-sample splits.

This means the parameter experiments support a nuanced claim: cross-residual models are tunable and can benefit from dataset-specific optimization, but on the smallest dataset the variance of the evaluation protocol can dominate the gains from local parameter changes.
