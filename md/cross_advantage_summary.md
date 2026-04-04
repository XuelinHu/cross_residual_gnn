# Cross Advantage Summary

## Scope

This note summarizes the final topic-facing implementation results and explains where cross-residual variants help, where they do not, and why the conclusions differ from earlier exploratory runs.

Main datasets:

- `PROTEINS`
- `DD`
- `ENZYMES`

Compared models:

- `PlainGNN`
- `NodeResGNN`
- `NodeCrossGNN`
- `GraphResGNN`
- `GraphCrossGNN`

## Final Topic-Facing Results

### PROTEINS

- `GraphResGNN`: `0.71965 ± 0.04200`
- `PlainGNN`: `0.70708 ± 0.02735`
- `NodeCrossGNN`: `0.69537 ± 0.04049`
- `NodeResGNN`: `0.69265 ± 0.04573`
- `GraphCrossGNN`: `0.68546 ± 0.04849`

Takeaway:

- best overall model is `GraphResGNN`
- best cross model is `NodeCrossGNN`
- best cross model is `0.01171` below `PlainGNN`
- best cross model is `0.02427` below the best non-cross model

Interpretation:

- cross does not win on `PROTEINS` under the final protocol
- `NodeCrossGNN` is slightly better than `NodeResGNN`, but not better than `PlainGNN`
- graph-level residual propagation is more effective here than graph-level cross exchange

### DD

- `NodeResGNN`: `0.73001 ± 0.02015`
- `PlainGNN`: `0.72156 ± 0.01632`
- `NodeCrossGNN`: `0.71560 ± 0.01743`
- `GraphCrossGNN`: `0.70879 ± 0.02701`
- `GraphResGNN`: `0.70530 ± 0.04091`

Takeaway:

- best overall model is `NodeResGNN`
- best cross model is `NodeCrossGNN`
- best cross model is `0.00595` below `PlainGNN`
- best cross model is `0.01441` below the best non-cross model

Interpretation:

- the final `DD` benchmark clearly favors residual reuse over cross interaction
- `GraphCrossGNN` is slightly better than `GraphResGNN`, but both graph-level models lag behind `NodeResGNN`
- node-level residual reuse is the most stable design for large sparse protein graphs in this final protocol

### ENZYMES

- `GraphResGNN`: `0.30667 ± 0.07789`
- `GraphCrossGNN`: `0.29167 ± 0.07322`
- `NodeCrossGNN`: `0.28500 ± 0.05281`
- `NodeResGNN`: `0.24500 ± 0.07044`
- `PlainGNN`: `0.23500 ± 0.04927`

Takeaway:

- best overall model is `GraphResGNN`
- best cross model is `GraphCrossGNN`
- best cross model is `0.05667` above `PlainGNN`
- best cross model is `0.01500` below the best non-cross model

Interpretation:

- this is the dataset where cross is most visibly useful in the final suite
- both cross variants improve over `PlainGNN`
- `NodeCrossGNN` also clearly improves over `NodeResGNN`
- however, `GraphResGNN` still remains the best final model

## What Is The Real Cross Advantage

The cleanest answer is:

- cross is not the strongest default architecture in the final topic-facing benchmark
- cross is most useful as a selective improvement mechanism, not as a universal winner
- the strongest evidence for cross in the final suite comes from `ENZYMES`, where cross variants improve substantially over plain stacking

More specifically:

- `NodeCrossGNN` can outperform `NodeResGNN` on datasets where dual-branch exchange helps preserve more diverse intermediate features
- `GraphCrossGNN` can outperform `PlainGNN` on some structured classification settings
- but neither cross variant is consistently stronger than the best residual baseline

## Why This Differs From Earlier Exploratory Results

Earlier exploratory runs suggested stronger cross-residual gains on some datasets. Those gains weakened after the protocol was tightened.

The main reasons are:

- stratified outer folds replaced the earlier weaker splitting behavior
- train/validation splitting became stratified instead of relying on fragile contiguous slices
- training added `ReduceLROnPlateau` and gradient clipping
- the topic-facing benchmark removed some datasets that were easier to narrate for cross but weaker for the target venue

So the final results are more trustworthy than the earlier exploratory wins.

## Final Claim You Can Defend

The strongest defensible claim is:

- Cross-residual design is a meaningful architectural idea for biomolecular graph learning.
- It improves over plain stacking on selected datasets, especially ENZYMES.
- However, residual reuse remains the strongest default baseline in the final topic-facing benchmark.
- Therefore, the contribution should be framed as a reusable design space and analysis of information-flow mechanisms, not as a claim that cross universally outperforms residual baselines.
