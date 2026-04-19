# Experiment Index

## Final status

The active paper-facing experiment source is now `V3`.

`V3` already consolidates:

- full main benchmark from previous `V2`
- supplementary gate ablations
- cross-gate ablations
- residual-mode ablations
- residual parameter sweeps

Use `V3` as the single source of truth for paper writing.

## Final experiment matrix in V3

| block | scope | status |
|---|---|---:|
| Main benchmark | `6 datasets x 9 methods x 4 operators x 5 folds` | `720/720` |
| AIDS gate ablation | `AIDS + GraphResGNN + GINConv`, `4 gate settings x 5 folds` | `20/20` |
| Cross + gate ablation | `6 targets x 4 gate settings x 5 folds` | `120/120` |
| Residual-mode ablation | `4 targets x 5 residual settings x 5 folds` | `100/100` |
| Residual parameter sweeps | `4 targets x 2 new settings x 5 folds` | `40/40` |

Current unique completed jobs in `V3`:

- main benchmark: `720`
- `aids_supp_*`: `43`
- `cross_gate_*`: `120`
- `residual_mode_*`: `100`
- `residual_param_*`: `40`

Note:

- `logs/V3` still contains some historical duplicates from earlier failed or restarted runs
- for analysis and writing, use unique experiment tags rather than raw file count

## Main documents to use

### For experiment scope and status

- [current_experiment_inventory_and_gaps.md](/ds1/workspace/ai/cross_residual_gnn/md/current_experiment_inventory_and_gaps.md)
- [v3_consolidation_and_extension_plan.md](/ds1/workspace/ai/cross_residual_gnn/md/v3_consolidation_and_extension_plan.md)

### For benchmark comparison and ranking

- [fold_stats_main_benchmark.md](/ds1/workspace/ai/cross_residual_gnn/md/fold_stats_main_benchmark.md)
- [fold_stats_main_benchmark_cn.md](/ds1/workspace/ai/cross_residual_gnn/md/fold_stats_main_benchmark_cn.md)
- [fold_stats_all_models_ranked_cn.md](/ds1/workspace/ai/cross_residual_gnn/md/fold_stats_all_models_ranked_cn.md)
- [all_ablation_analysis.md](/ds1/workspace/ai/cross_residual_gnn/md/all_ablation_analysis.md)

### For cross-focused narrative

- [cross_advantage_summary.md](/ds1/workspace/ai/cross_residual_gnn/md/cross_advantage_summary.md)
- [cross_residual_delta_table_cn.md](/ds1/workspace/ai/cross_residual_gnn/md/cross_residual_delta_table_cn.md)

### For supplementary experiments

- [aids_supplementary_experiments_V2.md](/ds1/workspace/ai/cross_residual_gnn/md/aids_supplementary_experiments_V2.md)
- [cross_gate_ablation_plan_V2.md](/ds1/workspace/ai/cross_residual_gnn/md/cross_gate_ablation_plan_V2.md)
- [parameter_sensitivity_analysis.md](/ds1/workspace/ai/cross_residual_gnn/md/parameter_sensitivity_analysis.md)
- [sensitivity_summary.md](/ds1/workspace/ai/cross_residual_gnn/md/sensitivity_summary.md)

### For tables and appendix artifacts

- [all_exp_tables.tex](/ds1/workspace/ai/cross_residual_gnn/md/all_exp_tables.tex)
- [all_exp_tables_appendix.tex](/ds1/workspace/ai/cross_residual_gnn/md/all_exp_tables_appendix.tex)
- [dataset_statistics_tables.tex](/ds1/workspace/ai/cross_residual_gnn/md/dataset_statistics_tables.tex)
- [topic_exp_tables.tex](/ds1/workspace/ai/cross_residual_gnn/md/topic_exp_tables.tex)

## Suggested paper writing order

1. Use `V3` as the only experiment version in the paper.
2. Use the main benchmark table first.
3. Add cross-focused analysis as a mechanism discussion, not as a universal-best claim.
4. Put gate ablation, cross-gate ablation, and residual-mode studies into supplementary or ablation sections.
5. Treat old single-fold parameter studies as supporting sensitivity evidence, not as the main quantitative claim.

## Practical conclusion

The experiment side is already sufficient to begin the middle draft.

What is still worth doing later is not more large reruns, but:

- summarizing `residual_mode_*` and `residual_param_*` into compact tables
- cleaning duplicate runtime files if you want a tidier repository
