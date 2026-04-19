# Final Project Index

Use this file as the single entry point for the cleaned final repository state.

## Final version

- active experiment version: `V3`
- main final summary: [EXPERIMENT_INDEX.md](/ds1/workspace/ai/cross_residual_gnn/md/EXPERIMENT_INDEX.md)
- script summary: [SCRIPT_INDEX.md](/ds1/workspace/ai/cross_residual_gnn/md/SCRIPT_INDEX.md)
- residual summary: [V3_residual_summary.md](/ds1/workspace/ai/cross_residual_gnn/md/V3_residual_summary.md)
- paper outline: [paper_final_draft_outline.md](/ds1/workspace/ai/cross_residual_gnn/md/paper_final_draft_outline.md)

## Final workflow

The recommended final workflow entry is:

- [run_final_v3_pipeline.py](/ds1/workspace/ai/cross_residual_gnn/py/run_final_v3_pipeline.py)

Typical usage:

```bash
python py/run_final_v3_pipeline.py
python py/run_final_v3_pipeline.py --steps consolidate summarize reports figures
```

## Final source files

- training entry: [graph_classify_v3.py](/ds1/workspace/ai/cross_residual_gnn/geomatric/graph_classify_v3.py)
- version path config: [experiment_paths.py](/ds1/workspace/ai/cross_residual_gnn/geomatric/experiment_paths.py)

## Final experiment assets

- benchmark and supplementary outputs: `logs/V3`
- structured records: `records/V3`
- tensorboard snapshots: `runs/V3`

## Cleanup rule

The repository now keeps:

- final writing documents
- final experiment index documents
- final reproducible scripts
- final `V3` logs and result records

The repository no longer keeps:

- session notes
- temporary missing-job checklists
- temporary todo files
- phased consolidation notes that were only useful during reruns
