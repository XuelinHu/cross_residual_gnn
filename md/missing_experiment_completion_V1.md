# Missing Experiment Completion

- Updated: 2026-04-17 11:29:08
- Version: V1
- Target scope: 6 datasets, 5 folds
- Baseline target: 120, completed: 120, missing: 0
- Operator target: 450, completed: 450, missing: 0
- Total target: 570, completed: 570, missing: 0

## Per Dataset
- `PROTEINS`: baseline 20/20, operators 75/75
- `DD`: baseline 20/20, operators 75/75
- `ENZYMES`: baseline 20/20, operators 75/75
- `MUTAG`: baseline 20/20, operators 75/75
- `AIDS`: baseline 20/20, operators 75/75
- `Mutagenicity`: baseline 20/20, operators 75/75

## Baseline Coverage
- `GraphSAGEBaseline` / `SAGEConv`: 30/30
- `GINBaseline` / `GINConv`: 30/30
- `JKNetBaseline` / `GCNConv`: 30/30
- `APPNPBaseline` / `GCNConv`: 30/30

## Operator Coverage
- `PlainGNN`: GATConv 30/30, SAGEConv 30/30, GINConv 30/30
- `NodeResGNN`: GATConv 30/30, SAGEConv 30/30, GINConv 30/30
- `NodeCrossGNN`: GATConv 30/30, SAGEConv 30/30, GINConv 30/30
- `GraphResGNN`: GATConv 30/30, SAGEConv 30/30, GINConv 30/30
- `GraphCrossGNN`: GATConv 30/30, SAGEConv 30/30, GINConv 30/30

## Next Pending Jobs
- none
