# Missing Experiment Completion

- Updated: 2026-04-16 00:05:18
- Target scope: 6 datasets, 5 folds
- Baseline target: 120, completed: 120, missing: 0
- Operator target: 600, completed: 600, missing: 0
- Total target: 720, completed: 720, missing: 0

## Per Dataset
- `PROTEINS`: baseline 20/20, operators 100/100
- `DD`: baseline 20/20, operators 100/100
- `ENZYMES`: baseline 20/20, operators 100/100
- `MUTAG`: baseline 20/20, operators 100/100
- `AIDS`: baseline 20/20, operators 100/100
- `Mutagenicity`: baseline 20/20, operators 100/100

## Baseline Coverage
- `GraphSAGEBaseline` / `SAGEConv`: 30/30
- `GINBaseline` / `GINConv`: 30/30
- `JKNetBaseline` / `GCNConv`: 30/30
- `APPNPBaseline` / `GCNConv`: 30/30

## Operator Coverage
- `PlainGNN`: GATConv 30/30, TransformerConv 30/30, SAGEConv 30/30, GINConv 30/30
- `NodeResGNN`: GATConv 30/30, TransformerConv 30/30, SAGEConv 30/30, GINConv 30/30
- `NodeCrossGNN`: GATConv 30/30, TransformerConv 30/30, SAGEConv 30/30, GINConv 30/30
- `GraphResGNN`: GATConv 30/30, TransformerConv 30/30, SAGEConv 30/30, GINConv 30/30
- `GraphCrossGNN`: GATConv 30/30, TransformerConv 30/30, SAGEConv 30/30, GINConv 30/30

## Next Pending Jobs
- none
