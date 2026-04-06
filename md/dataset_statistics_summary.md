# Dataset Statistics Summary

This note consolidates the dataset facts used in the revised paper narrative.

## Main Biological Package

### PROTEINS
- source: PyG TUDataset
- task_type: graph classification
- split_protocol: stratified 5-fold CV + inner validation split
- role: main biological benchmark
- graphs: 1113
- classes: 2
- num_features: 3
- avg_nodes: 39.06
- avg_edges: 72.82
- note: Computed from local dataset export

### DD
- source: PyG TUDataset
- task_type: graph classification
- split_protocol: stratified 5-fold CV + inner validation split
- role: main biological benchmark
- graphs: 1178
- classes: 2
- num_features: 89
- avg_nodes: 284.32
- avg_edges: 715.66
- note: Computed from local dataset export

### ENZYMES
- source: PyG TUDataset
- task_type: graph classification
- split_protocol: stratified 5-fold CV + inner validation split
- role: main biological benchmark
- graphs: 600
- classes: 6
- num_features: 3
- avg_nodes: 32.63
- avg_edges: 62.14
- note: Computed from local dataset export

## Supplementary Robustness Package

### MUTAG
- source: PyG TUDataset
- task_type: graph classification
- split_protocol: stratified 5-fold CV + inner validation split
- role: supplementary robustness dataset
- graphs: 188
- classes: 2
- num_features: 7
- avg_nodes: 17.93
- avg_edges: 19.79
- note: Computed from local dataset export

### AIDS
- source: PyG TUDataset
- task_type: graph classification
- split_protocol: stratified 5-fold CV + inner validation split
- role: supplementary robustness dataset
- graphs: 2000
- classes: 2
- num_features: 38
- avg_nodes: 15.69
- avg_edges: 16.20
- note: Computed from local dataset export

### Mutagenicity
- source: PyG TUDataset
- task_type: graph classification
- split_protocol: stratified 5-fold CV + inner validation split
- role: supplementary robustness dataset
- graphs: 4337
- classes: 2
- num_features: 14
- avg_nodes: 29.99
- avg_edges: 30.57
- note: Computed from local dataset export
