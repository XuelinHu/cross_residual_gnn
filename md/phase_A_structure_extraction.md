# Phase A: Code Structure Extraction
## Analysis of Geomatraic/ Implementation

**Date**: 2025-02-09
**Purpose**: Understand ECR-GNN implementation for writing Proposed Model chapter

---

## 1. Training Entry Points

### Main Training Function
**Location**: `graph_classify_v2.py:382-476`

```python
def train_model(start_index):
    train_loader, test_loader, dataset = load_data(start_index)
    # Model instantiation based on args.gname
    # Optimizer: Adam (lr=args.lr, weight_decay=0.01)
    # Loss: CrossEntropyLoss
    # TensorBoard logging for loss, accuracy, weights, gradients
```

### Training Modes
1. **`true_train()`** (line 542): Full 5-fold cross-validation, depths 1-5, all models
2. **`pre_check_train()`** (line 597): Pre-check training with limited epochs
3. **`debug()`** (line 508): Quick debug run (1 epoch, MUTAG only)
4. **Main Entry** (line 810-811): Currently calls `pre_check_train()`

---

## 2. Model Architecture Hierarchy

### 2.1 Base Model: BlockGNN
**Location**: `graph_classify_v2.py:204-239`

**Architecture**:
```python
class BlockGNN(torch.nn.Module):
    - to_hidden: Operator (input_dim → hidden_dim)
    - sequence: List of operators (hidden_dim → hidden_dim), length = hidden_layer
    - lin: Classifier (hidden_dim → num_classes)
```

**Forward Pass**:
1. `x = to_hidden(x, edge_index)`  # Initial projection
2. Loop through `sequence`:
   - Optional: Add `graph_hidden` to node features (line 224-227)
   - `x = F.relu(model(x, edge_index))`
   - `x = F.dropout(x, p=args.drop)`
3. `global_mean = global_mean_pool(x, batch)`  # Readout
4. Optional: `global_mean = global_mean + graph_hidden` (line 234)
5. `y = lin(F.dropout(global_mean, p=args.drop))`  # Classification

**Characteristics**:
- Sequential layer stacking
- No residual connections
- Single-branch architecture
- `res_graph` flag only controls adding external `graph_hidden`

---

### 2.2 Single-Branch Residual: ResBlockGnn
**Location**: `graph_classify_v2.py:242-276`

**Key Innovation** - **Intra-Branch Residual Connection** (line 265):
```python
x_cur = F.relu(m(x_cur + x_pre, edge_index))
```

**Forward Pass**:
1. `x_cur = F.relu(to_hidden(x, edge_index))`
2. Initialize: `x_pre = torch.zeros(x_cur.shape)`
3. Loop through `sequence`:
   - `x_temp = x_cur`  # Save current state
   - Optional: Add `graph_hidden` to node features
   - **`x_cur = F.relu(m(x_cur + x_pre, edge_index))`**  # Residual connection!
   - `x_pre = x_temp`  # Update history
4. Readout + optional graph_hidden addition
5. Classification

**Residual Mechanism**:
- Maintains `x_pre` (previous layer representation)
- Adds previous layer to current layer: `x_cur + x_pre`
- **Within single branch only**

---

### 2.3 Node-Level Cross-Residual: CrossBlockGnn ⭐
**Location**: `graph_classify_v2.py:279-321`

**Core Innovation** - **Cross-Branch Residual Connections**:

**Architecture**:
```python
class CrossBlockGnn(torch.nn.Module):
    - to_hidden_1: Operator for branch 1
    - to_hidden_2: Operator for branch 2
    - sequence: 2 × hidden_layer operators (alternating between branches)
    - lin: Classifier
```

**Forward Pass**:
```python
# Initialize two branches
x_cur_1 = F.relu(to_hidden_1(x, edge_index))
x_cur_2 = F.relu(to_hidden_2(x, edge_index))
x_pre_1 = torch.zeros(x_cur_1.shape)
x_pre_2 = torch.zeros(x_cur_2.shape)

# Loop through layers (2 layers per iteration)
while i < len(sequence):
    x_temp_1 = x_cur_1
    x_temp_2 = x_cur_2

    # Branch 1 uses history from branch 2!
    x_cur_1 = F.relu(sequence[i](x_cur_1 + x_pre_2, edge_index))
    x_cur_1 = F.dropout(x_cur_1, p=args.drop, training=self.training)

    # Branch 2 uses history from branch 1!
    x_cur_2 = F.relu(sequence[i+1](x_cur_2 + x_pre_1, edge_index))
    x_cur_2 = F.dropout(x_cur_2, p=args.drop, training=self.training)

    # Swap history
    x_pre_1 = x_temp_1
    x_pre_2 = x_temp_2
    i += 2

# Merge branches
x_cur = x_cur_1 + x_cur_2
global_mean = global_mean_pool(x_cur, batch)
# ... classification
```

**Cross-Residual Mechanism**:
- Branch 1: `x_cur_1 = layer(x_cur_1 + x_pre_2)` ← Uses **branch 2's history**
- Branch 2: `x_cur_2 = layer(x_cur_2 + x_pre_1)` ← Uses **branch 1's history**
- Information **exchanges** between branches at each layer
- **Node-level cross-residual connections**

---

### 2.4 Graph-Level Sequential: GraphBlockGnn
**Location**: `graph_classify_v2.py:324-335`

**Architecture**:
```python
class GraphBlockGnn(torch.nn.Module):
    - inner_model1: BlockGNN (res_graph=False)
    - inner_model2: BlockGNN (res_graph=False)
    - lin: Classifier
```

**Forward Pass**:
```python
_, g = self.inner_model1(x, edge_index, batch)
y, g = self.inner_model2(x, edge_index, batch, g)  # Pass graph_hidden
y = self.lin(g)
```

**Graph-Level Mechanism**:
- Sequential processing (not parallel!)
- `inner_model2` receives `graph_hidden` from `inner_model1`
- Graph-level representation passed between branches
- **No cross-branch communication (only sequential feedforward)**

---

### 2.5 Graph-Level Residual: ResGraphBlockGnn
**Location**: `graph_classify_v2.py:338-351`

**Architecture**:
```python
class ResGraphBlockGnn(torch.nn.Module):
    - inner_model1: BlockGNN (res_graph=True)
    - inner_model2: BlockGNN (res_graph=True)
    - inner_model3: BlockGNN (res_graph=True)
    - lin: Classifier
```

**Forward Pass**:
```python
_, g = self.inner_model1(x, edge_index, batch)
y, g = self.inner_model2(x, edge_index, batch, g)  # Add previous graph_hidden
y, g = self.inner_model3(x, edge_index, batch, g)  # Add previous graph_hidden
y = self.lin(g)
```

**Graph-Level Residual**:
- Sequential processing through 3 branches
- Each branch receives and adds `graph_hidden` from previous branch
- **Graph-level residual connection** (not cross-branch!)

---

### 2.6 Graph-Level Cross-Residual: CrossGraphBlockGnn ⭐
**Location**: `graph_classify_v2.py:354-379`

**Core Innovation** - **Graph-Level Cross-Branch Residual**:

**Architecture**:
```python
class CrossGraphBlockGnn(torch.nn.Module):
    - sequence: 4 BlockGNN modules (res_graph=False)
    - lin: Classifier
```

**Forward Pass**:
```python
g_1, g_2 = None, None
i = 0
while i < len(sequence):
    # BlockGNN 1 & 2 process in parallel
    _, global_mean_1 = sequence[i](x, edge_index, batch, g_1)
    _, global_mean_2 = sequence[i+1](x, edge_index, batch, g_2)

    # Cross graph-level representations!
    g_1 = global_mean_2  # Branch 1 gets graph_hidden from branch 2
    g_2 = global_mean_1  # Branch 2 gets graph_hidden from branch 1

    i += 2

global_mean = g_1 + g_2  # Merge
y = F.dropout(global_mean, p=args.drop, training=self.training)
y = self.lin(global_mean)
```

**Graph-Level Cross-Residual Mechanism**:
- 4 BlockGNN modules processed in **2 parallel pairs**
- Pair 1: `sequence[0]` and `sequence[1]`
  - Branch 1 receives `g_1` (which was `g_2` from previous iteration)
  - Branch 2 receives `g_2` (which was `g_1` from previous iteration)
- **Cross-branch exchange at graph-level**
- **Graph-level residual propagation**

---

## 3. Operator Instantiation

### Supported Operators
**Location**: `graph_classify_v2.py:191-201`

```python
def get_block_model(model_name, feature, hidden_channels):
    if model_name == 'GCNConv':
        return GCNConv(feature, hidden_channels)
    elif model_name == 'GATConv':
        return GATConv(feature, hidden_channels)
    elif model_name == 'TransformerConv':
        return TransformerConv(feature, hidden_channels)
    else:
        print(f'f model not found,model_name:{model_name}')
        return None
```

**Operators Used**:
- **GCNConv**: Graph Convolutional Network (low-pass filtering)
- **GATConv**: Graph Attention Network (attention-weighted aggregation)
- **TransformerConv**: Graph Transformer (self-attention over nodes)

**NOT Currently Implemented** (mentioned in comments but not in code):
- MixHopConv (line 10 comment)
- DirGNNConv (line 10 comment)
- AntiSymmetricConv (line 10 comment)
- GraphSAGE
- GIN

**Framework Nature**:
- GCN/GAT/Transformer are **instances** of operators, not the framework itself
- The framework treats operators as **modular, pluggable components**
- `model_name` parameter specifies which operator to use
- **Multi-operator architecture** (can theoretically support any PyTorch Geometric operator)

---

## 4. Readout and Classification Mechanism

### Readout Function
**Location**: Used in all models

```python
from torch_geometric.nn import global_mean_pool

global_mean = global_mean_pool(x, batch)
```

- **Global mean pooling**: Aggregates node representations to graph-level
- Simple and effective
- No hierarchical pooling (unlike DiffPool, SAGPool)

### Classification
```python
y = F.dropout(global_mean, p=args.drop, training=self.training)
y = self.lin(global_mean)  # Linear classifier
```

- Single linear layer
- Dropout regularization
- Cross-entropy loss (line 404)

---

## 5. Data Loading and Configuration

### Datasets
**Location**: `graph_classify_v2.py:167-188`

```python
from torch_geometric.datasets import TUDataset

def load_data(start_index=0):
    dataset = TUDataset(root=..., name=args.ds)  # args.ds: MUTAG, DD, MSRC_9, AIDS
    # 5-fold cross-validation split
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, dataset
```

**Datasets**:
- **MUTAG**: 188 graphs, molecular toxicity prediction
- **DD**: 1178 graphs, chemical compound classification
- **MSRC_9**: 231 graphs, image segmentation graphs
- **AIDS**: 2000 graphs, molecular classification

### Configuration
**Location**: `graph_classify_v2.py:40-59`

```python
parser.add_argument('--name', type=str, default='mlp')  # Operator name
parser.add_argument('--gname', type=str, default='mlp')  # Model name
parser.add_argument('--ds', type=str, default='MUTAG')  # Dataset
parser.add_argument('--ep', type=int, default=1000 * 1.5)  # Epochs
parser.add_argument('--lr', default=1e-2, type=float)  # Learning rate
parser.add_argument('--drop', type=float, default=0.6)  # Dropout rate
parser.add_argument('--dim', type=int, default=64)  # Hidden dimensions
parser.add_argument('--h_layer', type=int, default=2)  # Number of hidden layers
```

---

## 6. Model Comparison Summary

| Model | Type | Residual | Cross-Branch | Level | Key Innovation |
|-------|------|----------|--------------|-------|----------------|
| **BlockGNN** | Single | ✗ | ✗ | - | Baseline sequential GNN |
| **ResBlockGnn** | Single | ✓ (intra-branch) | ✗ | Node | Standard residual connections |
| **CrossBlockGnn** | Dual (parallel) | ✓ | ✓ | **Node** | **Cross-branch node-level residual** |
| **GraphBlockGnn** | Dual (sequential) | ✗ | ✗ | Graph | Sequential graph passing |
| **ResGraphBlockGnn** | Triple (sequential) | ✓ (intra-branch) | ✗ | Graph | Sequential graph residual |
| **CrossGraphBlockGnn** | Quad (parallel) | ✓ | ✓ | **Graph** | **Cross-branch graph-level residual** |

---

## 7. Key Findings for Proposed Model Chapter

### Core Innovations to Describe

1. **Two Types of Cross-Residual Connections**:
   - **Node-Level** (CrossBlockGnn): Exchange intermediate node representations between branches
   - **Graph-Level** (CrossGraphBlockGnn): Exchange graph-level embeddings between branches

2. **Multi-Operator Framework**:
   - GCN, GAT, Transformer are **operator instances**, not separate models
   - `get_block_model()` function provides unified interface
   - Framework treats operators as modular components

3. **Cross-Branch Information Exchange**:
   - **Node-Level**: `x_cur_1 = layer(x_cur_1 + x_pre_2)` (branch 1 uses branch 2's history)
   - **Graph-Level**: `g_1 = global_mean_2; g_2 = global_mean_1` (swap graph embeddings)

4. **Modular Design**:
   - `res_graph` parameter controls whether graph_hidden is added
   - Easy to extend with new operators
   - Supports flexible depth and branching configurations

### Critical Constraints

1. **NOT a single GCN improvement** - GCN is just one operator instance
2. **Multi-operator framework** - Designed to unify multiple operators
3. **Two distinct cross-residual mechanisms** - Node-level and graph-level
4. **No fabricated modules** - All descriptions must match actual code

---

## 8. Proposed Model Chapter Structure

Based on code analysis, the chapter should include:

1. **Overview**: Multi-operator cross-residual framework for graph classification
2. **Cross-Residual Block Architecture**:
   - Node-level cross-residual (CrossBlockGnn)
   - Graph-level cross-residual (CrossGraphBlockGnn)
   - Mathematical formulations
3. **Operator Instantiation**: How framework accommodates GCN/GAT/Transformer via `get_block_model()`
4. **Readout and Classification**: Global mean pooling + linear classifier

---

## 9. Mathematical Formulations to Include

### Node-Level Cross-Residual
For two branches $k=1,2$ at layer $\ell$:
$$\mathbf{h}_i^{(\ell+1, 1)} = \sigma\left(\mathbf{W}^{(\ell, 1)} \cdot \text{AGG}\left(\left\{\mathbf{h}_j^{(\ell, 1)}\right\}, \mathbf{h}_i^{(\ell, 1)} + \mathbf{h}_i^{(\ell-1, 2)}\right)\right)$$
$$\mathbf{h}_i^{(\ell+1, 2)} = \sigma\left(\mathbf{W}^{(\ell, 2)} \cdot \text{AGG}\left(\left\{\mathbf{h}_j^{(\ell, 2)}\right\}, \mathbf{h}_i^{(\ell, 2)} + \mathbf{h}_i^{(\ell-1, 1)}\right)\right)$$

### Graph-Level Cross-Residual
For branch pairs $(p_1, p_2)$:
$$\mathbf{h}_{\mathcal{G}}^{(p_1, t)} = \mathbf{h}_{\mathcal{G}}^{(p_1, t-1)} + \mathbf{h}_{\mathcal{G}}^{(p_2, t-1)}$$
$$\mathbf{h}_{\mathcal{G}}^{(p_2, t)} = \mathbf{h}_{\mathcal{G}}^{(p_2, t-1)} + \mathbf{h}_{\mathcal{G}}^{p_1, t-1)}$$

---

**Phase A Status**: ✅ **COMPLETED**
**Ready for Phase B**: Writing 00_05_proposed_model.tex
