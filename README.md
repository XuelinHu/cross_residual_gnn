# Cross Residual Graph Neural Networks

<p align="center">
  <img height="20" src="https://img.shields.io/badge/PyTorch-2.0+-red" />
  <img height="20" src="https://img.shields.io/badge/PyTorch_Geometric-2.3+-blue" />
  <img height="20" src="https://img.shields.io/badge/Python-3.8+-green" />
  <img height="20" src="https://img.shields.io/badge/License-GPL_v3.0-purple" />
</p>

A research implementation of Cross Residual Graph Neural Networks for enhanced graph classification and node classification tasks. This project explores novel residual connection architectures to improve gradient flow and enable deeper GNN models.

## 🚀 Key Features

- **Cross Residual Connections**: Novel residual architectures that skip layers and combine information from different depths
- **Multiple GNN Variants**: Support for GCN, GAT, Transformer, and advanced convolution types
- **Graph & Node Classification**: Comprehensive implementations for both graph-level and node-level tasks
- **Extensive Benchmarking**: Tested on standard TU Dortmund graph classification datasets
- **Modular Design**: Clean, extensible codebase for easy experimentation
- **Comprehensive Logging**: Detailed experiment tracking and result visualization

## 📁 Project Structure

```
cross_residual_gnn/
├── geomatric/                    # Main GNN implementation directory
│   ├── graph_classify_v2.py      # Main graph classification implementation
│   ├── node_classify.py          # Node classification implementation
│   ├── analysis.py               # Data analysis and visualization
│   ├── generate_graph.py         # Graph generation utilities
│   └── achivement/               # Experimental results and variants
│       ├── graph_classify.py     # Cross residual implementation
│       ├── graph_classify_v1.py  # Version 1 implementation
│       └── node_classify.py      # Node classification experiments
├── records/                      # Experiment results and logs
├── utils.py                      # Utility functions
├── logging_config.py             # Logging configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/cross_residual_gnn.git
cd cross_residual_gnn

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Alternative Installation

```bash
# Install PyTorch Geometric manually (if requirements.txt fails)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

## 🚀 Quick Start

### Graph Classification

```bash
# Run ResGCN on MUTAG dataset
python geomatric/graph_classify_v2.py --name ResGCN --ds MUTAG --hidden 64 --ep 200

# Run ResGAT on DD dataset
python geomatric/graph_classify_v2.py --name ResGAT --ds DD --hidden 128 --heads 8 --ep 300

# Compare with baseline models
python geomatric/graph_classify_v2.py --name GCN --ds MUTAG --hidden 64 --ep 200
```

### Node Classification

```bash
# Run node classification experiments
python geomatric/achivement/node_classify.py --name ResGCN --hidden 64 --ep 500
```

## 📊 Supported Datasets

The implementation supports standard TUDataset benchmark datasets:

- **MUTAG** (188 graphs)
- **DD** (1178 graphs)
- **COIL-RAG** (600 graphs)
- **MSRC_9** (231 graphs)
- **AIDS** (2000 graphs)
- **Mutagenicity** (4337 graphs)

## 🧠 Model Architectures

### Standard Models
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network
- **Transformer**: Graph Transformer

### Cross Residual Models
- **ResGCN**: GCN with cross residual connections
- **ResGAT**: GAT with cross residual connections

### Advanced Convolution Types
- **MixHopConv**: Multi-hop neighborhood aggregation
- **DirGNNConv**: Directional graph neural network
- **AntiSymmetricConv**: Anti-symmetric graph convolution

## ⚙️ Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--name` | Model type (GCN, ResGCN, GAT, ResGAT) | `GCN` |
| `--ds` | Dataset name | `MUTAG` |
| `--hidden` | Hidden layer dimensions | `64` |
| `--lr` | Learning rate | `0.01` |
| `--drop` | Dropout rate | `0.5` |
| `--ep` | Number of epochs | `200` |
| `--heads` | Number of attention heads (GAT only) | `8` |

## 📈 Results

Experimental results show that cross residual connections improve performance on several benchmark datasets:

- **MUTAG**: ResGCN achieves ~85% accuracy (vs ~80% for standard GCN)
- **DD**: ResGAT shows improved convergence and stability
- **Training**: Faster convergence and better gradient flow

Results are automatically saved in the `records/` directory with detailed logs and visualizations.

## 🔬 Core Innovation

The **cross residual connections** implemented in this project enable:

1. **Better Gradient Flow**: Residual connections prevent vanishing gradients in deep architectures
2. **Multi-Scale Feature Learning**: Information from different network depths is effectively combined
3. **Improved Training Stability**: More robust training across different datasets and architectures

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a Pull Request

For major changes, please open an issue first to discuss your proposed modifications.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{cross_residual_gnn2024,
  title={Cross Residual Graph Neural Networks for Enhanced Graph Classification},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 📄 License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## 🐛 Issues

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/cross_residual_gnn/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## 📚 Paper Documentation

### Paper Structure
The research paper is organized in the `paper/` directory with the following structure:

```
paper/
├── main.tex                      # Main LaTeX document (IEEE conference format)
├── references.bib                # Bibliography with 69 references
├── paper_all_detailed_summaries.json  # Detailed summaries of 57 papers
├── compile.bat                   # Compilation script for Windows
└── sections/
    ├── 00_01_abstract.tex        # Abstract
    ├── 00_02_introduction.tex    # Introduction (✅ Completed)
    ├── 00_03_related_work.tex    # Related Work (✅ Completed)
    ├── 00_04_task_definition.tex # Task Definition (✅ Completed)
    ├── 00_05_proposed_model.tex  # Proposed Model
    ├── 00_06_datasets.tex        # Datasets
    ├── 00_07_experiments.tex     # Experiments
    └── 00_08_conclusion.tex      # Conclusion
```

### Latest Updates
- **2025-02-09**: Datasets chapter completed ✅
  - **Code analysis**: Analyzed graph_classify_v2.py and analysis.py
  - **Datasets identified**: MUTAG, DD, MSRC_9, AIDS from TUDataset
  - **Full chapter writing** (145 lines, 6 subsections):
    - Dataset Overview: 4 datasets with domain descriptions
    - Dataset Statistics: Summary table with key metrics
    - Data Loading and Configuration: TUDataset loader details
    - Data Splitting Protocol: 5-fold cross-validation with mathematical formulation
    - Data Preprocessing and Augmentation: Minimal preprocessing strategy
    - Evaluation Metrics: Accuracy ± standard deviation
  - **Key details**:
    * Data source: TUDataset \cite{keriven2020benchmark}
    * Loader: PyTorch Geometric \cite{fey2019fast}
    * 5-fold split formula: train/test split mathematical specification
    * No feature normalization, no data augmentation
    * Batch size: 32, Random seed: 1024
  - **Placeholder usage**: Feature dimensions marked as [PLACEHOLDER] (not specified in code)
  - All descriptions consistent with actual implementation

- **2025-02-09**: Proposed Model chapter completed ✅
  - **Phase A**: Comprehensive code structure analysis (md/phase_A_structure_extraction.md)
    - Analyzed graph_classify_v2.py (811 lines)
    - Identified 6 model classes: BlockGNN, ResBlockGnn, CrossBlockGnn, GraphBlockGnn, ResGraphBlockGnn, CrossGraphBlockGnn
    - Extracted exact forward pass logic and mathematical formulations
  - **Phase B**: Full chapter writing (152 lines, 5 subsections)
    - Overview: Multi-operator cross-residual architecture with GCN/GAT/Transformer as instances
    - Node-Level Cross-Residual Block: CrossBlockGnn mechanism (Eq. 6-11)
    - Graph-Level Cross-Residual Block: CrossGraphBlockGnn mechanism (Eq. 15-17)
    - Operator Instantiation Framework: Modular operator design (Eq. 18-26)
    - Readout and Classification: Global mean pooling + linear classifier
  - **Key innovations**:
    * Node-level cross-branch: $\mathbf{H}^{(\ell+1, 1)} = \sigma(\Phi_1(\mathbf{H}^{(\ell, 1)}) + \mathbf{H}^{(\ell-1, 2)})$
    * Graph-level cross-branch: $\mathbf{h}_{\mathcal{G}}^{(k_1, t+1)} = \mathbf{h}_{\mathcal{G}}^{(k_1, t)} + \mathbf{h}_{\mathcal{G}}^{(k_2, t)}$
  - **Critical emphasis**: NOT a single GCN improvement - GCN is just one operator instance
  - All descriptions 100% consistent with actual code - no fabricated modules

- **2025-02-09**: Task Definition chapter completed
  - 5 subsections: Graph Representation, Graph Classification, Multi-Operator Message Passing, Graph-Level Representation, Learning Objective
  - 3 equation environments: message passing, readout function, cross-entropy loss
  - Emphasizes framework性质: "treats graph convolution operators as modular components"
  - Supports binary, multi-class, and multi-label classification
  - Cites 5 key operator papers (GCN, GAT, GraphSAGE, GIN, Transformer)

- **2025-02-09**: Related Work chapter completed
  - 5-category structure covering 40+ papers
  - Categories: (1) Graph Classification GNNs, (2) Message Passing Operators, (3) Deep Training Challenges, (4) Multi-Operator Architectures, (5) Cross-Layer Aggregation
  - Clear positioning: "critical gap - operators as alternatives, not complementary components"
  - All citations from reference.bib with correct \cite{bibkey} format

- **2025-02-09**: Introduction chapter completed
  - 5-paragraph structure: background, GNN paradigms, related work gap, proposed method, contributions
  - Emphasizes cross-residual framework unifying multiple operators (GCN/GAT/Transformer)
  - Based on actual code implementation (CrossBlockGnn, CrossGraphBlockGnn)
  - Cites 15+ key papers (GCN, GAT, ResNet, JK networks, DenseGNN, etc.)

### Reference Summary
- **Total References**: 69 papers
- **Detailed Summaries**: 57 papers with abstracts, contributions, and relevance analysis
- **Total Citations**: 286,780
- **Key Papers**:
  - Kipf & Welling 2016 (GCN) - 25,000 citations
  - Veličković et al. 2018 (GAT) - 15,000 citations
  - He et al. 2016 (ResNet) - 160,000 citations
  - Xu et al. 2018 (GIN) - 13,000 citations

### Compile the Paper
```bash
# On Windows
cd paper
compile.bat

# On Linux/Mac
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 📊 Experiment Results

### Result Files
- `records/v3result.xlsx` - Experimental results v3
- `records/v4result.xlsx` - Experimental results v4

### Datasets
- **TUDataset**: MUTAG, DD, COIL-RAG, MSRC_9, AIDS, Mutagenicity
- **Planetoid**: Cora, CiteSeer, PubMed

## 📂 Working Directories

### MD Directory
Contains markdown documentation:
- `SESSION_RECORD.md` - Detailed session history and progress

### PY Directory
Contains Python utility scripts and analysis tools

## 🔗 Related Work

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

**Last Updated**: 2025-02-09