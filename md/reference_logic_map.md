# Reference Logic Map

This note reorganizes the merged paper corpus into the final citation flow for the revised manuscript.

Recommended section order:

1. Introduction: foundations, deep GNN degradation, residual information flow, biomolecular motivation, plant extension logic
2. Related Work: graph classification, pooling, hybrid architectures, biological graph learning, benchmark dataset context
3. Datasets: TU biological datasets and supplementary robustness datasets
4. Conclusion: plant-oriented future integration and omics extension

## GNN Foundations

- `kipf2016semi` (2016): Semi-supervised classification with graph convolutional networks
- `gilmer2017neural` (2017): Neural message passing for quantum chemistry
- `hamilton2017inductive` (2017): Inductive representation learning on large graphs
- `abu2019watch` (2018): Watch your step: Learning node embeddings via graph attention
- `klicpera2018combining` (2018): Combining label propagation and simple models out-performs graph neural networks
- `li2018deeper` (2018): Deeper insights into graph convolutional networks for semi-supervised learning
- `schlichtkrull2018modeling` (2018): Modeling relational data with graph convolutional networks
- `velivckovic2017graph` (2018): Graph attention networks
- `xu2018how` (2019): How powerful are graph neural networks?
- `xu2018powerful` (2019): How powerful are graph neural networks?
- `zhang2019personalized` (2019): Personalized graph neural networks with attention mechanism
- `bianchi2020graph` (2020): Graph neural networks with convolutional ARMA filters
- `oono2020simple` (2020): Simple and deep graph convolutional networks
- `zhou2020graph` (2020): Graph neural networks: A review of methods and applications
- `li2021training` (2021): Training graph neural networks with 1000 layers
- `wang2020multiscale` (2021): Optimization of graph neural networks: A survey
- `wang2021multi` (2021): Multi-hop attention graph neural networks
- `wu2020comprehensive` (2021): A comprehensive survey on graph neural networks
- `zhang2021graph` (2021): Graph neural networks: A survey of methods and applications
- `beddar2022analysis` (2022): An analysis of the expressive power of graph neural networks
- `li2022learning` (2022): Learning graph normalization for graph neural networks
- `reiserGraphNeuralNetworks2022` (2022): Graph Neural Networks for Materials Science and Chemistry
- `dongGraphNeuralNetworks2023` (2023): Graph Neural Networks in IoT: A Survey
- `guptaAgriGNNNovelGenotypicTopological2023` (2023): Agri-GNN: A Novel Genotypic-Topological Graph Neural Network Framework Built on GraphSAGE for Optimized Yield Prediction
- `liGraphNeuralNetwork2023` (2023): Graph Neural Network for Spatiotemporal Data: Methods and Applications
- `sun2023attention` (2023): Attention-based graph neural networks: a survey
- `owoeyeGraphNeuralNetwork2024` (2024): Graph Neural Network with Quasi-Data Augmentation for Modelling Food Web Relationships
- `paulSystematicReviewGraph2024` (2024): A Systematic Review of Graph Neural Network in Healthcare-Based Applications: Recent Advances, Trends, and Future Directions
- `wangMaizeYieldPrediction2024` (2024): Maize Yield Prediction with Trait-Missing Data via Bipartite Graph Neural Network
- `anakokInterpretabilityGraphNeural2025` (2025): Interpretability of Graph Neural Networks to Assess Effects of Global Change Drivers on Ecological Networks
- `samsonTransferLearningbasedGraph2025` (2025): Transfer Learning-Based Graph Neural Network Approach for Accurate Vegetable Disease Prediction
- `yangHeterogeneousPlantbasedFood2026` (2026): Heterogeneous Plant-Based Food Quality Evaluation Based on near-Infrared Spectroscopy Coupled with Graph Neural Network

## Deep GNN Stability

- `rong2019dropedge` (2019): DropEdge: Towards deep graph convolutional networks on node classification
- `chen2023revisiting` (2020): Revisiting over-smoothing in deep GCNs
- `zhao2020pair` (2020): Pairnorm: Tackling over-smoothing in gns
- `zhao2020tackling` (2020): Tackling over-smoothing for general graph learning
- `alon2020bottleneck` (2021): On the bottleneck of graph neural networks
- `cai2021graphnorm` (2021): GraphNorm: A principled approach to accelerating graph neural network training
- `topping2021understanding` (2021): Understanding over-squashing and bottlenecks on graphs via curvature
- `li2023non` (2023): A non-asymptotic analysis of oversmoothing in graph neural networks

## Residual And Cross-Layer Architectures

- `he2016deep` (2016): Deep residual learning for image recognition
- `bresson2017residual` (2017): Residual gated graph convnets
- `xu2018representation` (2018): Representation learning on graphs with jumping knowledge networks
- `abu2019mixhop` (2019): MixHop: Higher-order graph convolutional architectures via sparsified neighborhood mixing
- `gao2019graph` (2019): Graph U-Net
- `du2024densegnn` (2024): DenseGNN: universal and scalable deeper graph neural networks for high-performance property prediction in crystals and molecules

## Graph Classification And Pooling

- `duvenaud2015convolutional` (2015): Convolutional networks on graphs for learning molecular fingerprints
- `ying2018hierarchical` (2018): Hierarchical graph representation learning with differentiable pooling
- `fey2019fast` (2019): Fast graph representation learning with PyTorch Geometric
- `lee2019self` (2019): Self-attention graph pooling
- `mesquita2020rethinking` (2020): Rethinking pooling in graph neural networks
- `wang2023haarpool` (2020): Haar graph pooling
- `du2021multi` (2021): Multi-channel pooling graph neural networks
- `pham2021hierarchical` (2021): Hierarchical pooling in graph neural networks to enhance graph classification
- `li2022generalization` (2022): Generalization analysis of message passing neural networks on graph classification data
- `liu2022graph` (2023): Graph pooling for graph neural networks
- `li2024graph` (2024): Graph pooling for graph-level representation learning: a survey

## Biomolecular And Protein Graph Learning

- `chen2020conditional` (2020): Conditional graph convolutions for drug-protein binding prediction

## Graph Datasets And Biological Benchmarks

- `morris2020tudataset` (2020): TUDataset: A collection of benchmark datasets for learning with graphs

## Plant-Oriented Motivation

- `ayeshabarvinCropRecommendationSystems2023` (2023): Crop Recommendation Systems Based on Soil and Environmental Factors Using Graph Convolution Neural Network: A Systematic Literature Review
- `suiIdentificationPlantVacuole2023` (2023): Identification of Plant Vacuole Proteins by Using Graph Neural Network and Contact Maps
- `beraPNDNetPlantNutrition2024` (2024): PND-net: Plant Nutrition Deficiency and Disease Classification Using Graph Convolutional Network
- `changPredictingAbioticStressresponsive2024` (2024): Predicting Abiotic Stress-Responsive miRNA in Plants Based on Multi-Source Features Fusion and Graph Neural Network
- `zhangImprovingPlantMiRNAtarget2024` (2024): Improving Plant miRNA-target Prediction with Self-Supervised k-Mer Embedding and Spectral Graph Convolutional Neural Network
- `baongoImprovingPlantFunctional2025` (2025): Improving Plant Functional Annotation from Knowledge Graphs Using Graph Neural Networks
- `bossounGraphNeuralNetwork2025` (2025): A Graph Neural Network Approach for Early Plant Disease Detection
- `charulekhaUncoveringComplicatedPlant` (2025): Uncovering Complicated Plant Biological Networks through the Assistance of Artificial Intelligence Tools
- `princySoilMicrobiomeInspiredGraph2025` (2025): Soil Microbiome-Inspired Graph Neural Networks for Crop Yield Prediction

## Additional Related Work

- `gori2005new` (2005): A new model for learning in graph domains
- `yang2016revisiting` (2016): Revisiting semi-supervised learning with graph embeddings
- `cangea2018towards` (2018): Towards sparse hierarchical graph classifiers
- `beaini2020directional` (2020): Directional graph networks
- `cai2020understanding` (2020): Understanding graph embedding methods and their applications
- `dwivedi2020generalization` (2020): A generalization of transformer networks to graphs
- `gravina2022anti` (2022): Anti-symmetric DGN: A stable architecture for deep graph networks
- `rossi2023edge` (2023): Edge directionality improves learning on heterophilic graphs
- `vatsaMaizePhenotypeClassification2024` (2024): Maize Phenotype Classification Using GNN: Research Perspective
