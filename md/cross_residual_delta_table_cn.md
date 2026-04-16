# 交叉残差相对提升差值表

说明：
- NodeCross-Plain = 节点交叉残差 相对 无残差 PlainGNN 的提升
- NodeCross-NodeRes = 节点交叉残差 相对 节点残差 NodeResGNN 的提升
- GraphCross-Plain = 图级交叉残差 相对 无残差 PlainGNN 的提升
- GraphCross-GraphRes = 图级交叉残差 相对 图级残差 GraphResGNN 的提升

## PROTEINS
| 算子 | NodeCross-Plain | NodeCross-NodeRes | GraphCross-Plain | GraphCross-GraphRes |
|---|---:|---:|---:|---:|
| GCNConv | -0.01168 | -0.00087 | -0.02429 | -0.04134 |
| GATConv | -0.01528 | -0.01799 | +0.00538 | +0.00634 |
| TransformerConv | +0.00002 | +0.00807 | +0.02429 | +0.00176 |
| SAGEConv | -0.02430 | -0.01171 | -0.00452 | -0.00983 |
| GINConv | -0.00093 | -0.00808 | +0.00896 | +0.00089 |

## DD
| 算子 | NodeCross-Plain | NodeCross-NodeRes | GraphCross-Plain | GraphCross-GraphRes |
|---|---:|---:|---:|---:|
| GCNConv | +0.00930 | +0.00509 | -0.00089 | +0.02211 |
| GATConv | -0.00942 | -0.01278 | +0.01268 | +0.01611 |
| TransformerConv | +0.11796 | +0.03488 | +0.08566 | +0.08566 |
| SAGEConv | +0.02627 | +0.01357 | -0.00428 | +0.03294 |
| GINConv | +0.02725 | +0.00678 | +0.00178 | -0.01189 |

## ENZYMES
| 算子 | NodeCross-Plain | NodeCross-NodeRes | GraphCross-Plain | GraphCross-GraphRes |
|---|---:|---:|---:|---:|
| GCNConv | +0.03333 | +0.01333 | +0.03500 | -0.03167 |
| GATConv | +0.03333 | +0.02666 | +0.01000 | -0.01000 |
| TransformerConv | +0.06666 | +0.04500 | +0.04333 | -0.01667 |
| SAGEConv | +0.02666 | +0.01666 | +0.02833 | -0.03167 |
| GINConv | +0.07500 | +0.03667 | +0.04167 | -0.02166 |

## MUTAG
| 算子 | NodeCross-Plain | NodeCross-NodeRes | GraphCross-Plain | GraphCross-GraphRes |
|---|---:|---:|---:|---:|
| GCNConv | +0.01608 | +0.00541 | -0.01067 | -0.01608 |
| GATConv | -0.00014 | +0.00000 | +0.00498 | +0.02674 |
| TransformerConv | +0.04779 | +0.03755 | +0.01024 | +0.00484 |
| SAGEConv | -0.01067 | -0.01593 | -0.01067 | -0.00569 |
| GINConv | +0.02675 | +0.01608 | -0.00483 | +0.00583 |

## AIDS
| 算子 | NodeCross-Plain | NodeCross-NodeRes | GraphCross-Plain | GraphCross-GraphRes |
|---|---:|---:|---:|---:|
| GCNConv | +0.02800 | +0.00450 | +0.02550 | -0.04500 |
| GATConv | +0.02950 | +0.03700 | +0.00200 | -0.00900 |
| TransformerConv | +0.03900 | +0.01750 | +0.00550 | -0.02650 |
| SAGEConv | +0.01950 | -0.01250 | +0.01100 | -0.01800 |
| GINConv | +0.02750 | +0.00550 | +0.00600 | -0.05400 |

## Mutagenicity
| 算子 | NodeCross-Plain | NodeCross-NodeRes | GraphCross-Plain | GraphCross-GraphRes |
|---|---:|---:|---:|---:|
| GCNConv | +0.02191 | +0.01360 | +0.02145 | +0.00069 |
| GATConv | +0.11946 | +0.01568 | +0.11163 | +0.07359 |
| TransformerConv | +0.10049 | +0.01845 | +0.10602 | +0.00692 |
| SAGEConv | +0.01798 | +0.01406 | +0.00161 | -0.00048 |
| GINConv | +0.11985 | -0.00138 | +0.11685 | +0.00945 |

