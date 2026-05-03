# Statistical Significance Summary

## Key Findings

### PROTEINS

**Operator: GCNConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0117 | 0.2861 | 0.3750 | +0.550 | not significant |
| node-cross vs plain | -0.0081 | 0.5710 | 1.0000 | -0.276 | not significant |
| graph-res vs plain | +0.0045 | 0.3747 | 0.5000 | +0.446 | not significant |
| graph-cross vs plain | -0.0009 | 0.9611 | 1.0000 | -0.023 | not significant |
| node-cross vs node-res | -0.0198 | 0.3020 | 0.3750 | -0.529 | not significant |
| graph-cross vs graph-res | -0.0054 | 0.7514 | 1.0000 | -0.152 | not significant |
| graph-res vs node-res | -0.0072 | 0.5876 | 0.6875 | -0.263 | not significant |
| node-cross vs graph-cross | -0.0072 | 0.3374 | 0.6250 | -0.487 | not significant |

**Operator: GATConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0000 | 0.9992 | 1.0000 | +0.000 | not significant |
| node-cross vs plain | -0.0109 | 0.6473 | 0.6250 | -0.221 | not significant |
| graph-res vs plain | +0.0152 | 0.3455 | 0.4375 | +0.478 | not significant |
| graph-cross vs plain | -0.0081 | 0.6228 | 0.6250 | -0.238 | not significant |
| node-cross vs node-res | -0.0109 | 0.6902 | 0.8125 | -0.192 | not significant |
| graph-cross vs graph-res | -0.0234 | 0.0082 | 0.0625 | -2.179 | marginal |
| graph-res vs node-res | +0.0152 | 0.4591 | 0.8125 | +0.366 | not significant |
| node-cross vs graph-cross | -0.0027 | 0.8376 | 0.8750 | -0.098 | not significant |

**Operator: SAGEConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0063 | 0.5243 | 0.6250 | +0.312 | not significant |
| node-cross vs plain | -0.0018 | 0.8811 | 1.0000 | -0.071 | not significant |
| graph-res vs plain | +0.0296 | 0.0195 | 0.0625 | +1.690 | marginal |
| graph-cross vs plain | +0.0036 | 0.6216 | 0.8125 | +0.239 | not significant |
| node-cross vs node-res | -0.0081 | 0.1809 | 0.2500 | -0.724 | not significant |
| graph-cross vs graph-res | -0.0260 | 0.0305 | 0.0625 | -1.467 | marginal |
| graph-res vs node-res | +0.0233 | 0.1708 | 0.2500 | +0.746 | not significant |
| node-cross vs graph-cross | -0.0054 | 0.6591 | 1.0000 | -0.213 | not significant |

**Operator: GINConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0117 | 0.2765 | 0.5000 | +0.563 | not significant |
| node-cross vs plain | -0.0081 | 0.6320 | 0.8125 | -0.232 | not significant |
| graph-res vs plain | +0.0395 | 0.0690 | 0.1250 | +1.104 | not significant |
| graph-cross vs plain | +0.0036 | 0.7429 | 1.0000 | +0.157 | not significant |
| node-cross vs node-res | -0.0198 | 0.1248 | 0.1875 | -0.866 | not significant |
| graph-cross vs graph-res | -0.0359 | 0.0062 | 0.0625 | -2.362 | marginal |
| graph-res vs node-res | +0.0278 | 0.0795 | 0.1250 | +1.046 | not significant |
| node-cross vs graph-cross | -0.0117 | 0.4275 | 0.6250 | -0.395 | not significant |


### DD

**Operator: GCNConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0118 | 0.4292 | 0.6250 | +0.393 | not significant |
| node-cross vs plain | +0.0194 | 0.2856 | 0.4375 | +0.551 | not significant |
| graph-res vs plain | +0.0305 | 0.2235 | 0.3125 | +0.644 | not significant |
| graph-cross vs plain | +0.0237 | 0.1227 | 0.1875 | +0.873 | not significant |
| node-cross vs node-res | +0.0076 | 0.3945 | 0.4375 | +0.426 | not significant |
| graph-cross vs graph-res | -0.0068 | 0.6368 | 0.6250 | -0.228 | not significant |
| graph-res vs node-res | +0.0187 | 0.2280 | 0.3125 | +0.636 | not significant |
| node-cross vs graph-cross | -0.0043 | 0.5259 | 0.3125 | -0.310 | not significant |

**Operator: GATConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0017 | 0.6237 | 1.0000 | +0.237 | not significant |
| node-cross vs plain | +0.0348 | 0.1154 | 0.1250 | +0.897 | not significant |
| graph-res vs plain | +0.0204 | 0.3262 | 0.6250 | +0.500 | not significant |
| graph-cross vs plain | +0.0187 | 0.3574 | 0.4375 | +0.465 | not significant |
| node-cross vs node-res | +0.0331 | 0.1205 | 0.1250 | +0.880 | not significant |
| graph-cross vs graph-res | -0.0017 | 0.8147 | 0.8125 | -0.112 | not significant |
| graph-res vs node-res | +0.0187 | 0.3586 | 0.8125 | +0.463 | not significant |
| node-cross vs graph-cross | +0.0161 | 0.0752 | 0.0625 | +1.068 | not significant |

**Operator: SAGEConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0160 | 0.3475 | 0.3750 | +0.476 | not significant |
| node-cross vs plain | +0.0229 | 0.1240 | 0.1875 | +0.869 | not significant |
| graph-res vs plain | +0.0212 | 0.1118 | 0.1250 | +0.909 | not significant |
| graph-cross vs plain | +0.0042 | 0.6989 | 1.0000 | +0.186 | not significant |
| node-cross vs node-res | +0.0068 | 0.2257 | 0.3125 | +0.640 | not significant |
| graph-cross vs graph-res | -0.0170 | 0.0032 | 0.0625 | -2.816 | marginal |
| graph-res vs node-res | +0.0051 | 0.6245 | 0.6250 | +0.237 | not significant |
| node-cross vs graph-cross | +0.0187 | 0.0268 | 0.0625 | +1.529 | marginal |

**Operator: GINConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0008 | 0.9458 | 0.8125 | +0.032 | not significant |
| node-cross vs plain | +0.0246 | 0.1071 | 0.1250 | +0.926 | not significant |
| graph-res vs plain | +0.0017 | 0.8916 | 0.8750 | +0.065 | not significant |
| graph-cross vs plain | +0.0076 | 0.6722 | 0.6250 | +0.204 | not significant |
| node-cross vs node-res | +0.0238 | 0.1033 | 0.1875 | +0.940 | not significant |
| graph-cross vs graph-res | +0.0058 | 0.7624 | 0.8750 | +0.145 | not significant |
| graph-res vs node-res | +0.0009 | 0.9538 | 0.8125 | +0.028 | not significant |
| node-cross vs graph-cross | +0.0170 | 0.2742 | 0.8125 | +0.566 | not significant |


### ENZYMES

**Operator: GCNConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | -0.0000 | 1.0000 | 1.0000 | -0.000 | not significant |
| node-cross vs plain | +0.0033 | 0.8466 | 0.6250 | +0.092 | not significant |
| graph-res vs plain | +0.0650 | 0.2481 | 0.3125 | +0.604 | not significant |
| graph-cross vs plain | +0.0167 | 0.5258 | 0.6250 | +0.310 | not significant |
| node-cross vs node-res | +0.0033 | 0.8835 | 0.8750 | +0.070 | not significant |
| graph-cross vs graph-res | -0.0483 | 0.2607 | 0.3125 | -0.585 | not significant |
| graph-res vs node-res | +0.0650 | 0.2800 | 0.3125 | +0.558 | not significant |
| node-cross vs graph-cross | -0.0133 | 0.4997 | 0.8750 | -0.331 | not significant |

**Operator: GATConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0050 | 0.6974 | 0.8750 | +0.187 | not significant |
| node-cross vs plain | +0.0317 | 0.3694 | 0.6250 | +0.452 | not significant |
| graph-res vs plain | +0.0300 | 0.1369 | 0.2500 | +0.830 | not significant |
| graph-cross vs plain | -0.0000 | 1.0000 | 1.0000 | -0.000 | not significant |
| node-cross vs node-res | +0.0267 | 0.3495 | 0.8125 | +0.473 | not significant |
| graph-cross vs graph-res | -0.0300 | 0.0250 | 0.0625 | -1.564 | marginal |
| graph-res vs node-res | +0.0250 | 0.2034 | 0.3125 | +0.679 | not significant |
| node-cross vs graph-cross | +0.0317 | 0.2437 | 0.3125 | +0.611 | not significant |

**Operator: SAGEConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | -0.0167 | 0.1027 | 0.1250 | -0.943 | not significant |
| node-cross vs plain | +0.0067 | 0.6135 | 0.6250 | +0.245 | not significant |
| graph-res vs plain | +0.0817 | 0.1212 | 0.1875 | +0.878 | not significant |
| graph-cross vs plain | +0.0050 | 0.6657 | 0.8750 | +0.208 | not significant |
| node-cross vs node-res | +0.0233 | 0.0447 | 0.0625 | +1.292 | marginal |
| graph-cross vs graph-res | -0.0767 | 0.1059 | 0.1250 | -0.931 | not significant |
| graph-res vs node-res | +0.0983 | 0.0618 | 0.0625 | +1.150 | not significant |
| node-cross vs graph-cross | +0.0017 | 0.9281 | 1.0000 | +0.043 | not significant |

**Operator: GINConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0300 | 0.3512 | 0.6250 | +0.471 | not significant |
| node-cross vs plain | +0.0317 | 0.0090 | 0.0625 | +2.124 | marginal |
| graph-res vs plain | +0.0767 | 0.0018 | 0.0625 | +3.315 | marginal |
| graph-cross vs plain | +0.0317 | 0.1061 | 0.1250 | +0.930 | not significant |
| node-cross vs node-res | +0.0017 | 0.9487 | 1.0000 | +0.031 | not significant |
| graph-cross vs graph-res | -0.0450 | 0.1027 | 0.1250 | -0.943 | not significant |
| graph-res vs node-res | +0.0467 | 0.1660 | 0.1875 | +0.756 | not significant |
| node-cross vs graph-cross | -0.0000 | 1.0000 | 1.0000 | -0.000 | not significant |


### MUTAG

**Operator: GCNConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0108 | 0.1778 | 0.5000 | +0.730 | not significant |
| node-cross vs plain | +0.0055 | 0.6086 | 0.5000 | +0.248 | not significant |
| graph-res vs plain | -0.0105 | 0.6921 | 0.7500 | -0.190 | not significant |
| graph-cross vs plain | +0.0054 | 0.3739 | 1.0000 | +0.447 | not significant |
| node-cross vs node-res | -0.0053 | 0.3739 | 1.0000 | -0.447 | not significant |
| graph-cross vs graph-res | +0.0159 | 0.5012 | 0.6250 | +0.330 | not significant |
| graph-res vs node-res | -0.0213 | 0.4320 | 0.6250 | -0.390 | not significant |
| node-cross vs graph-cross | +0.0001 | 0.9874 | 1.0000 | +0.008 | not significant |

**Operator: GATConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0110 | 0.7539 | 1.0000 | +0.150 | not significant |
| node-cross vs plain | +0.0272 | 0.5432 | 1.0000 | +0.297 | not significant |
| graph-res vs plain | +0.0058 | 0.9087 | 0.8750 | +0.055 | not significant |
| graph-cross vs plain | +0.0377 | 0.3121 | 0.3750 | +0.517 | not significant |
| node-cross vs node-res | +0.0162 | 0.2080 | 0.5000 | +0.671 | not significant |
| graph-cross vs graph-res | +0.0319 | 0.1491 | 0.3125 | +0.798 | not significant |
| graph-res vs node-res | -0.0051 | 0.7601 | 0.7500 | -0.146 | not significant |
| node-cross vs graph-cross | -0.0105 | 0.3811 | 0.7500 | -0.440 | not significant |

**Operator: SAGEConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | -0.0055 | 0.6086 | 0.5000 | -0.248 | not significant |
| node-cross vs plain | -0.0267 | 0.1905 | 0.5000 | -0.704 | not significant |
| graph-res vs plain | -0.0321 | 0.0726 | 0.1250 | -1.083 | not significant |
| graph-cross vs plain | -0.0054 | 0.8233 | 0.8750 | -0.107 | not significant |
| node-cross vs node-res | -0.0212 | 0.2459 | 0.3750 | -0.608 | not significant |
| graph-cross vs graph-res | +0.0267 | 0.2929 | 0.3125 | +0.541 | not significant |
| graph-res vs node-res | -0.0266 | 0.0358 | 0.1250 | -1.392 | marginal |
| node-cross vs graph-cross | -0.0213 | 0.3757 | 0.4375 | -0.445 | not significant |

**Operator: GINConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | -0.0051 | 0.8337 | 0.8750 | -0.100 | not significant |
| node-cross vs plain | -0.0159 | 0.6630 | 0.7500 | -0.210 | not significant |
| graph-res vs plain | -0.0102 | 0.8265 | 0.6250 | -0.105 | not significant |
| graph-cross vs plain | -0.0162 | 0.6672 | 0.7500 | -0.207 | not significant |
| node-cross vs node-res | -0.0108 | 0.6149 | 0.5000 | -0.244 | not significant |
| graph-cross vs graph-res | -0.0060 | 0.8558 | 0.8125 | -0.087 | not significant |
| graph-res vs node-res | -0.0051 | 0.8651 | 1.0000 | -0.081 | not significant |
| node-cross vs graph-cross | +0.0003 | 0.9855 | 1.0000 | +0.009 | not significant |


### AIDS

**Operator: GCNConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0320 | 0.2764 | 0.2500 | +0.563 | not significant |
| node-cross vs plain | +0.0555 | 0.0388 | 0.0625 | +1.354 | marginal |
| graph-res vs plain | +0.0795 | 0.0388 | 0.0625 | +1.355 | marginal |
| graph-cross vs plain | +0.0445 | 0.1460 | 0.1250 | +0.806 | not significant |
| node-cross vs node-res | +0.0235 | 0.1177 | 0.1250 | +0.889 | not significant |
| graph-cross vs graph-res | -0.0350 | 0.1090 | 0.1250 | -0.919 | not significant |
| graph-res vs node-res | +0.0475 | 0.0363 | 0.0625 | +1.386 | marginal |
| node-cross vs graph-cross | +0.0110 | 0.2822 | 0.3125 | +0.555 | not significant |

**Operator: GATConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | -0.0160 | 0.4605 | 1.0000 | -0.365 | not significant |
| node-cross vs plain | +0.0165 | 0.0921 | 0.1250 | +0.986 | not significant |
| graph-res vs plain | -0.0060 | 0.7515 | 0.8750 | -0.152 | not significant |
| graph-cross vs plain | -0.0065 | 0.2903 | 0.5000 | -0.545 | not significant |
| node-cross vs node-res | +0.0325 | 0.1962 | 0.1875 | +0.693 | not significant |
| graph-cross vs graph-res | -0.0005 | 0.9796 | 0.8125 | -0.012 | not significant |
| graph-res vs node-res | +0.0100 | 0.7238 | 0.8750 | +0.170 | not significant |
| node-cross vs graph-cross | +0.0230 | 0.0179 | 0.0625 | +1.732 | marginal |

**Operator: SAGEConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | -0.0050 | 0.6073 | 0.7500 | -0.249 | not significant |
| node-cross vs plain | +0.0000 | 1.0000 | 0.6250 | +0.000 | not significant |
| graph-res vs plain | +0.0295 | 0.2954 | 0.3750 | +0.538 | not significant |
| graph-cross vs plain | -0.0100 | 0.3406 | 0.4375 | -0.483 | not significant |
| node-cross vs node-res | +0.0050 | 0.2488 | 0.3125 | +0.603 | not significant |
| graph-cross vs graph-res | -0.0395 | 0.1130 | 0.1250 | -0.905 | not significant |
| graph-res vs node-res | +0.0345 | 0.2349 | 0.3750 | +0.625 | not significant |
| node-cross vs graph-cross | +0.0100 | 0.5624 | 0.6250 | +0.282 | not significant |

**Operator: GINConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0020 | 0.5769 | 0.7500 | +0.271 | not significant |
| node-cross vs plain | +0.0370 | 0.2392 | 0.2500 | +0.618 | not significant |
| graph-res vs plain | +0.0475 | 0.2544 | 0.3125 | +0.595 | not significant |
| graph-cross vs plain | -0.0070 | 0.5062 | 0.6250 | -0.326 | not significant |
| node-cross vs node-res | +0.0350 | 0.2676 | 0.3125 | +0.575 | not significant |
| graph-cross vs graph-res | -0.0545 | 0.1620 | 0.1250 | -0.766 | not significant |
| graph-res vs node-res | +0.0455 | 0.2733 | 0.4375 | +0.567 | not significant |
| node-cross vs graph-cross | +0.0440 | 0.1400 | 0.1250 | +0.822 | not significant |


### Mutagenicity

**Operator: GCNConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0044 | 0.5971 | 0.8750 | +0.256 | not significant |
| node-cross vs plain | +0.0108 | 0.2008 | 0.2500 | +0.684 | not significant |
| graph-res vs plain | +0.0191 | 0.0501 | 0.1250 | +1.240 | not significant |
| graph-cross vs plain | +0.0162 | 0.1310 | 0.1250 | +0.847 | not significant |
| node-cross vs node-res | +0.0065 | 0.5879 | 0.6250 | +0.263 | not significant |
| graph-cross vs graph-res | -0.0030 | 0.5449 | 0.6250 | -0.295 | not significant |
| graph-res vs node-res | +0.0148 | 0.1249 | 0.1875 | +0.866 | not significant |
| node-cross vs graph-cross | -0.0053 | 0.4763 | 0.6250 | -0.351 | not significant |

**Operator: GATConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0113 | 0.1011 | 0.3125 | +0.949 | not significant |
| node-cross vs plain | +0.0198 | 0.0794 | 0.0625 | +1.047 | not significant |
| graph-res vs plain | +0.0078 | 0.2729 | 0.3750 | +0.568 | not significant |
| graph-cross vs plain | +0.0095 | 0.3657 | 0.4375 | +0.456 | not significant |
| node-cross vs node-res | +0.0085 | 0.1952 | 0.1875 | +0.695 | not significant |
| graph-cross vs graph-res | +0.0016 | 0.8745 | 1.0000 | +0.075 | not significant |
| graph-res vs node-res | -0.0035 | 0.5832 | 1.0000 | -0.267 | not significant |
| node-cross vs graph-cross | +0.0104 | 0.3238 | 0.4375 | +0.503 | not significant |

**Operator: SAGEConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0141 | 0.1058 | 0.0625 | +0.931 | not significant |
| node-cross vs plain | +0.0129 | 0.1270 | 0.1250 | +0.860 | not significant |
| graph-res vs plain | +0.0115 | 0.2367 | 0.4375 | +0.622 | not significant |
| graph-cross vs plain | +0.0115 | 0.3326 | 0.4375 | +0.493 | not significant |
| node-cross vs node-res | -0.0012 | 0.7871 | 0.8125 | -0.129 | not significant |
| graph-cross vs graph-res | +0.0000 | 0.9993 | 1.0000 | +0.000 | not significant |
| graph-res vs node-res | -0.0025 | 0.7141 | 0.8125 | -0.176 | not significant |
| node-cross vs graph-cross | +0.0014 | 0.8320 | 0.6250 | +0.101 | not significant |

**Operator: GINConv**

| Comparison | Δ | t p-value | W p-value | Cohen's d | Verdict |
|---|---|---|---|---|---|
| residual vs plain | +0.0134 | 0.1365 | 0.1875 | +0.831 | not significant |
| node-cross vs plain | +0.0168 | 0.1015 | 0.1875 | +0.948 | not significant |
| graph-res vs plain | +0.0182 | 0.1386 | 0.1250 | +0.826 | not significant |
| graph-cross vs plain | +0.0101 | 0.1466 | 0.1875 | +0.804 | not significant |
| node-cross vs node-res | +0.0035 | 0.5030 | 0.4375 | +0.329 | not significant |
| graph-cross vs graph-res | -0.0081 | 0.5128 | 0.6250 | -0.321 | not significant |
| graph-res vs node-res | +0.0048 | 0.6367 | 0.8125 | +0.228 | not significant |
| node-cross vs graph-cross | +0.0067 | 0.3337 | 0.5625 | +0.491 | not significant |

