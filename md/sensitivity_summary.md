# Sensitivity Summary

## PROTEINS
### NodeCrossGNN
- `h_layer`: best `3` with acc `0.69955`; values: 3 -> 0.69955, 4 -> 0.67713, 5 -> 0.67713
- `drop`: best `0.2` with acc `0.67713`; values: 0.2 -> 0.67713, 0.3 -> 0.65022, 0.5 -> 0.66368
- `lr`: best `0.003` with acc `0.67713`; values: 0.002 -> 0.66816, 0.003 -> 0.67713, 0.005 -> 0.66368

### GraphCrossGNN
- `h_layer`: best `4` with acc `0.69955`; values: 3 -> 0.68161, 4 -> 0.69955, 5 -> 0.67713
- `drop`: best `0.2` with acc `0.70404`; values: 0.2 -> 0.70404, 0.3 -> 0.69955, 0.5 -> 0.69955
- `lr`: best `0.002` with acc `0.69955`; values: 0.002 -> 0.69955, 0.003 -> 0.69955, 0.005 -> 0.67265

## DD
### NodeCrossGNN
- `h_layer`: best `4` with acc `0.72152`; values: 3 -> 0.71308, 4 -> 0.72152, 5 -> 0.71730
- `drop`: best `0.5` with acc `0.71730`; values: 0.2 -> 0.70042, 0.3 -> 0.71308, 0.5 -> 0.71730
- `lr`: best `0.005` with acc `0.74262`; values: 0.002 -> 0.73418, 0.003 -> 0.71308, 0.005 -> 0.74262

### GraphCrossGNN
- `h_layer`: best `3` with acc `0.74684`; values: 3 -> 0.74684, 4 -> 0.67511, 5 -> 0.72152
- `drop`: best `0.5` with acc `0.71730`; values: 0.2 -> 0.67511, 0.3 -> 0.71308, 0.5 -> 0.71730
- `lr`: best `0.005` with acc `0.72574`; values: 0.002 -> 0.67511, 0.003 -> 0.71730, 0.005 -> 0.72574

## ENZYMES
### NodeCrossGNN
- `h_layer`: best `3` with acc `0.23333`; values: 3 -> 0.23333, 4 -> 0.23333, 5 -> 0.19167
- `drop`: best `0.2` with acc `0.25000`; values: 0.2 -> 0.25000, 0.3 -> 0.23333, 0.5 -> 0.18333
- `lr`: best `0.003` with acc `0.23333`; values: 0.002 -> 0.16667, 0.003 -> 0.23333, 0.005 -> 0.22500

### GraphCrossGNN
- `h_layer`: best `3` with acc `0.20000`; values: 3 -> 0.20000, 4 -> 0.18333, 5 -> 0.15833
- `drop`: best `0.2` with acc `0.20833`; values: 0.2 -> 0.20833, 0.3 -> 0.18333, 0.5 -> 0.16667
- `lr`: best `0.002` with acc `0.20833`; values: 0.002 -> 0.20833, 0.003 -> 0.18333, 0.005 -> 0.17500
