# Cross-Residual Graph Neural Networks 中文说明

[English README](./README.md)

本仓库包含 CR-GNN 研究的代码、实验产物和论文源码。仓库现已收口到最终 `V3` 流程，并整理出可直接交付的投稿稿件包，入口位于 [`paper/`](./paper)。

## 投稿入口

- 英文主稿：[`paper/main.tex`](./paper/main.tex)
- 中文主稿：[`paper/main_chinese.tex`](./paper/main_chinese.tex)
- 英文 PDF：[`paper/main.pdf`](./paper/main.pdf)
- 中文 PDF：[`paper/main_chinese.pdf`](./paper/main_chinese.pdf)
- 投稿清单：[`paper/SUBMISSION_MANIFEST.md`](./paper/SUBMISSION_MANIFEST.md)

如果只关心论文交付，从 `paper/` 开始即可；如果需要复现实验表格和图，再进入下面的 `V3` 流程。

## 最终 V3 状态

- 当前训练入口：[`geomatric/graph_classify_v3.py`](./geomatric/graph_classify_v3.py)
- 最终流程入口：[`py/run_final_v3_pipeline.py`](./py/run_final_v3_pipeline.py)
- 当前有效表格与统计：[`md/all_exp_tables.tex`](./md/all_exp_tables.tex)、[`md/statistical_tests_main.tex`](./md/statistical_tests_main.tex)、[`md/statistical_tests_supp.tex`](./md/statistical_tests_supp.tex)
- 当前有效图片：`figures/exp/*_V3.pdf` 与 `paper/figures/exp/*_V3.pdf`
- 项目索引：[`md/FINAL_PROJECT_INDEX.md`](./md/FINAL_PROJECT_INDEX.md)

现在只应把 `V3` 视为当前有效版本。`V1`、`V2` 目录仅保留为历史归档与可复现参考，不再属于主流程。

## 仓库结构

```text
cross_residual_gnn/
├── data/                      # 本地数据集，不提交
├── figures/                   # 导出的实验图片
├── geomatric/                 # 当前有效 Python 包
├── logs/                      # 历史运行日志（归档 / 可复现）
├── md/                        # 最终表格、工作笔记与历史汇总
├── paper/                     # 当前 LaTeX 论文与论文图片
├── py/                        # 批处理、分析与导出脚本
├── records/                   # 最终分析 JSON 与版本记录
├── runs/                      # TensorBoard 归档
├── README.md
└── README-CN.md
```

## 环境

按仓库约定，Python 命令默认使用 Conda 环境 `pyg`：

```bash
conda activate pyg
```

推荐依赖：

- Python `3.10+`
- PyTorch
- PyTorch Geometric
- `tensorboard`
- `matplotlib`、`seaborn`、`pandas`、`openpyxl`

## 常用命令

单次训练：

```bash
conda activate pyg
python -m geomatric.graph_classify_v3 --mode single --ds PROTEINS
```

最终 V3 流程：

```bash
conda activate pyg
python py/run_final_v3_pipeline.py
python py/run_final_v3_pipeline.py --steps consolidate summarize reports figures
```

全量实验批跑：

```bash
conda activate pyg
python py/run_paper_experiments.py --dataset_group all --max_workers 6 --tensorboard
```

汇总并导出最终报告产物：

```bash
conda activate pyg
python py/summarize_paper_experiments.py --dataset_group all
python py/generate_all_result_reports.py
python py/generate_suite_analysis_figures.py
```

敏感性分析与数据统计：

```bash
conda activate pyg
python py/run_sensitivity_experiments.py --fold 0 --max_workers 6
python py/generate_sensitivity_reports.py
python py/generate_dataset_statistics_report.py
```

## 关键文件

代码与配置：

- [`geomatric/graph_classify_v3.py`](./geomatric/graph_classify_v3.py)
- [`geomatric/experiment_catalog.py`](./geomatric/experiment_catalog.py)
- [`geomatric/experiment_paths.py`](./geomatric/experiment_paths.py)
- [`py/run_final_v3_pipeline.py`](./py/run_final_v3_pipeline.py)

实验汇总：

- [`md/EXPERIMENT_INDEX.md`](./md/EXPERIMENT_INDEX.md)
- [`md/FINAL_PROJECT_INDEX.md`](./md/FINAL_PROJECT_INDEX.md)
- [`md/V3_residual_summary.md`](./md/V3_residual_summary.md)
- [`md/all_exp_tables_V3.tex`](./md/all_exp_tables_V3.tex)
- [`md/all_results_summary_V3.txt`](./md/all_results_summary_V3.txt)
- [`md/frontiers_topic_alignment.md`](./md/frontiers_topic_alignment.md)
- [`records/experiment_versions.json`](./records/experiment_versions.json)

论文源码：

- [`paper/main.tex`](./paper/main.tex)
- [`paper/main_chinese.tex`](./paper/main_chinese.tex)
- [`paper/sections/01_introduction_peerj.tex`](./paper/sections/01_introduction_peerj.tex)
- [`paper/sections/02_methods_peerj.tex`](./paper/sections/02_methods_peerj.tex)
- [`paper/sections/03_results_peerj.tex`](./paper/sections/03_results_peerj.tex)
- [`paper/sections/04_discussion_peerj.tex`](./paper/sections/04_discussion_peerj.tex)
- [`paper/sections/05_conclusions_peerj.tex`](./paper/sections/05_conclusions_peerj.tex)
- [`paper/sections_cn/`](./paper/sections_cn)

当前有效的论文主图：

- [`paper/figures/exp/fig1_full_suite_results_V3.pdf`](./paper/figures/exp/fig1_full_suite_results_V3.pdf)
- [`paper/figures/exp/fig2_cross_advantage_heatmap_V3.pdf`](./paper/figures/exp/fig2_cross_advantage_heatmap_V3.pdf)
- [`paper/figures/exp/fig4_topic_focus_results_V3.pdf`](./paper/figures/exp/fig4_topic_focus_results_V3.pdf)
- [`paper/figures/exp/fig5_protein_package_summary_V3.pdf`](./paper/figures/exp/fig5_protein_package_summary_V3.pdf)

## 论文当前状态

当前稿件已收口为 protein-oriented benchmark 论文：

- 主 biological evidence：`PROTEINS` 与 `DD`
- supporting biological stress test：`ENZYMES`
- supplementary robustness：`MUTAG`、`AIDS`、`Mutagenicity`

英文和中文 PDF 都已经可以正常编译。当前剩余的只是版式级警告，例如浮动体位置、少量宽表格，以及两张结构图的 PDF 版本提示。

## 数据集分组

定义位于 [`geomatric/experiment_catalog.py`](./geomatric/experiment_catalog.py)：

- biological core：`PROTEINS`、`DD`、`ENZYMES`
- supplementary robustness：`MUTAG`、`AIDS`、`Mutagenicity`

## 归档策略

- 仓库不提交原始数据集。
- `logs/V1`、`logs/V2`、`runs/V1`、`runs/V2` 仅作为历史归档保留。
- `logs/V3` 与 `runs/V3` 属于复现实验记录，不是论文源码。
- `md/` 目录中包含部分历史笔记或探索性文件，真正有效的来源应以当前无后缀主表、V3 图和 `paper/` 投稿清单为准。

## 备注

- 修改论文结论时，应以当前 `V3` 表格和图片为准，不要再参考早期 `LATEST` 风格说明。
- 如需对外移交仓库，建议直接从 [`paper/SUBMISSION_MANIFEST.md`](./paper/SUBMISSION_MANIFEST.md) 开始。
