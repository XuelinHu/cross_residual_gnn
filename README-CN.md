# Cross-Residual Graph Neural Networks 中文说明

[English README](./README.md)

本仓库包含 CR-GNN 研究的代码、实验产物和论文源码。仓库现已收口到最终 `V3` 流程，当前 paper-facing 版本以 [`paper/`](./paper) 下的稿件为准。

## 最终 V3 状态

- 当前训练入口：[`geomatric/graph_classify_v3.py`](./geomatric/graph_classify_v3.py)
- 最终流程入口：[`py/run_final_v3_pipeline.py`](./py/run_final_v3_pipeline.py)
- 最终实验产物：`logs/V3`、`records/V3`、`runs/V3`
- 项目索引：[`md/FINAL_PROJECT_INDEX.md`](./md/FINAL_PROJECT_INDEX.md)
- 论文主文件：[`paper/main.tex`](./paper/main.tex)

现在只应把 `V3` 视为当前有效版本。更早的补跑记录、阶段性整理笔记和过时归档目录不再属于主流程。

## 仓库结构

```text
cross_residual_gnn/
├── data/                      # 本地数据集，不提交
├── figures/                   # 导出的实验图片
├── geomatric/                 # 当前有效 Python 包
├── logs/                      # 运行日志与 V1/V2/V3 归档输出
├── md/                        # 汇总、表格与论文分析说明
├── paper/                     # 当前 LaTeX 论文与论文图片
├── py/                        # 批处理、分析、汇总脚本
├── records/                   # 文本汇总与 V1/V2/V3 归档记录
├── runs/                      # TensorBoard 日志
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

汇总与导出报告：

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

论文源码：

- [`paper/main.tex`](./paper/main.tex)
- [`paper/sections/01_introduction_peerj.tex`](./paper/sections/01_introduction_peerj.tex)
- [`paper/sections/02_methods_peerj.tex`](./paper/sections/02_methods_peerj.tex)
- [`paper/sections/03_results_peerj.tex`](./paper/sections/03_results_peerj.tex)
- [`paper/sections/04_discussion_peerj.tex`](./paper/sections/04_discussion_peerj.tex)
- [`paper/sections/05_conclusions_peerj.tex`](./paper/sections/05_conclusions_peerj.tex)

已经存在但尚未充分纳入正文的论文图：

- [`paper/figures/exp/fig1_full_suite_results.pdf`](./paper/figures/exp/fig1_full_suite_results.pdf)
- [`paper/figures/exp/fig2_cross_advantage_heatmap.pdf`](./paper/figures/exp/fig2_cross_advantage_heatmap.pdf)
- [`paper/figures/exp/fig4_topic_focus_results.pdf`](./paper/figures/exp/fig4_topic_focus_results.pdf)
- [`paper/figures/exp/fig5_protein_package_summary.pdf`](./paper/figures/exp/fig5_protein_package_summary.pdf)

## 论文当前状态

当前稿件更准确的定位是：

- 一个 biomolecular graph classification 方法论文
- 以 `PROTEINS`、`DD`、`ENZYMES` 作为 biological core
- 以 `MUTAG`、`AIDS`、`Mutagenicity` 作为 supplementary robustness 数据集

这份稿件还不是完全收口的最终版。当前源码仍需要：

- 围绕 V3 主线进一步收束 scope
- 让文字结论逐条对齐当前表格
- 补齐 methods 中未闭合的数学定义
- 把已有主结果图重新接回正文

关于选题匹配和改稿判断，优先看 [`md/frontiers_topic_alignment.md`](./md/frontiers_topic_alignment.md)。

## 数据集分组

定义位于 [`geomatric/experiment_catalog.py`](./geomatric/experiment_catalog.py)：

- biological core：`PROTEINS`、`DD`、`ENZYMES`
- supplementary robustness：`MUTAG`、`AIDS`、`Mutagenicity`

## 备注

- 仓库不提交原始数据集。
- 运行目录可能较大，并保留 V1/V2/V3 历史输出。
- 修改论文结论时，应以 V3 表格和图为准，不要再参考早期 `LATEST` 风格说明。
