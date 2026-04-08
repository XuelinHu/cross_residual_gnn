# Cross-Residual Graph Neural Networks 中文说明

[English README](./README.md)

本仓库是 CR-GNN（Cross-Residual Graph Neural Networks）图分类研究代码，核心目标是在统一训练协议下，对比普通堆叠、残差复用和交叉残差的信息传递方式。

当前主训练入口是 [geomatric/graph_classify_v3.py](./geomatric/graph_classify_v3.py)。共享的数据集定义、日志工具和钉钉通知工具都已经放到 [geomatric/](./geomatric) 包下面。

## 功能概览

当前主线支持 5 个图分类结构：

- `PlainGNN`
- `NodeResGNN`
- `NodeCrossGNN`
- `GraphResGNN`
- `GraphCrossGNN`

当前代码还包含：

- 分层 5 折交叉验证
- 训练集内部验证集切分
- `ReduceLROnPlateau`
- 梯度裁剪
- TensorBoard 记录
- JSON 结果快照导出
- 批量实验、汇总、画图和论文产物生成脚本

旧版 V2 相关内容已经归档到 [achivement_V2/](./achivement_V2)。

## 仓库路径总览

```text
cross_residual_gnn/
├── achivement_V2/               # 旧版 V2 代码与记录
├── data/                        # 本地数据集目录，Git 忽略
├── figures/                     # 导出的实验图片
├── geomatric/                   # 当前有效 Python 包
├── logs/                        # 训练 JSON 快照，Git 忽略
├── md/                          # Markdown / TeX 汇总文件
├── paper/                       # 论文 LaTeX 与图片
├── py/                          # 批处理与分析脚本
├── records/                     # 文本记录，Git 忽略
├── runs/                        # TensorBoard 日志，Git 忽略
├── tmp/                         # 本地临时文件，Git 忽略
├── README.md
└── README-CN.md
```

## 核心文件路径

### 核心包路径

- [geomatric/__init__.py](./geomatric/__init__.py)：包入口
- [geomatric/graph_classify_v3.py](./geomatric/graph_classify_v3.py)：主训练 / 主评估入口
- [geomatric/experiment_catalog.py](./geomatric/experiment_catalog.py)：数据集分组与元信息
- [geomatric/logging_config.py](./geomatric/logging_config.py)：统一日志配置
- [geomatric/dingtalk_util.py](./geomatric/dingtalk_util.py)：钉钉 Markdown / ActionCard 通知工具

### 运行脚本路径

- [py/run_paper_experiments.py](./py/run_paper_experiments.py)：全量论文实验批跑
- [py/run_sensitivity_experiments.py](./py/run_sensitivity_experiments.py)：参数敏感性扫描
- [py/run_enzymes_tuned_cross.py](./py/run_enzymes_tuned_cross.py)：ENZYMES 调优交叉模型实验
- [py/summarize_paper_experiments.py](./py/summarize_paper_experiments.py)：读取最新日志并汇总
- [py/generate_all_result_reports.py](./py/generate_all_result_reports.py)：生成总报告文本和 TeX
- [py/generate_suite_analysis_figures.py](./py/generate_suite_analysis_figures.py)：生成全套对比图
- [py/generate_dataset_statistics_report.py](./py/generate_dataset_statistics_report.py)：生成数据集统计
- [py/generate_sensitivity_reports.py](./py/generate_sensitivity_reports.py)：生成敏感性分析报告
- [py/generate_exp_figures.py](./py/generate_exp_figures.py)：根据记录文件生成图表
- [py/generate_method_figure.py](./py/generate_method_figure.py)：方法图辅助脚本
- [py/generate_topic_tables.py](./py/generate_topic_tables.py)：面向主题的 LaTeX 表格
- [py/export_analysis_artifacts.py](./py/export_analysis_artifacts.py)：导出单次训练分析产物
- [py/organize_references_from_corpus.py](./py/organize_references_from_corpus.py)：参考文献整理
- [py/plot_style.py](./py/plot_style.py)：绘图样式工具

### 结果快照与输出路径

这里的“快照”主要指训练过程和实验汇总生成的结果文件：

- `logs/train_<dataset>_<model>_<operator>_fold<k>__<timestamp>.json`
  日志目录根路径：[logs/](./logs)
- `logs/dataset_stats_<dataset>__<timestamp>.json`
  日志目录根路径：[logs/](./logs)
- `records/suite_<suite_name>_<dataset>__<timestamp>.txt`
  记录目录根路径：[records/](./records)
- `runs/<experiment_name>_<timestamp>/events.out.tfevents.*`
  TensorBoard 根路径：[runs/](./runs)
- `figures/analysis/*`
  分析产物目录：[figures/analysis/](./figures/analysis)
- `figures/exp/*`
  实验图目录：[figures/exp/](./figures/exp)
- `paper/figures/exp/*`
  论文图目录：[paper/figures/exp/](./paper/figures/exp)

### Markdown / TeX 汇总文件路径

- [md/all_results_summary.txt](./md/all_results_summary.txt)
- [md/all_exp_tables.tex](./md/all_exp_tables.tex)
- [md/all_exp_tables_appendix.tex](./md/all_exp_tables_appendix.tex)
- [md/all_ablation_analysis.md](./md/all_ablation_analysis.md)
- [md/sensitivity_summary.md](./md/sensitivity_summary.md)
- [md/parameter_sensitivity_analysis.md](./md/parameter_sensitivity_analysis.md)
- [md/dataset_statistics_summary.md](./md/dataset_statistics_summary.md)
- [md/dataset_statistics_summary.json](./md/dataset_statistics_summary.json)
- [md/dataset_statistics_tables.tex](./md/dataset_statistics_tables.tex)
- [md/topic_results_summary.txt](./md/topic_results_summary.txt)
- [md/topic_exp_tables.tex](./md/topic_exp_tables.tex)
- [md/cross_advantage_summary.md](./md/cross_advantage_summary.md)
- [md/formal_experiment_protocol.md](./md/formal_experiment_protocol.md)
- [md/final_implementation_todo.md](./md/final_implementation_todo.md)
- [md/paper_gap_checklist.md](./md/paper_gap_checklist.md)
- [md/reference_logic_map.md](./md/reference_logic_map.md)
- [md/frontiers_topic_alignment.md](./md/frontiers_topic_alignment.md)
- [md/topic_aligned_dataset_shortlist.md](./md/topic_aligned_dataset_shortlist.md)
- [md/SESSION_RECORD.md](./md/SESSION_RECORD.md)

### 图像文件路径

当前实验图：

- [figures/exp/fig1_full_suite_results.png](./figures/exp/fig1_full_suite_results.png)
- [figures/exp/fig1_full_suite_results.pdf](./figures/exp/fig1_full_suite_results.pdf)
- [figures/exp/fig2_cross_advantage_heatmap.png](./figures/exp/fig2_cross_advantage_heatmap.png)
- [figures/exp/fig2_cross_advantage_heatmap.pdf](./figures/exp/fig2_cross_advantage_heatmap.pdf)
- [figures/exp/fig3_rank_winner_summary.png](./figures/exp/fig3_rank_winner_summary.png)
- [figures/exp/fig3_rank_winner_summary.pdf](./figures/exp/fig3_rank_winner_summary.pdf)
- [figures/exp/fig4_topic_focus_results.png](./figures/exp/fig4_topic_focus_results.png)
- [figures/exp/fig4_topic_focus_results.pdf](./figures/exp/fig4_topic_focus_results.pdf)
- [figures/exp/fig5_protein_package_summary.png](./figures/exp/fig5_protein_package_summary.png)
- [figures/exp/fig5_protein_package_summary.pdf](./figures/exp/fig5_protein_package_summary.pdf)

论文图片目录：

- [paper/figures/](./paper/figures)
- [paper/figures/exp/](./paper/figures/exp)
- [paper/figures/cr_gnn_schematic.png](./paper/figures/cr_gnn_schematic.png)
- [paper/figures/task_model_comparison.png](./paper/figures/task_model_comparison.png)

## 数据集路径

仓库本身不提交原始数据集，但本地数据路径已经固定：

- [data/](./data)：本地数据根目录
- `data/TUDataset/`：TU 数据集下载目录
- `data/OGB/`：OGB 图属性预测数据集下载目录
- [data/.gitkeep](./data/.gitkeep)：占位文件

当前在 [geomatric/experiment_catalog.py](./geomatric/experiment_catalog.py) 中定义的数据集分组：

- 主数据集：`PROTEINS`、`DD`、`ENZYMES`
- 补充数据集：`MUTAG`、`AIDS`、`Mutagenicity`

## 论文文件路径

- [paper/main.tex](./paper/main.tex)
- [paper/main.pdf](./paper/main.pdf)
- [paper/references.bib](./paper/references.bib)
- [paper/compile.bat](./paper/compile.bat)
- [paper/README.md](./paper/README.md)
- [paper/paper_corpus_merged.json](./paper/paper_corpus_merged.json)
- [paper/sections/abstract.tex](./paper/sections/abstract.tex)
- [paper/sections/introduction.tex](./paper/sections/introduction.tex)
- [paper/sections/related_work.tex](./paper/sections/related_work.tex)
- [paper/sections/task_definition.tex](./paper/sections/task_definition.tex)
- [paper/sections/proposed_model.tex](./paper/sections/proposed_model.tex)
- [paper/sections/datasets.tex](./paper/sections/datasets.tex)
- [paper/sections/experiments.tex](./paper/sections/experiments.tex)
- [paper/sections/conclusion.tex](./paper/sections/conclusion.tex)
- [paper/sections/appendix.tex](./paper/sections/appendix.tex)

## 常用运行命令

### 1. 单次训练

```bash
python -m geomatric.graph_classify_v3 \
  --mode single \
  --ds PROTEINS \
  --gname NodeCrossGNN \
  --name GCNConv \
  --ep 240 \
  --patience 80 \
  --lr 0.003 \
  --weight_decay 5e-5 \
  --drop 0.2 \
  --dim 64 \
  --h_layer 4 \
  --batch_size 32 \
  --grad_clip 2.0 \
  --tensorboard
```

也可以直接运行脚本路径：

```bash
python geomatric/graph_classify_v3.py --mode single --ds PROTEINS
```

### 2. 全量实验批跑

```bash
python py/run_paper_experiments.py --dataset_group all --max_workers 6 --tensorboard
```

支持的数据集分组：

- `main`
- `topic`
- `extended`
- `all`

### 3. 汇总最新实验日志

```bash
python py/summarize_paper_experiments.py --dataset_group all
```

### 4. 生成总报告和图

```bash
python py/generate_all_result_reports.py
python py/generate_suite_analysis_figures.py
```

### 5. 运行参数敏感性分析

```bash
python py/run_sensitivity_experiments.py --fold 0 --max_workers 6
python py/generate_sensitivity_reports.py
```

### 6. 生成数据集统计

```bash
python py/generate_dataset_statistics_report.py
```

### 7. 查看 TensorBoard

```bash
tensorboard --logdir runs --port 6006
```

## 当前论文层面的结论

基于当前归档结果：

- residual 系列在当前主题相关的 TU 数据集上更稳定
- cross-residual 系列在部分补充数据集上仍然有选择性优势

因此目前证据支持的是“交叉残差的收益具有数据集依赖性”，而不是“cross 一定优于 residual”。

## 当前全套实验获胜模型

- `MUTAG`: `NodeCrossGNN`
- `PROTEINS`: `GraphResGNN`
- `DD`: `NodeResGNN`
- `ENZYMES`: `GraphResGNN`
- `AIDS`: `GraphResGNN`
- `Mutagenicity`: `NodeCrossGNN`

## 备注

- `logs/`、`runs/`、`records/`、`data/`、`tmp/` 这类运行期目录默认被 Git 忽略。
- `py/` 下分析脚本统一依赖 [geomatric/experiment_catalog.py](./geomatric/experiment_catalog.py) 中的数据集定义。
- 仓库不会提交下载后的原始数据集内容，README 只记录约定的数据目录路径。
