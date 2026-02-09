"""
Generate figures and tables for experiments chapter
Run this script to create:
1. Figures (PDF/PNG) in figures/exp_/
2. LaTeX table code in md/exp_tables.tex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# Set style for publication-quality figures
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'
rcParams['savefig.bbox'] = 'tight'

# Create output directories
os.makedirs('figures/exp', exist_ok=True)
os.makedirs('md', exist_ok=True)

# Load data
df = pd.read_excel('../records/v3result.xlsx')

# ============================================================================
# FIGURE 1: Bar chart of mean accuracy by model (all datasets)
# ============================================================================
fig1, ax = plt.subplots(figsize=(8, 5))

model_order = ['BlockGNN', 'ResBlockGnn', 'CrossBlockGnn',
               'GraphBlockGnn', 'ResGraphBlockGnn', 'CrossGraphBlockGnn']

# Calculate mean accuracy across all datasets for each model
mean_acc = df.groupby('gm')['acc'].mean().reindex(model_order)
std_acc = df.groupby('gm')['acc'].std().reindex(model_order)

# Plot bar chart with error bars
bars = ax.bar(range(len(model_order)), mean_acc, yerr=std_acc,
              capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2,
              color=['#e74c3c', '#3498db', '#3498db', '#2ecc71', '#2ecc71', '#9b59b6'])

ax.set_xticks(range(len(model_order)))
ax.set_xticklabels([f'{m}\n(n={len(df[df["gm"]==m])})' for m in model_order],
                   fontsize=9, rotation=0, ha='center')
ax.set_ylabel('Mean Accuracy', fontsize=11, fontweight='bold')
ax.set_xlabel('Model Architecture', fontsize=11, fontweight='bold')
ax.set_ylim([0.65, 0.82])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_title('Average Performance Across All Datasets', fontsize=12, fontweight='bold', pad=15)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, mean_acc, std_acc)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/exp/fig1_average_performance.pdf')
plt.savefig('figures/exp/fig1_average_performance.png')
plt.close()
print("[OK] Figure 1 saved: Average performance bar chart")

# ============================================================================
# FIGURE 2: Depth sensitivity (line plot)
# ============================================================================
fig2, ax = plt.subplots(figsize=(8, 5))

# Filter data for MUTAG, GCN, dim=32
mutag_gcn_32 = df[(df['ds'] == 'MUTAG') & (df['model'] == 'GCNConv') & (df['dim'] == 32)]

# Calculate mean accuracy for each (model, depth) combination
depth_data = mutag_gcn_32.groupby(['gm', 'h'])['acc'].mean().unstack()

# Plot line plot
markers = {'BlockGNN': 'o', 'ResBlockGnn': 's', 'CrossBlockGnn': '^',
          'GraphBlockGnn': 'd', 'ResGraphBlockGnn': 'v', 'CrossGraphBlockGnn': 'p'}
colors = {'BlockGNN': '#e74c3c', 'ResBlockGnn': '#3498db', 'CrossBlockGnn': '#3498db',
          'GraphBlockGnn': '#2ecc71', 'ResGraphBlockGnn': '#2ecc71', 'CrossGraphBlockGnn': '#9b59b6'}

for model in model_order:
    if model in depth_data.index:
        ax.plot(range(1, 6), depth_data.loc[model],
               marker=markers[model], color=colors[model],
               linewidth=2, markersize=8, label=model)

ax.set_xticks(range(1, 6))
ax.set_xticklabels([f'h={i}' for i in range(1, 6)], fontsize=10)
ax.set_xlabel('Number of Hidden Layers', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_ylim([0.55, 0.85])
ax.grid(alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_title('Depth Sensitivity on MUTAG (GCN, dim=32)', fontsize=12, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/exp/fig2_depth_sensitivity.pdf')
plt.savefig('figures/exp/fig2_depth_sensitivity.png')
plt.close()
print("[OK] Figure 2 saved: Depth sensitivity line plot")

# ============================================================================
# FIGURE 3: Box plot of accuracy distribution (5 folds)
# ============================================================================
fig3, ax = plt.subplots(figsize=(10, 6))

# Prepare data for box plot (use MUTAG as example)
box_data = []
box_labels = []
for model in model_order:
    model_df = df[(df['gm'] == model) & (df['ds'] == 'MUTAG') &
                   (df['model'] == 'GCNConv') & (df['dim'] == 32) & (df['h'] == 2)]
    if len(model_df) > 0:
        # Extract acc0-acc4 values
        fold_accs = []
        for _, row in model_df.iterrows():
            fold_accs.extend([row['acc0'], row['acc1'], row['acc2'], row['acc3'], row['acc4']])
        box_data.append(fold_accs)
        box_labels.append(model)

# Create box plot
bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                showmeans=True, meanline=True)

# Color boxes
colors_box = ['#e74c3c', '#3498db', '#3498db', '#2ecc71', '#2ecc71', '#9b59b6']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_xlabel('Model Architecture', fontsize=11, fontweight='bold')
ax.set_ylim([0.5, 0.9])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_title('5-Fold CV Distribution on MUTAG (GCN, h=2, dim=32)', fontsize=12, fontweight='bold', pad=15)
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig('figures/exp/fig3_boxplot_distribution.pdf')
plt.savefig('figures/exp/fig3_boxplot_distribution.png')
plt.close()
print("[OK] Figure 3 saved: Box plot distribution")

# ============================================================================
# FIGURE 4: Heatmap of model × dataset performance
# ============================================================================
fig4, ax = plt.subplots(figsize=(8, 6))

# Create pivot table: models as rows, datasets as columns
heatmap_data = df.pivot_table(values='acc', index='gm', columns='ds', aggfunc='mean')
heatmap_data = heatmap_data.reindex(model_order)

# Plot heatmap
im = ax.imshow(heatmap_data.values, cmap='YlGnBu', aspect='auto', vmin=0.5, vmax=1.0)

# Set ticks
ax.set_xticks(np.arange(len(heatmap_data.columns)))
ax.set_yticks(np.arange(len(heatmap_data.index)))
ax.set_xticklabels(heatmap_data.columns, fontsize=10)
ax.set_yticklabels(heatmap_data.index, fontsize=10)

# Add text annotations
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        text = ax.text(j, i, f'{heatmap_data.values[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
ax.set_ylabel('Model Architecture', fontsize=11, fontweight='bold')
ax.set_title('Model Performance Heatmap (Mean Accuracy)', fontsize=12, fontweight='bold', pad=15)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Accuracy', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/exp/fig4_heatmap_performance.pdf')
plt.savefig('figures/exp/fig4_heatmap_performance.png')
plt.close()
print("[OK] Figure 4 saved: Performance heatmap")

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)

# ============================================================================
# Generate LaTeX Tables
# ============================================================================
print("\nGenerating LaTeX tables...")

latex_tables = []

# ============================================================================
# TABLE 1: Main Results
# ============================================================================
table1_latex = r"""
% Table 1: Main Results on All Datasets
\begin{table}[t]
\centering
\caption{Graph classification accuracy (mean $\pm$ std over 5-fold CV) on TUDataset benchmarks. Best results in bold.}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
Model & MUTAG & DD & MSRC\_9 & AIDS \\
\midrule
"""

# Get best results for each dataset
for model in model_order:
    model_df = df[df['gm'] == model]
    row_data = []
    for ds in ['MUTAG', 'DD', 'MSRC_9', 'AIDS']:
        ds_df = model_df[model_df['ds'] == ds]
        if len(ds_df) > 0:
            best = ds_df.loc[ds_df['acc'].idxmax()]
            acc = best['acc']
            std = best['acc_std_dev']
            row_data.append(f'{acc:.3f} $\\pm$ {std:.4f}')
        else:
            row_data.append('--')

    # Bold the best result for each dataset
    is_best = False
    for i, ds in enumerate(['MUTAG', 'DD', 'MSRC_9', 'AIDS']):
        ds_df = df[df['ds'] == ds]
        if len(ds_df) > 0:
            best_acc = ds_df['acc'].max()
            if abs(float(row_data[i].split('$\\pm$')[0]) - best_acc) < 0.001:
                row_data[i] = f'\\textbf{{{row_data[i]}}}'

    model_name = model.replace('_', '\\_')
    table1_latex += f"{model_name} & {' & '.join(row_data)} \\\\\n"

table1_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

latex_tables.append(('Table 1: Main Results', table1_latex))

# ============================================================================
# TABLE 2: Ablation Study - Impact of Residual Mechanism
# ============================================================================
table2_latex = r"""
% Table 2: Ablation Study on Residual Mechanisms (MUTAG, GCN, dim=64)
\begin{table}[t]
\centering
\caption{Ablation study on different residual mechanisms. Results on MUTAG with GCN operator and dim=64.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Residual Type & h=1 & h=2 & h=3 & h=4 & h=5 \\
\midrule
"""

# Filter for MUTAG, GCN, dim=64
mutag_gcn_64 = df[(df['ds'] == 'MUTAG') & (df['model'] == 'GCNConv') & (df['dim'] == 64)]

ablation_models = [
    ('None', 'BlockGNN'),
    ('Intra-branch', 'ResBlockGnn'),
    ('Node-level cross', 'CrossBlockGnn'),
    ('Sequential graph', 'GraphBlockGnn'),
    ('Sequential graph residual', 'ResGraphBlockGnn'),
    ('Graph-level cross', 'CrossGraphBlockGnn'),
]

for residual_name, model_name in ablation_models:
    model_df = mutag_gcn_64[mutag_gcn_64['gm'] == model_name]
    row_data = []
    for h in [1, 2, 3, 4, 5]:
        h_df = model_df[model_df['h'] == h]
        if len(h_df) > 0:
            acc = h_df['acc'].iloc[0]
            row_data.append(f'{acc:.3f}')
        else:
            row_data.append('--')

    table2_latex += f"{residual_name} & {' & '.join(row_data)} \\\\\n"

table2_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

latex_tables.append(('Table 2: Ablation Study', table2_latex))

# ============================================================================
# TABLE 3: Efficiency Comparison
# ============================================================================
table3_latex = r"""
% Table 3: Efficiency Comparison (MUTAG, GCN, dim=32, h=2)
\begin{table}[t]
\centering
\caption{Computational efficiency comparison. Execution time in seconds for 5-fold CV.}
\label{tab:efficiency}
\begin{tabular}{lcc}
\toprule
Model & Time (s) & Relative Overhead \\
\midrule
"""

# Get timing data for MUTAG, GCN, dim=32, h=2
timing_df = df[(df['ds'] == 'MUTAG') & (df['model'] == 'GCNConv') &
               (df['dim'] == 32) & (df['h'] == 2)]

baseline_time = None
for model in ['BlockGNN', 'ResBlockGnn', 'CrossBlockGnn', 'CrossGraphBlockGnn']:
    model_df = timing_df[timing_df['gm'] == model]
    if len(model_df) > 0:
        time = model_df['execution_time'].iloc[0]
        if baseline_time is None:
            baseline_time = time
        overhead = (time / baseline_time - 1) * 100
        model_name = model.replace('_', '\\_')
        table3_latex += f"{model_name} & {time:.1f} & {overhead:+.1f}\\% \\\\\n"

table3_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

latex_tables.append(('Table 3: Efficiency Comparison', table3_latex))

# Save all tables to file
with open('md/exp_tables.tex', 'w', encoding='utf-8') as f:
    for title, latex in latex_tables:
        f.write(f"% {title}\n")
        f.write(latex)
        f.write("\n\n")

print(f"[OK] Generated {len(latex_tables)} LaTeX tables")
for title, _ in latex_tables:
    print(f"  - {title}")

print("\n" + "="*60)
print("Table generation completed!")
print("="*60)
