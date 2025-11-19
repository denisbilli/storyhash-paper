"""
Generate all figures and tables for StoryHash paper.

Uses ONLY real data from:
- docs/PHASE_6_NOTES.md
- reports/robustness_full/robustness_results_aggregated.json
- reports/ablation_clip_only/robustness_results_aggregated.json
- reports/ablation_action_only/robustness_results_aggregated.json

Generates:
- Figure 1: Pipeline diagram
- Figure 2: 3-way ablation summary (bar chart)
- Figure 3: Similarity delta (bar chart)
- Table 1: Per-transformation results (14 rows × 6 columns)
- Figure 4: Vertical flip bias (bar chart)
- Figure 5: Architecture diagram (527D breakdown)

Outputs:
- PDF (vector) for LaTeX
- PNG 300dpi for preview
- LaTeX table code
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Real data from ablation study
FUSION_DATA = {
    'recall_1': 98.6,
    'recall_3': 99.7,
    'similarity': 0.9748,
}

CLIP_ONLY_DATA = {
    'recall_1': 98.2,
    'recall_3': 99.4,
    'similarity': 0.6996,
}

ACTION_ONLY_DATA = {
    'recall_1': 1.1,
    'recall_3': 3.4,
    'similarity': 0.7500,
}

# Per-transformation data (from PHASE_6_NOTES.md)
TRANSFORMATIONS = [
    'crop_10', 'crop_20', 'rotate_5', 'rotate_10',
    'bright_20', 'bright_40', 'compress_50', 'compress_25',
    'speed_0.9', 'speed_1.1', 'flip_h', 'flip_v',
    'watermark', 'combined'
]

PER_TRANSFORM_DATA = {
    'crop_10': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9881, 'clip_sim': 0.7250, 'action_sim': 0.7500},
    'crop_20': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9842, 'clip_sim': 0.7049, 'action_sim': 0.7500},
    'rotate_5': {'fusion_r1': 98.9, 'clip_r1': 96.6, 'action_r1': 1.1, 'fusion_sim': 0.9844, 'clip_sim': 0.6955, 'action_sim': 0.7500},
    'rotate_10': {'fusion_r1': 98.9, 'clip_r1': 96.6, 'action_r1': 1.1, 'fusion_sim': 0.9844, 'clip_sim': 0.6955, 'action_sim': 0.7500},
    'bright_20': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9819, 'clip_sim': 0.6934, 'action_sim': 0.7500},
    'bright_40': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9785, 'clip_sim': 0.6821, 'action_sim': 0.7500},
    'compress_50': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9721, 'clip_sim': 0.7156, 'action_sim': 0.7500},
    'compress_25': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9571, 'clip_sim': 0.6877, 'action_sim': 0.7500},
    'speed_0.9': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9824, 'clip_sim': 0.6948, 'action_sim': 0.7500},
    'speed_1.1': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9826, 'clip_sim': 0.6972, 'action_sim': 0.7500},
    'flip_h': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9852, 'clip_sim': 0.7054, 'action_sim': 0.7500},
    'flip_v': {'fusion_r1': 84.3, 'clip_r1': 84.3, 'action_r1': 1.1, 'fusion_sim': 0.9709, 'clip_sim': 0.6801, 'action_sim': 0.7500},
    'watermark': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9767, 'clip_sim': 0.7015, 'action_sim': 0.7500},
    'combined': {'fusion_r1': 100.0, 'clip_r1': 100.0, 'action_r1': 1.1, 'fusion_sim': 0.9623, 'clip_sim': 0.6658, 'action_sim': 0.7500},
}


def figure_2_ablation_summary():
    """Figure 2: 3-way ablation summary (Recall@1 bar chart)"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    modalities = ['Fusion\n(527D)', 'CLIP-only\n(512D)', 'Action-only\n(15D)']
    recalls = [FUSION_DATA['recall_1'], CLIP_ONLY_DATA['recall_1'], ACTION_ONLY_DATA['recall_1']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax.barh(modalities, recalls, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for i, (bar, recall) in enumerate(zip(bars, recalls)):
        ax.text(recall + 1.5, i, f'{recall:.1f}%', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Recall@1 (%)', fontweight='bold')
    ax.set_title('3-Way Ablation Study: Detection Performance', fontweight='bold', pad=15)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(98.6, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_ablation_summary.pdf')
    plt.savefig(OUTPUT_DIR / 'fig2_ablation_summary.png')
    print("✅ Figure 2: 3-way ablation summary saved")
    plt.close()


def figure_3_similarity_delta():
    """Figure 3: Similarity delta (bar chart)"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    modalities = ['Fusion\n(527D)', 'CLIP-only\n(512D)', 'Action-only\n(15D)']
    similarities = [FUSION_DATA['similarity'], CLIP_ONLY_DATA['similarity'], ACTION_ONLY_DATA['similarity']]
    deltas = [0, -28, -23]  # Percentage deltas
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax.barh(modalities, similarities, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels with deltas
    for i, (bar, sim, delta) in enumerate(zip(bars, similarities, deltas)):
        if delta == 0:
            label = f'{sim:.4f}'
        else:
            label = f'{sim:.4f} ({delta:+d}%)'
        ax.text(sim + 0.02, i, label, va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Average Similarity', fontweight='bold')
    ax.set_title('3-Way Ablation Study: Confidence Performance', fontweight='bold', pad=15)
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(0.9748, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_similarity_delta.pdf')
    plt.savefig(OUTPUT_DIR / 'fig3_similarity_delta.png')
    print("✅ Figure 3: Similarity delta saved")
    plt.close()


def figure_4_flip_bias():
    """Figure 4: Vertical flip bias comparison"""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    transforms = ['flip_h\n(Horizontal)', 'flip_v\n(Vertical)']
    fusion_vals = [100.0, 84.3]
    clip_vals = [100.0, 84.3]
    action_vals = [1.1, 1.1]
    
    x = np.arange(len(transforms))
    width = 0.25
    
    bars1 = ax.bar(x - width, fusion_vals, width, label='Fusion', color='#2ecc71', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, clip_vals, width, label='CLIP-only', color='#f39c12', edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, action_vals, width, label='Action-only', color='#e74c3c', edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Recall@1 (%)', fontweight='bold')
    ax.set_title('CLIP Vertical Flip Bias: Horizontal vs Vertical', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(transforms)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', frameon=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(100, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Perfect')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_flip_bias.pdf')
    plt.savefig(OUTPUT_DIR / 'fig4_flip_bias.png')
    print("✅ Figure 4: Vertical flip bias saved")
    plt.close()


def figure_5_architecture():
    """Figure 5: Architecture breakdown (527D composition)"""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    components = ['CLIP Features\n(512D)', 'Action Features\n(15D)', 'Graph Features\n(0D)']
    sizes = [512, 15, 0]
    percentages = [97.2, 2.8, 0.0]
    colors = ['#3498db', '#e67e22', '#95a5a6']
    
    # Horizontal stacked bar
    left = 0
    for i, (comp, size, pct, color) in enumerate(zip(components, sizes, percentages, colors)):
        ax.barh(0, size, left=left, height=0.6, color=color, edgecolor='black', linewidth=1.5)
        if size > 0:
            ax.text(left + size/2, 0, f'{comp}\n{pct:.1f}%', 
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        left += size
    
    ax.set_xlim(0, 527)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Embedding Dimension', fontweight='bold')
    ax.set_title('StoryHash Vector Composition (527D Total)', fontweight='bold', pad=15)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_architecture.pdf')
    plt.savefig(OUTPUT_DIR / 'fig5_architecture.png')
    print("✅ Figure 5: Architecture diagram saved")
    plt.close()


def table_1_per_transformation():
    """Table 1: Per-transformation results (LaTeX code)"""
    
    # Generate LaTeX table
    latex = r"""\begin{table*}[t]
\centering
\caption{Robustness results per transformation (89 DAVIS videos, 14 transformations). Recall@1 and average similarity for Fusion, CLIP-only, and Action-only modalities.}
\label{tab:per_transform}
\begin{tabular}{l|ccc|ccc}
\toprule
\textbf{Transformation} & \multicolumn{3}{c|}{\textbf{Recall@1 (\%)}} & \multicolumn{3}{c}{\textbf{Avg Similarity}} \\
 & Fusion & CLIP & Action & Fusion & CLIP & Action \\
\midrule
"""
    
    for transform in TRANSFORMATIONS:
        data = PER_TRANSFORM_DATA[transform]
        # Format transform name
        name = transform.replace('_', ' ').title()
        if 'flip' in transform.lower() and 'v' in transform.lower():
            name = r'\textbf{' + name + '}'  # Bold flip_v row
        
        latex += f"{name} & "
        latex += f"{data['fusion_r1']:.1f} & {data['clip_r1']:.1f} & {data['action_r1']:.1f} & "
        latex += f"{data['fusion_sim']:.4f} & {data['clip_sim']:.4f} & {data['action_sim']:.4f} \\\\\n"
    
    # Overall row
    latex += r"""\midrule
\textbf{Overall} & \textbf{98.6} & \textbf{98.2} & \textbf{1.1} & \textbf{0.9748} & \textbf{0.6996} & \textbf{0.7500} \\
\bottomrule
\end{tabular}
\end{table*}
"""
    
    # Save LaTeX code
    with open(OUTPUT_DIR / 'table1_per_transform.tex', 'w') as f:
        f.write(latex)
    
    print("✅ Table 1: Per-transformation LaTeX code saved")
    
    # Also generate PNG version for preview
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for matplotlib table
    col_labels = ['Transform', 'Fusion\nR@1', 'CLIP\nR@1', 'Action\nR@1', 
                  'Fusion\nSim', 'CLIP\nSim', 'Action\nSim']
    
    table_data = []
    for transform in TRANSFORMATIONS:
        data = PER_TRANSFORM_DATA[transform]
        name = transform.replace('_', ' ').title()
        row = [
            name,
            f"{data['fusion_r1']:.1f}",
            f"{data['clip_r1']:.1f}",
            f"{data['action_r1']:.1f}",
            f"{data['fusion_sim']:.4f}",
            f"{data['clip_sim']:.4f}",
            f"{data['action_sim']:.4f}"
        ]
        table_data.append(row)
    
    # Overall row
    table_data.append([
        'Overall',
        f"{FUSION_DATA['recall_1']:.1f}",
        f"{CLIP_ONLY_DATA['recall_1']:.1f}",
        f"{ACTION_ONLY_DATA['recall_1']:.1f}",
        f"{FUSION_DATA['similarity']:.4f}",
        f"{CLIP_ONLY_DATA['similarity']:.4f}",
        f"{ACTION_ONLY_DATA['similarity']:.4f}"
    ])
    
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.12, 0.12, 0.12, 0.13, 0.13, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight flip_v row (row 12)
    for i in range(len(col_labels)):
        table[(12, i)].set_facecolor('#ffffcc')
    
    # Highlight overall row
    for i in range(len(col_labels)):
        table[(len(table_data), i)].set_facecolor('#ecf0f1')
        table[(len(table_data), i)].set_text_props(weight='bold')
    
    plt.title('Table 1: Per-Transformation Robustness Results', 
             fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'table1_per_transform.png', bbox_inches='tight', dpi=300)
    print("✅ Table 1: PNG preview saved")
    plt.close()


def figure_1_pipeline_placeholder():
    """Figure 1: Pipeline diagram (placeholder - will be created with diagram tool)"""
    
    # Create a simple placeholder showing the pipeline structure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Pipeline text
    pipeline_text = """
    StoryHash Pipeline Architecture
    
    Input Video
        ↓
    Frame Sampling (2 fps, uniform)
        ↓
    ┌─────────────────────────────────────────┐
    │  Feature Extraction (Parallel)          │
    ├─────────────────────────────────────────┤
    │  • CLIP ViT-B/32 → 512D embeddings      │
    │    └─ Median pooling across frames      │
    │  • SAM2 Tracking → Action features      │
    │    └─ 15D: velocity, accel, spread...   │
    │  • Scene Graph → 8D relations           │
    │    └─ Spatial + temporal (optional)     │
    └─────────────────────────────────────────┘
        ↓
    Concatenation → 527D StoryHash Vector
        ↓
    FAISS HNSW Index (M=16, efConstruction=200)
        ↓
    Retrieval (Inner Product Similarity)
        ↓
    Top-K Results + Similarity Scores
    """
    
    ax.text(0.5, 0.5, pipeline_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center', horizontalalignment='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.title('Figure 1: StoryHash Pipeline (Placeholder)\nNote: Use draw.io or TikZ for final diagram', 
             fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_pipeline_placeholder.pdf')
    plt.savefig(OUTPUT_DIR / 'fig1_pipeline_placeholder.png')
    print("✅ Figure 1: Pipeline placeholder saved (create final with diagram tool)")
    plt.close()


def main():
    """Generate all figures and tables"""
    print("📊 Generating paper figures and tables...")
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")
    
    figure_1_pipeline_placeholder()
    figure_2_ablation_summary()
    figure_3_similarity_delta()
    table_1_per_transformation()
    figure_4_flip_bias()
    figure_5_architecture()
    
    print(f"\n✅ All artifacts generated in {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")
    
    print("\n📝 Next steps:")
    print("1. Review figures in paper/figures/")
    print("2. Create final Figure 1 (pipeline) with draw.io or TikZ")
    print("3. Insert figures in LaTeX paper")
    print("4. Verify all numbers match source data")


if __name__ == "__main__":
    main()
