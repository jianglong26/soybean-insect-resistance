"""
Quadrant Stability Analysis Plots
仅保留跨时间点稳定性分析（已移除旧的timepoints/average象限图）。
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ..core.analyzer import MultiModalInsectResistanceAnalyzer


def analyze_quadrant_stability_across_timepoints(feature_type='dinov3'):
    """
    分析品种在不同时间点的象限稳定性。

    IDEAL定义：特征差异 < 当期中位数 且 产量损失率 < 当期中位数。
    注意：该函数用于稳定性比较，不参与综合排名主分数。

    Args:
        feature_type: 'dinov3', 'vi', 'fusion'
    """
    print(f"\n{'='*80}")
    print(f"Quadrant Stability Analysis Across Timepoints ({feature_type.upper()})")
    print(f"{'='*80}\n")

    analyzer = MultiModalInsectResistanceAnalyzer(feature_type=feature_type)

    if feature_type == 'dinov3':
        control_features = analyzer.dinov3_control
        nocontrol_features = analyzer.dinov3_nocontrol
    elif feature_type == 'vi':
        control_features = analyzer.vi_control
        nocontrol_features = analyzer.vi_nocontrol
    elif feature_type == 'fusion':
        control_features = analyzer.fused_control
        nocontrol_features = analyzer.fused_nocontrol
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    metadata_path = analyzer.data_dir / 'dataset_metadata.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    first_nocontrol = list(nocontrol_features.keys())[0]
    dates = [img['date'] for img in metadata[first_nocontrol]['image_sequence']]
    date_labels = [d[5:] for d in dates]
    n_timepoints = len(dates)

    genotypes = sorted(analyzer.nocontrol_df['genotype'].unique())
    genotype_data = {}

    for genotype in genotypes:
        control_yield = analyzer.control_df[
            analyzer.control_df['genotype'] == genotype
        ]['grain_yield'].mean()

        nocontrol_yield = analyzer.nocontrol_df[
            analyzer.nocontrol_df['genotype'] == genotype
        ]['grain_yield'].mean()

        yield_loss_rate = (control_yield - nocontrol_yield) / control_yield * 100 if control_yield > 0 else 0

        control_plots = [pid for pid, info in control_features.items() if info['genotype'] == genotype]
        nocontrol_plots = [pid for pid, info in nocontrol_features.items() if info['genotype'] == genotype]

        if not control_plots or not nocontrol_plots:
            continue

        control_feat_mean = np.mean([control_features[pid]['features'] for pid in control_plots], axis=0)
        nocontrol_feat_mean = np.mean([nocontrol_features[pid]['features'] for pid in nocontrol_plots], axis=0)

        feature_diffs = [
            np.linalg.norm(control_feat_mean[t] - nocontrol_feat_mean[t])
            for t in range(n_timepoints)
        ]

        nocontrol_ndm = pd.to_numeric(
            analyzer.nocontrol_df[analyzer.nocontrol_df['genotype'] == genotype]['ndm'],
            errors='coerce'
        ).mean()

        genotype_data[genotype] = {
            'feature_diffs': feature_diffs,
            'yield_loss_rate': yield_loss_rate,
            'nocontrol_yield': nocontrol_yield,
            'nocontrol_ndm': nocontrol_ndm,
        }

    quadrant_by_timepoint = {}
    for t in range(n_timepoints):
        feature_diffs_t = [genotype_data[g]['feature_diffs'][t] for g in genotype_data.keys()]
        loss_rates = [genotype_data[g]['yield_loss_rate'] for g in genotype_data.keys()]

        median_diff = np.median(feature_diffs_t)
        median_loss = np.median(loss_rates)

        ideal_genotypes = []
        for g in genotype_data.keys():
            if (
                genotype_data[g]['feature_diffs'][t] < median_diff
                and genotype_data[g]['yield_loss_rate'] < median_loss
            ):
                ideal_genotypes.append(g)

        quadrant_by_timepoint[t] = {
            'date': dates[t],
            'date_label': date_labels[t],
            'ideal_genotypes': ideal_genotypes,
            'median_diff': median_diff,
            'median_loss': median_loss,
        }

    genotype_ideal_count = {g: 0 for g in genotype_data.keys()}
    for t in range(n_timepoints):
        for g in quadrant_by_timepoint[t]['ideal_genotypes']:
            genotype_ideal_count[g] += 1

    always_ideal = [g for g, count in genotype_ideal_count.items() if count == n_timepoints]
    mostly_ideal = [
        g for g, count in genotype_ideal_count.items()
        if count >= n_timepoints * 0.7 and count < n_timepoints
    ]

    # 稳定性优先，同分时按早熟优先
    genotypes_sorted = sorted(
        genotype_ideal_count.keys(),
        key=lambda x: (
            genotype_ideal_count[x],
            -genotype_data[x]['nocontrol_ndm'] if np.isfinite(genotype_data[x]['nocontrol_ndm']) else -9999
        ),
        reverse=True
    )

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.5, 1.2], hspace=0.45)

    ax1 = fig.add_subplot(gs[0])

    stability_matrix = np.zeros((len(genotypes_sorted), n_timepoints))
    for i, g in enumerate(genotypes_sorted):
        for t in range(n_timepoints):
            if g in quadrant_by_timepoint[t]['ideal_genotypes']:
                stability_matrix[i, t] = 1

    im1 = ax1.imshow(stability_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)

    ax1.set_xlabel('Time Point', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Genotype (Sorted by Stability)', fontsize=13, fontweight='bold')
    ax1.set_title(
        'Genotype Stability in Ideal Quadrant (Lower-left) Across Time\\n'
        'Green = In Ideal Quadrant, Red = Not in Ideal Quadrant (Tie-break by Early Maturity)',
        fontsize=15,
        fontweight='bold',
        pad=15,
    )

    ax1.set_xticks(range(n_timepoints))
    ax1.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=11)
    ax1.set_yticks(range(len(genotypes_sorted)))
    ax1.set_yticklabels(genotypes_sorted, fontsize=9)

    for i, g in enumerate(genotypes_sorted):
        if g in always_ideal:
            ax1.get_yticklabels()[i].set_color('darkgreen')
            ax1.get_yticklabels()[i].set_fontweight('bold')
            ax1.get_yticklabels()[i].set_fontsize(11)
        elif g in mostly_ideal:
            ax1.get_yticklabels()[i].set_color('green')
            ax1.get_yticklabels()[i].set_fontweight('bold')

    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1])
    cbar1.set_ticklabels(['Not Ideal', 'Ideal'])

    ax2 = fig.add_subplot(gs[1])
    timepoint_ideal_counts = [len(quadrant_by_timepoint[t]['ideal_genotypes']) for t in range(n_timepoints)]
    bars = ax2.bar(
        range(n_timepoints),
        timepoint_ideal_counts,
        color=['#2ecc71' if c > np.mean(timepoint_ideal_counts) else '#f39c12' for c in timepoint_ideal_counts],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8,
    )

    for bar, count in zip(bars, timepoint_ideal_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.axhline(np.mean(timepoint_ideal_counts), color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Mean: {np.mean(timepoint_ideal_counts):.1f}')

    ax2.set_xlabel('Time Point', fontsize=12, fontweight='bold')
    ax2.set_ylabel('No. of Genotypes in Ideal Quadrant', fontsize=11, fontweight='bold')
    ax2.set_title('Number of Ideal Genotypes at Each Time Point', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(range(n_timepoints))
    ax2.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax2.legend(loc='upper right', fontsize=11)

    fig.suptitle(f'Quadrant Stability Analysis - {feature_type.upper()}', fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout()

    save_path = analyzer.output_dir / f'quadrant_stability_analysis_{feature_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.06, facecolor='white')
    plt.close()
    print(f"✓ Quadrant stability analysis saved: {save_path}")

    summary_path = analyzer.output_dir / f'quadrant_stability_summary_{feature_type}.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write(f'QUADRANT STABILITY ANALYSIS SUMMARY - {feature_type.upper()}\n')
        f.write('=' * 80 + '\n\n')
        f.write(f'Always in Ideal Quadrant ({len(always_ideal)} genotypes, {n_timepoints}/{n_timepoints} timepoints):\n')
        if always_ideal:
            for g in always_ideal:
                f.write(f'  - {g}\n')
        else:
            f.write('  None\n')
        f.write('\n')

        f.write(f'Mostly in Ideal Quadrant ({len(mostly_ideal)} genotypes, >={int(n_timepoints*0.7)}/{n_timepoints} timepoints):\n')
        if mostly_ideal:
            for g in mostly_ideal:
                count = genotype_ideal_count[g]
                percentage = (count / n_timepoints) * 100
                f.write(f'  - {g}: {count}/{n_timepoints} timepoints ({percentage:.1f}%)\n')
        else:
            f.write('  None\n')
        f.write('\n')

        f.write('=' * 80 + '\n')
        f.write('IDEAL GENOTYPES BREAKDOWN BY TIME POINT\n')
        f.write('=' * 80 + '\n\n')
        for t in range(n_timepoints):
            ideal_genotypes = quadrant_by_timepoint[t]['ideal_genotypes']
            f.write(f'Time Point {t+1} ({date_labels[t]}): {len(ideal_genotypes)} genotypes\n')
            if ideal_genotypes:
                for i in range(0, len(ideal_genotypes), 5):
                    chunk = ideal_genotypes[i:i+5]
                    f.write(f"  {', '.join(chunk)}\n")
            else:
                f.write('  None\n')
            f.write('\n')

        f.write('=' * 80 + '\n')

    print(f"✓ Stability summary saved: {summary_path}")

    stability_data = []
    for g in genotypes_sorted:
        row = {
            'genotype': g,
            'ideal_count': genotype_ideal_count[g],
            'ideal_percentage': genotype_ideal_count[g] / n_timepoints * 100,
            'nocontrol_ndm': genotype_data[g]['nocontrol_ndm'],
            'stability_category': 'Always' if g in always_ideal else ('Mostly' if g in mostly_ideal else 'Sometimes'),
        }
        for t in range(n_timepoints):
            row[f'timepoint_{t+1}_{date_labels[t]}'] = 'Ideal' if g in quadrant_by_timepoint[t]['ideal_genotypes'] else 'Not Ideal'
        stability_data.append(row)

    df_stability = pd.DataFrame(stability_data)
    csv_path = analyzer.output_dir / f'quadrant_stability_data_{feature_type}.csv'
    df_stability.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Stability data saved: {csv_path}")

    print(f"\n{'='*80}")
    print('STABILITY SUMMARY')
    print(f"{'='*80}")
    print(f"\nAlways in Ideal Quadrant: {len(always_ideal)} genotypes")
    if always_ideal:
        print(f"  {', '.join(always_ideal)}")
    print(f"\nMostly in Ideal Quadrant (>={int(n_timepoints*0.7)}/{n_timepoints}): {len(mostly_ideal)} genotypes")
    if mostly_ideal:
        for g in mostly_ideal:
            print(f"  {g}: {genotype_ideal_count[g]}/{n_timepoints}")

    print(f"\nIdeal Genotypes Count by Time Point:")
    for t in range(n_timepoints):
        print(f"  {date_labels[t]}: {len(quadrant_by_timepoint[t]['ideal_genotypes'])} genotypes")

    print(f"\n{'='*80}\n")

    return df_stability, always_ideal, mostly_ideal
