"""
Comprehensive ranking visualization functions.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from ..core.analyzer import MultiModalInsectResistanceAnalyzer
from .metrics import (
    robust_minmax,
    infer_gain_stabilizer,
    compute_gain_rate,
    similarity_from_distance,
)

# Improve CJK rendering and overall readability across figures.
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12


def _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', pad_inches=0.06, facecolor='white'):
    """Save the current figure to both PNG and PDF."""
    save_path = Path(save_path)
    save_kwargs = {'dpi': dpi}
    if bbox_inches is not None:
        save_kwargs['bbox_inches'] = bbox_inches
    if pad_inches is not None:
        save_kwargs['pad_inches'] = pad_inches
    if facecolor is not None:
        save_kwargs['facecolor'] = facecolor

    plt.savefig(save_path, **save_kwargs)
    plt.savefig(save_path.with_suffix('.pdf'), **save_kwargs)


def _annotate_all_points_with_auto_avoid(ax, x, y, labels, is_top_mask=None):
    """Annotate all points using local 8-direction placement with overlap avoidance."""
    x = np.asarray(x)
    y = np.asarray(y)
    labels = list(labels)
    if is_top_mask is None:
        is_top_mask = np.zeros(len(labels), dtype=bool)
    else:
        is_top_mask = np.asarray(is_top_mask, dtype=bool)

    # 8 local directions around each point (in offset points):
    # N, NE, E, SE, S, SW, W, NW
    candidate_offsets = [
        (0, 11),
        (9, 9),
        (12, 0),
        (9, -9),
        (0, -11),
        (-9, -9),
        (-12, 0),
        (-9, 9),
    ]

    def _intersection_area(b1, b2):
        x0 = max(b1.x0, b2.x0)
        y0 = max(b1.y0, b2.y0)
        x1 = min(b1.x1, b2.x1)
        y1 = min(b1.y1, b2.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float((x1 - x0) * (y1 - y0))

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)

    placed_bboxes = []
    texts = []

    # Place highlighted genotypes first so they get better positions.
    priority_idx = list(np.where(is_top_mask)[0]) + list(np.where(~is_top_mask)[0])

    for i in priority_idx:
        label = labels[i]
        is_top = bool(is_top_mask[i])

        best_ann = None
        best_bbox = None
        best_score = float('inf')

        for dx, dy in candidate_offsets:
            ann = ax.annotate(
                label,
                (x[i], y[i]),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=9.5,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=6,
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.03, 1.08)

            overlap_penalty = sum(_intersection_area(bbox, prev) for prev in placed_bboxes)
            outside_penalty = 0.0
            if bbox.x0 < axes_bbox.x0:
                outside_penalty += (axes_bbox.x0 - bbox.x0)
            if bbox.y0 < axes_bbox.y0:
                outside_penalty += (axes_bbox.y0 - bbox.y0)
            if bbox.x1 > axes_bbox.x1:
                outside_penalty += (bbox.x1 - axes_bbox.x1)
            if bbox.y1 > axes_bbox.y1:
                outside_penalty += (bbox.y1 - axes_bbox.y1)

            # Prefer minimal overlap, then keep labels close to points.
            distance_penalty = (dx * dx + dy * dy) ** 0.5
            score = overlap_penalty * 1000.0 + outside_penalty * 5000.0 + distance_penalty

            if score < best_score:
                if best_ann is not None:
                    best_ann.remove()
                best_ann = ann
                best_bbox = bbox
                best_score = score
            else:
                ann.remove()

        if best_ann is not None and best_bbox is not None:
            texts.append(best_ann)
            placed_bboxes.append(best_bbox)

    return texts


def create_comprehensive_ranking_visualization(feature_type='dinov3'):
    """
    创建30个品种的综合排名可视化
    包含：特征差异、NoControl产量、成熟度（NDM）
    """
    print(f"\n{'='*80}")
    print(f"Creating Comprehensive Ranking Visualization ({feature_type.upper()})")
    print(f"{'='*80}\n")
    
    # 初始化分析器
    analyzer = MultiModalInsectResistanceAnalyzer(feature_type=feature_type)
    
    # 获取特征数据
    if feature_type == 'dinov3':
        control_features = analyzer.dinov3_control
        nocontrol_features = analyzer.dinov3_nocontrol
    elif feature_type == 'vi':
        control_features = analyzer.vi_control
        nocontrol_features = analyzer.vi_nocontrol
    elif feature_type == 'fusion':
        control_features = analyzer.fused_control
        nocontrol_features = analyzer.fused_nocontrol
    
    # 收集数据
    genotypes = sorted(analyzer.nocontrol_df['genotype'].unique())
    data_list = []

    for genotype in genotypes:
        control_plots = [pid for pid, info in control_features.items() 
                        if info['genotype'] == genotype]
        nocontrol_plots = [pid for pid, info in nocontrol_features.items() 
                          if info['genotype'] == genotype]
        
        if not control_plots or not nocontrol_plots:
            continue
        
        # 计算平均特征差异
        control_feat_mean = np.mean([control_features[pid]['features'] 
                                     for pid in control_plots], axis=0)
        nocontrol_feat_mean = np.mean([nocontrol_features[pid]['features'] 
                                       for pid in nocontrol_plots], axis=0)
        
        n_timepoints = control_feat_mean.shape[0]
        feature_diffs = [np.linalg.norm(control_feat_mean[t] - nocontrol_feat_mean[t]) 
                        for t in range(n_timepoints)]
        mean_diff = np.mean(feature_diffs)
        
        # 获取产量
        control_yield = analyzer.control_df[
            analyzer.control_df['genotype'] == genotype
        ]['grain_yield'].mean()
        
        nocontrol_yield = analyzer.nocontrol_df[
            analyzer.nocontrol_df['genotype'] == genotype
        ]['grain_yield'].mean()

        nocontrol_ndm = pd.to_numeric(
            analyzer.nocontrol_df[analyzer.nocontrol_df['genotype'] == genotype]['ndm'],
            errors='coerce'
        ).mean()
        
        # 综合评分（差异小 + NoControl产量高 + 早熟）
        data_list.append({
            'genotype': genotype,
            'mean_feature_diff': mean_diff,
            'control_yield': control_yield,
            'nocontrol_yield': nocontrol_yield,
            'nocontrol_ndm': nocontrol_ndm
        })
    
    df = pd.DataFrame(data_list)

    # Unified metric system:
    # 1) Similarity from Euclidean feature distance.
    # 2) Stabilized gain rate from control/nocontrol yield.
    # 3) Robust quantile-based normalization for scoring.
    mean_diff = df['mean_feature_diff'].values
    df['mean_feature_similarity'] = similarity_from_distance(mean_diff)

    tau_gain = infer_gain_stabilizer(df['control_yield'].values, quantile=10.0, floor=1.0)
    df['yield_gain_rate'] = compute_gain_rate(df['control_yield'].values, df['nocontrol_yield'].values, tau=tau_gain)

    similarity_score = robust_minmax(df['mean_feature_similarity'].values, inverse=False)
    nocontrol_yield_score = robust_minmax(df['nocontrol_yield'].values, inverse=False)
    ndm_early_score = robust_minmax(df['nocontrol_ndm'].values, inverse=True)

    w_diff = 0.35
    w_yield = 0.45
    w_maturity = 0.20

    df['score_feature_similarity_norm'] = similarity_score
    df['score_high_nocontrol_yield_norm'] = nocontrol_yield_score
    df['score_yield_gain_norm'] = robust_minmax(df['yield_gain_rate'].values, inverse=False)
    df['score_early_maturity_norm'] = ndm_early_score

    df['resistance_score'] = (
        w_diff * df['score_feature_similarity_norm'] +
        w_yield * df['score_high_nocontrol_yield_norm'] +
        w_maturity * df['score_early_maturity_norm']
    ) * 100
    
    df = df.sort_values('resistance_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    # ========================================================================
    # 拆分成3个独立图片，避免拥挤
    # ========================================================================
    
    print("✓ Part 1/3 - Ranking bar chart skipped by configuration")
    
    # === 图2：NoControl散点图（单图，颜色=成熟度） ===
    fig2, ax3 = plt.subplots(1, 1, figsize=(12, 9))

    x_median = float(np.nanmedian(df['mean_feature_similarity']))
    y_nocontrol_median = float(np.nanmedian(df['nocontrol_yield']))

    # 早熟更深色：使用反向蓝色映射，低NDM -> 深蓝。
    scatter3 = ax3.scatter(
        df['mean_feature_similarity'],
        df['nocontrol_yield'],
        s=200,
        alpha=0.78,
        c=df['nocontrol_ndm'],
        cmap='Blues_r',
        edgecolors='black',
        linewidth=1.5,
        vmin=float(np.nanmin(df['nocontrol_ndm'])),
        vmax=float(np.nanmax(df['nocontrol_ndm'])),
    )

    top_mask = df['rank'].values <= 5
    _annotate_all_points_with_auto_avoid(
        ax3,
        df['mean_feature_similarity'].values,
        df['nocontrol_yield'].values,
        df['genotype'].values,
        is_top_mask=top_mask,
    )

    ax3.set_xlabel('Mean Feature Similarity', fontsize=16, fontweight='bold')
    ax3.set_ylabel('NoControl Yield (kg/ha)', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.axvline(x_median, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
    ax3.axhline(y_nocontrol_median, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
    ax3.tick_params(axis='both', labelsize=13)

    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('NoControl Maturity (NDM, Earlier = Darker)', fontsize=14, fontweight='bold')
    cbar3.ax.tick_params(labelsize=12)

    plt.tight_layout()
    
    save_path2 = analyzer.output_dir / f'comprehensive_ranking_scatter_{feature_type}.png'
    _save_png_and_pdf(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Part 2/4 - NoControl scatter saved: {save_path2}")

    # === 图3：增产率二维散点图（单图，颜色=成熟度） ===
    fig_gain, ax_gain = plt.subplots(1, 1, figsize=(12, 9))

    y_gain_median = float(np.nanmedian(df['yield_gain_rate']))
    scatter_gain = ax_gain.scatter(
        df['mean_feature_similarity'],
        df['yield_gain_rate'],
        s=200,
        alpha=0.78,
        c=df['nocontrol_ndm'],
        cmap='Blues_r',
        edgecolors='black',
        linewidth=1.5,
        vmin=float(np.nanmin(df['nocontrol_ndm'])),
        vmax=float(np.nanmax(df['nocontrol_ndm'])),
    )

    _annotate_all_points_with_auto_avoid(
        ax_gain,
        df['mean_feature_similarity'].values,
        df['yield_gain_rate'].values,
        df['genotype'].values,
        is_top_mask=top_mask,
    )

    ax_gain.set_xlabel('Mean Feature Similarity', fontsize=16, fontweight='bold')
    ax_gain.set_ylabel('Yield Gain Rate', fontsize=16, fontweight='bold')
    ax_gain.grid(True, alpha=0.3, linestyle=':')
    ax_gain.axvline(x_median, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_gain.axhline(y_gain_median, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_gain.tick_params(axis='both', labelsize=13)

    cbar_gain = plt.colorbar(scatter_gain, ax=ax_gain)
    cbar_gain.set_label('NoControl Maturity (NDM, Earlier = Darker)', fontsize=14, fontweight='bold')
    cbar_gain.ax.tick_params(labelsize=12)

    plt.tight_layout()

    save_path_gain = analyzer.output_dir / f'comprehensive_ranking_gain_scatter_{feature_type}.png'
    _save_png_and_pdf(save_path_gain, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Part 3/4 - Gain-rate scatter saved: {save_path_gain}")
    
    # === 图4：完整数据表（所有30个品种）===
    fig3, ax4 = plt.subplots(figsize=(18, 14))
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    table_data.append(['Rank', 'Genotype', 'Feature\nSim', 'Control\nYield', 'NoControl\nYield', 'Gain\nRate', 'NDM', 'Score'])
    
    # 显示所有30个品种
    for idx, row in df.iterrows():
        table_data.append([
            f"{row['rank']}",
            row['genotype'],
            f"{row['mean_feature_similarity']:.2f}",
            f"{row['control_yield']:.0f}",
            f"{row['nocontrol_yield']:.0f}",
            f"{row['yield_gain_rate']:.3f}",
            f"{row['nocontrol_ndm']:.1f}" if pd.notna(row['nocontrol_ndm']) else 'NA',
            f"{row['resistance_score']:.1f}"
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.07, 0.15, 0.10, 0.11, 0.12, 0.13, 0.09, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 表头样式
    for i in range(8):
        table[(0, i)].set_facecolor('#2ecc71')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # 根据排名渐变着色
    for i in range(1, len(table_data)):
        score = df.iloc[i-1]['resistance_score']
        color_val = score / 100
        bg_color = plt.cm.RdYlGn(color_val)
        
        for j in range(8):
            table[(i, j)].set_facecolor(bg_color)
            if j == 0:  # Rank列加粗
                table[(i, j)].set_text_props(weight='bold', fontsize=11)
    
    plt.tight_layout()
    save_path3 = analyzer.output_dir / f'comprehensive_ranking_table_{feature_type}.png'
    _save_png_and_pdf(save_path3, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Part 3/3 - Data table saved: {save_path3}")
    
    # 保存完整数据
    csv_path = analyzer.output_dir / f'comprehensive_ranking_data_{feature_type}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Complete ranking data saved: {csv_path}")

    # Save a concise text report for direct use in documentation.
    report_path = analyzer.output_dir / f'comprehensive_ranking_report_{feature_type}.txt'
    top_n = len(df)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=' * 90 + '\n')
        f.write(f'COMPREHENSIVE RANKING REPORT - {feature_type.upper()}\n')
        f.write('=' * 90 + '\n\n')
        f.write('Scoring Formula:\n')
        f.write('  Score = 0.35*FeatureSimilarity + 0.45*HighNoControlYield + 0.20*EarlyMaturity(NDM)\n')
        f.write('  GainRate = (NoControl - Control) / (Control + tau)\n')
        f.write(f'  tau (stabilizer) = {tau_gain:.3f}\n\n')

        f.write('All Genotypes (Ranked):\n')
        f.write('-' * 90 + '\n')
        f.write('Rank | Genotype | Score | ControlYield | NoControlYield | GainRate | FeatureSim | NDM\n')
        for i in range(top_n):
            row = df.iloc[i]
            f.write(
                f"{int(row['rank']):>4d} | "
                f"{row['genotype']:<10} | "
                f"{row['resistance_score']:>6.2f} | "
                f"{row['control_yield']:>12.0f} | "
                f"{row['nocontrol_yield']:>14.0f} | "
                f"{row['yield_gain_rate']:>8.3f} | "
                f"{row['mean_feature_similarity']:>10.3f} | "
                f"{row['nocontrol_ndm']:>5.1f}\n"
            )

        f.write('\nSummary Statistics:\n')
        f.write(f"  Feature Similarity: mean={df['mean_feature_similarity'].mean():.3f}, std={df['mean_feature_similarity'].std():.3f}\n")
        f.write(f"  Control Yield: mean={df['control_yield'].mean():.1f}, std={df['control_yield'].std():.1f}\n")
        f.write(f"  NoControl Yield: mean={df['nocontrol_yield'].mean():.1f}, std={df['nocontrol_yield'].std():.1f}\n")
        f.write(f"  Gain Rate: mean={df['yield_gain_rate'].mean():.3f}, std={df['yield_gain_rate'].std():.3f}\n")
        f.write(f"  NDM: mean={df['nocontrol_ndm'].mean():.1f}, std={df['nocontrol_ndm'].std():.1f}\n")
    print(f"✓ Ranking text report saved: {report_path}")
    
    # 打印统计摘要
    print(f"\n{'='*80}")
    print("Statistical Summary")
    print(f"{'='*80}")
    print(f"\nFeature Similarity Statistics:")
    print(f"  Mean: {df['mean_feature_similarity'].mean():.3f}")
    print(f"  Std:  {df['mean_feature_similarity'].std():.3f}")
    print(f"  Range: [{df['mean_feature_similarity'].min():.3f}, {df['mean_feature_similarity'].max():.3f}]")

    print(f"\nYield Gain Rate Statistics (stabilized):")
    print(f"  tau (denominator stabilizer): {tau_gain:.3f}")
    print(f"  Mean: {df['yield_gain_rate'].mean():.3f}")
    print(f"  Std:  {df['yield_gain_rate'].std():.3f}")
    print(f"  Range: [{df['yield_gain_rate'].min():.3f}, {df['yield_gain_rate'].max():.3f}]")
    
    print(f"\nControl Yield Statistics:")
    print(f"  Mean: {df['control_yield'].mean():.1f} kg/ha")
    print(f"  Std:  {df['control_yield'].std():.1f} kg/ha")

    print(f"\nNoControl NDM Statistics (smaller = earlier maturity):")
    print(f"  Mean: {df['nocontrol_ndm'].mean():.1f}")
    print(f"  Std:  {df['nocontrol_ndm'].std():.1f}")
    print(f"  Range: [{df['nocontrol_ndm'].min():.1f}, {df['nocontrol_ndm'].max():.1f}]")
    
    print(f"\nNoControl Yield Statistics:")
    print(f"  Mean: {df['nocontrol_yield'].mean():.1f} kg/ha")
    print(f"  Std:  {df['nocontrol_yield'].std():.1f} kg/ha")
    
    print(f"\n{'='*80}\n")
    
    return df

