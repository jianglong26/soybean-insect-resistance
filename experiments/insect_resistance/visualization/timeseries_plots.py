"""
Time-series Analysis Plots
时间序列分析可视化
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats
from pathlib import Path
import json

from ..core.analyzer import MultiModalInsectResistanceAnalyzer
from .metrics import robust_minmax, infer_gain_stabilizer, compute_gain_rate

# Improve CJK rendering and overall readability across figures.
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12


def _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', pad_inches=0.06, facecolor='white', bbox_extra_artists=None):
    """Save the current figure to both PNG and PDF."""
    save_path = Path(save_path)
    save_kwargs = {'dpi': dpi}
    if bbox_inches is not None:
        save_kwargs['bbox_inches'] = bbox_inches
    if pad_inches is not None:
        save_kwargs['pad_inches'] = pad_inches
    if facecolor is not None:
        save_kwargs['facecolor'] = facecolor
    if bbox_extra_artists is not None:
        save_kwargs['bbox_extra_artists'] = bbox_extra_artists

    targets = [save_path, save_path.with_suffix('.pdf')]
    for target in targets:
        try:
            plt.savefig(target, **save_kwargs)
        except PermissionError:
            # Allow analysis to continue when a target file is open in another program.
            print(f"Warning: skip saving locked file: {target}")

try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False

    plt.tight_layout()
    
    # 保存
    save_path = output_dir / f'image_comparison_{genotype_folder}.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Image comparison saved: {save_path}")
    print(f"  - {n_times} timepoints visualized")
    print(f"\n{'='*80}\n")


def visualize_all_timepoints_difference(feature_type='dinov3'):
    """
    Visualize feature difference changes over time for all 30 genotypes
    Shows how feature differences between control and nocontrol evolve across 8 timepoints
    
    Args:
        feature_type: 'dinov3', 'vi', 'fusion'
    """
    print(f"\n{'='*80}")
    print(f"Feature Difference Time-series for All Genotypes ({feature_type.upper()})")
    print(f"{'='*80}\n")
    
    # Initialize analyzer
    analyzer = MultiModalInsectResistanceAnalyzer(feature_type=feature_type)
    
    # Get feature data
    if feature_type == 'dinov3':
        control_features = analyzer.dinov3_control
        nocontrol_features = analyzer.dinov3_nocontrol
    elif feature_type == 'vi':
        control_features = analyzer.vi_control
        nocontrol_features = analyzer.vi_nocontrol
    elif feature_type == 'fusion':
        control_features = analyzer.fused_control
        nocontrol_features = analyzer.fused_nocontrol
    
    # Collect data for all genotypes and timepoints
    genotypes = sorted(analyzer.nocontrol_df['genotype'].unique())
    
    # 动态获取日期标签（从metadata中读取）
    metadata_path = analyzer.data_dir / 'dataset_metadata.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    first_nocontrol = list(nocontrol_features.keys())[0]
    dates = [img['date'] for img in metadata[first_nocontrol]['image_sequence']]
    date_labels = [d[5:].replace('-', ' ') for d in dates]
    n_timepoints = len(dates)
    
    # Prepare data matrix: genotypes × timepoints
    data_matrix = []
    genotype_list = []
    
    for genotype in genotypes:
        control_plots = [pid for pid, info in control_features.items() 
                        if info['genotype'] == genotype]
        nocontrol_plots = [pid for pid, info in nocontrol_features.items() 
                          if info['genotype'] == genotype]
        
        if not control_plots or not nocontrol_plots:
            continue
        
        # Calculate feature differences at each timepoint
        control_feat_mean = np.mean([control_features[pid]['features'] 
                                     for pid in control_plots], axis=0)
        nocontrol_feat_mean = np.mean([nocontrol_features[pid]['features'] 
                                       for pid in nocontrol_plots], axis=0)
        
        n_timepoints = control_feat_mean.shape[0]
        feature_diffs = [np.linalg.norm(control_feat_mean[t] - nocontrol_feat_mean[t]) 
                        for t in range(n_timepoints)]
        
        data_matrix.append(feature_diffs)
        genotype_list.append(genotype)
    
    data_matrix = np.array(data_matrix)  # Shape: (30, 8)
    
    # Get yield loss rate for ranking
    yield_loss_dict = {}
    for genotype in genotype_list:
        control_yield = analyzer.control_df[
            analyzer.control_df['genotype'] == genotype
        ]['grain_yield'].mean()
        
        control_yield = analyzer.control_df[
            analyzer.control_df['genotype'] == genotype
        ]['grain_yield'].mean()

        nocontrol_yield = analyzer.nocontrol_df[
            analyzer.nocontrol_df['genotype'] == genotype
        ]['grain_yield'].mean()

        # Unified stabilized gain rate definition.
        tau_gain = infer_gain_stabilizer(
            analyzer.control_df['grain_yield'].to_numpy(dtype=float),
            quantile=10.0,
            floor=1.0,
        )
        yield_gain_rate = float(compute_gain_rate(control_yield, nocontrol_yield, tau=tau_gain))
        
        yield_loss_rate = (control_yield - nocontrol_yield) / control_yield * 100 if control_yield > 0 else 0
        yield_loss_dict[genotype] = yield_loss_rate
    
    # Calculate overall resistance score (lower diff + lower loss = better)
    mean_diffs = data_matrix.mean(axis=1)
    resistance_scores = []
    for i, genotype in enumerate(genotype_list):
        diff_norm = (mean_diffs[i] - mean_diffs.min()) / (mean_diffs.max() - mean_diffs.min())
        loss = yield_loss_dict[genotype]
        loss_norm = (loss - min(yield_loss_dict.values())) / (max(yield_loss_dict.values()) - min(yield_loss_dict.values()))
        score = 100 - (diff_norm * 50 + loss_norm * 50)
        resistance_scores.append(score)
    
    # Sort by resistance score
    sorted_indices = np.argsort(resistance_scores)[::-1]
    top5_genotypes = [genotype_list[i] for i in sorted_indices[:5]]
    
    # Create visualization: Line plot + Heatmap (2x1 layout, no summary)
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.3)
    
    # ===== Plot 1: Line plot - All genotypes =====
    ax1 = fig.add_subplot(gs[0])
    
    # Plot all genotypes with low opacity
    for i, genotype in enumerate(genotype_list):
        if genotype not in top5_genotypes:
            ax1.plot(range(n_timepoints), data_matrix[i], color='gray', alpha=0.2, linewidth=1)
    
    # Highlight Top 5 genotypes with distinct colors
    colors_top5 = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, genotype in enumerate(top5_genotypes):
        i = genotype_list.index(genotype)
        ax1.plot(range(n_timepoints), data_matrix[i], color=colors_top5[idx], 
                linewidth=3, marker=markers[idx], markersize=8, 
                label=f'{genotype} (Rank #{idx+1})', alpha=0.9)
    
    # Calculate and plot mean trend
    mean_trend = data_matrix.mean(axis=0)
    ax1.plot(range(n_timepoints), mean_trend, color='black', linewidth=3, 
            linestyle='--', marker='o', markersize=6, 
            label='Mean (All 30 Genotypes)', alpha=0.8)
    
    ax1.set_xlabel('Time Point', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feature Difference (Control - NoControl)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Feature Difference Evolution Over Time - {feature_type.upper()}\n'
                 f'All 30 Genotypes (Top 5 Highlighted)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax1.set_xticks(range(n_timepoints))
    ax1.set_xticklabels(date_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # ===== Plot 2: Heatmap - All genotypes × timepoints =====
    ax2 = fig.add_subplot(gs[1])
    
    # Sort genotypes by mean difference for better visualization
    mean_diffs_sorted = data_matrix.mean(axis=1)
    sorted_idx = np.argsort(mean_diffs_sorted)
    sorted_genotypes = [genotype_list[i] for i in sorted_idx]
    sorted_data = data_matrix[sorted_idx]
    
    im = ax2.imshow(sorted_data, aspect='auto', cmap='RdYlGn_r', 
                    interpolation='nearest')
    
    ax2.set_xlabel('Time Point', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Genotype (Sorted by Mean Difference)', fontsize=13, fontweight='bold')
    ax2.set_title('Feature Difference Heatmap\n(Low = Good Resistance)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 设置xlim使其与上方折线图对齐（在设置其他属性之前）
    ax2.set_xlim(ax1.get_xlim())
    
    ax2.set_xticks(range(n_timepoints))
    ax2.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=10)
    ax2.set_yticks(range(len(sorted_genotypes)))
    ax2.set_yticklabels(sorted_genotypes, fontsize=8)
    
    # Highlight Top 5 genotypes
    for idx, genotype in enumerate(top5_genotypes):
        y_pos = sorted_genotypes.index(genotype)
        ax2.get_yticklabels()[y_pos].set_color(colors_top5[idx])
        ax2.get_yticklabels()[y_pos].set_fontweight('bold')
        ax2.get_yticklabels()[y_pos].set_fontsize(10)
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Feature Difference', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure (without summary text)
    save_path = analyzer.output_dir / f'feature_difference_timeseries_{feature_type}.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Feature difference time-series saved: {save_path}")
    
    # ===== Save statistical summary as separate text file =====
    summary_text = f"STATISTICAL SUMMARY - {feature_type.upper()}\n"
    summary_text += f"{'='*80}\n\n"
    summary_text += f"Dataset Information:\n"
    summary_text += f"  - Genotypes: {len(genotype_list)}\n"
    summary_text += f"  - Timepoints: {n_timepoints}\n"
    summary_text += f"  - Total data points: {len(genotype_list) * n_timepoints}\n\n"
    
    summary_text += f"Feature Difference Statistics:\n"
    summary_text += f"  - Overall Range: [{data_matrix.min():.3f}, {data_matrix.max():.3f}]\n"
    summary_text += f"  - Mean ± Std: {data_matrix.mean():.3f} ± {data_matrix.std():.3f}\n\n"
    
    summary_text += f"Temporal Variation (Mean ± Std per timepoint):\n"
    for t, date in enumerate(date_labels):
        summary_text += f"  - {date}: {data_matrix[:, t].mean():.3f} ± {data_matrix[:, t].std():.3f}\n"
    
    summary_text += f"\nTop 5 Resistant Genotypes (Low Difference + Low Yield Loss):\n"
    for idx, genotype in enumerate(top5_genotypes):
        i = genotype_list.index(genotype)
        summary_text += f"  #{idx+1}. {genotype}\n"
        summary_text += f"       Feature Diff: {mean_diffs[i]:.3f}\n"
        summary_text += f"       Yield Loss: {yield_loss_dict[genotype]:.1f}%\n"
        summary_text += f"       Resistance Score: {resistance_scores[i]:.2f}\n"
    
    # Save summary as text file
    summary_path = analyzer.output_dir / f'feature_difference_timeseries_summary_{feature_type}.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"✓ Statistical summary saved: {summary_path}")
    
    # Save detailed data
    csv_data = []
    for i, genotype in enumerate(genotype_list):
        for t, date in enumerate(dates):
            csv_data.append({
                'genotype': genotype,
                'timepoint': t + 1,
                'date': date,
                'feature_difference': data_matrix[i, t],
                'yield_loss_rate': yield_loss_dict[genotype],
                'resistance_score': resistance_scores[i]
            })
    
    df = pd.DataFrame(csv_data)
    csv_path = analyzer.output_dir / f'feature_difference_timeseries_data_{feature_type}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Data saved: {csv_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("Statistical Summary")
    print(f"{'='*80}")
    print(f"\nGenotypes: {len(genotype_list)}")
    print(f"Timepoints: {n_timepoints}")
    print(f"\nFeature Difference:")
    print(f"  Range: [{data_matrix.min():.3f}, {data_matrix.max():.3f}]")
    print(f"  Mean ± Std: {data_matrix.mean():.3f} ± {data_matrix.std():.3f}")
    print(f"\nTemporal Variation (Mean per timepoint):")
    for t, date in enumerate(date_labels):
        print(f"  {date}: {data_matrix[:, t].mean():.3f} ± {data_matrix[:, t].std():.3f}")
    print(f"\nTop 5 Resistant Genotypes:")
    for idx, genotype in enumerate(top5_genotypes):
        i = genotype_list.index(genotype)
        print(f"  #{idx+1}. {genotype}: Diff={mean_diffs[i]:.3f}, Loss={yield_loss_dict[genotype]:.1f}%")
    
    print(f"\n{'='*80}\n")
    
    return df


def analyze_feature_vs_yield_timeseries(feature_type='dinov3'):
    """
    时序分析：将相关散点图升级为三维。

        说明：
        - 原有四张图保留 z轴=NoControl Yield。
        - 新增四张图使用 z轴=Yield Gain Rate，其中
            YieldGainRate=(NoControl-Control)/Control，再做0-1归一化。
        - 双排名改为：0.40*LowSimilarity + 0.40*HighGain + 0.20*EarlyMaturity。

    Args:
        feature_type: 'dinov3', 'vi', 'fusion'
    """
    print(f"\n{'='*80}")
    print(f"Feature Difference vs Yield Time-series Analysis ({feature_type.upper()})")
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

    sample_plot = list(nocontrol_features.keys())[0]
    n_timepoints = nocontrol_features[sample_plot]['features'].shape[0]

    metadata_path = analyzer.data_dir / 'dataset_metadata.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    first_nocontrol = list(nocontrol_features.keys())[0]
    dates = [img['date'] for img in metadata[first_nocontrol]['image_sequence']]
    date_labels = [d[5:] for d in dates]

    print(f"Number of time points: {n_timepoints}")
    print(f"Dates: {dates}")

    def _inverse_norm(values):
        return robust_minmax(values, inverse=True)

    def _forward_norm(values):
        return robust_minmax(values, inverse=False)

    def _rank_spread(values):
        """Percentile-rank spreading for display only (helps de-crowd near-boundary points)."""
        arr = np.asarray(values, dtype=float)
        out = np.full(arr.shape, 0.5, dtype=float)
        valid = np.isfinite(arr)
        if valid.sum() > 0:
            out[valid] = pd.Series(arr[valid]).rank(method='average', pct=True).to_numpy()
        return out

    # 收集数据
    genotypes = sorted(analyzer.nocontrol_df['genotype'].unique())
    genotype_data = {}

    for genotype in genotypes:
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

        control_yield = analyzer.control_df[
            analyzer.control_df['genotype'] == genotype
        ]['grain_yield'].mean()

        nocontrol_yield = analyzer.nocontrol_df[
            analyzer.nocontrol_df['genotype'] == genotype
        ]['grain_yield'].mean()

        yield_gain_rate = np.nan  # Placeholder; recomputed with unified stabilizer after all genotypes are collected.

        nocontrol_ndm = pd.to_numeric(
            analyzer.nocontrol_df[analyzer.nocontrol_df['genotype'] == genotype]['ndm'],
            errors='coerce'
        ).mean()

        genotype_data[genotype] = {
            'feature_diffs': feature_diffs,
            'control_yield': control_yield,
            'nocontrol_yield': nocontrol_yield,
            'yield_gain_rate': yield_gain_rate,
            'nocontrol_ndm': nocontrol_ndm,
            'mean_diff': np.mean(feature_diffs)
        }

    print(f"Valid genotypes: {len(genotype_data)}")

    names = list(genotype_data.keys())
    control_yields = np.array([genotype_data[g]['control_yield'] for g in names], dtype=float)
    yields = np.array([genotype_data[g]['nocontrol_yield'] for g in names], dtype=float)
    tau_gain = infer_gain_stabilizer(control_yields, quantile=10.0, floor=1.0)
    gain_rates = compute_gain_rate(control_yields, yields, tau=tau_gain)
    gain_norm = _forward_norm(gain_rates)
    ndm_values = np.array([genotype_data[g]['nocontrol_ndm'] for g in names], dtype=float)
    ndm_earliness = _inverse_norm(ndm_values)
    y_norm = _forward_norm(yields)
    # Readability settings for publication figures
    marker_size_fixed = 155.0
    text_size_timeseries = 7.8
    text_size_average = 8.6
    use_median_planes = True
    median_plane_alpha = 0.13
    projection_alpha = 0.34

    # Fixed color palette for 8 median-defined regions
    region_colors = {
        0: '#1F77B4',
        1: '#FF7F0E',
        2: '#2CA02C',
        3: '#D62728',
        4: '#9467BD',
        5: '#8C564B',
        6: '#E377C2',
        7: '#17BECF',
    }

    def _label_offset(i):
        # Deterministic tiny offsets to reduce text overlap in dense 3D scatter
        dx = [-0.010, -0.006, 0.006, 0.010, -0.008, 0.008][i % 6]
        dy = [0.20, -0.20, 0.30, -0.30, 0.15, -0.15][i % 6]
        # Ratio-based z offset; converted to absolute value with z-range in _place_labels_sparse.
        dz = [0.08, 0.05, 0.07, 0.045, 0.06, 0.04][i % 6]
        return dx, dy, dz

    def _draw_median_planes(ax, x_med, y_med, z_med, xlim, ylim, zlim):
        """Draw 3 orthogonal transparent planes crossing at median point."""
        if not use_median_planes:
            return

        yy, zz = np.meshgrid(np.linspace(ylim[0], ylim[1], 2), np.linspace(zlim[0], zlim[1], 2))
        xx = np.full_like(yy, x_med)
        ax.plot_surface(xx, yy, zz, color='#616161', alpha=median_plane_alpha, linewidth=0.35, edgecolor='#424242', shade=False)

        xx2, zz2 = np.meshgrid(np.linspace(xlim[0], xlim[1], 2), np.linspace(zlim[0], zlim[1], 2))
        yy2 = np.full_like(xx2, y_med)
        ax.plot_surface(xx2, yy2, zz2, color='#64b5f6', alpha=median_plane_alpha, linewidth=0.35, edgecolor='#1e88e5', shade=False)

        xx3, yy3 = np.meshgrid(np.linspace(xlim[0], xlim[1], 2), np.linspace(ylim[0], ylim[1], 2))
        zz3 = np.full_like(xx3, z_med)
        ax.plot_surface(xx3, yy3, zz3, color='#81c784', alpha=median_plane_alpha, linewidth=0.35, edgecolor='#2e7d32', shade=False)

    def _draw_octant_volumes(ax, x_med, y_med, z_med, xlim, ylim, zlim, alpha=0.055):
        """Draw 8 translucent cuboids split by median planes (true 3D octants)."""
        x_segments = [(xlim[0], x_med), (x_med, xlim[1])]
        # bit1 in region id is EarlyMaturity when y <= y_med, so map bit1=1 to lower y-range
        y_segments = [(y_med, ylim[1]), (ylim[0], y_med)]
        z_segments = [(zlim[0], z_med), (z_med, zlim[1])]

        def _cuboid_faces(x0, x1, y0, y1, z0, z1):
            v000 = (x0, y0, z0)
            v001 = (x0, y0, z1)
            v010 = (x0, y1, z0)
            v011 = (x0, y1, z1)
            v100 = (x1, y0, z0)
            v101 = (x1, y0, z1)
            v110 = (x1, y1, z0)
            v111 = (x1, y1, z1)
            return [
                [v000, v001, v011, v010],
                [v100, v101, v111, v110],
                [v000, v001, v101, v100],
                [v010, v011, v111, v110],
                [v000, v010, v110, v100],
                [v001, v011, v111, v101],
            ]

        for xb in (0, 1):
            for yb in (0, 1):
                for zb in (0, 1):
                    rid = xb + 2 * yb + 4 * zb
                    x0, x1 = x_segments[xb]
                    y0, y1 = y_segments[yb]
                    z0, z1 = z_segments[zb]
                    # Skip degenerate boxes that can appear when medians hit limits.
                    if (x1 - x0) <= 1e-12 or (y1 - y0) <= 1e-12 or (z1 - z0) <= 1e-12:
                        continue
                    faces = _cuboid_faces(x0, x1, y0, y1, z0, z1)
                    box = Poly3DCollection(
                        faces,
                        facecolors=region_colors[rid],
                        edgecolors='none',
                        linewidths=0.0,
                        alpha=alpha,
                    )
                    ax.add_collection3d(box)

    def _draw_zero_gain_plane(ax, xlim, ylim):
        """Draw explicit z=0 reference plane so negative gain is visible."""
        xx0, yy0 = np.meshgrid(np.linspace(xlim[0], xlim[1], 2), np.linspace(ylim[0], ylim[1], 2))
        zz0 = np.zeros_like(xx0)
        ax.plot_surface(xx0, yy0, zz0, color='#ffb3b3', alpha=0.16, linewidth=0.25, edgecolor='#d32f2f', shade=False)

    def _region_id_breeding(similarity, ndm_value, yield_value, sim_med, ndm_med, yield_med):
        # bit0: high similarity, bit1: early maturity (smaller NDM), bit2: high yield
        return int(similarity >= sim_med) + 2 * int(ndm_value <= ndm_med) + 4 * int(yield_value >= yield_med)

    region_desc = {
        0: 'LowSimilarity + LateMaturity + LowGain',
        1: 'HighSimilarity + LateMaturity + LowGain',
        2: 'LowSimilarity + EarlyMaturity + LowGain',
        3: 'HighSimilarity + EarlyMaturity + LowGain',
        4: 'LowSimilarity + LateMaturity + HighGain',
        5: 'HighSimilarity + LateMaturity + HighGain',
        6: 'LowSimilarity + EarlyMaturity + HighGain',
        7: 'HighSimilarity + EarlyMaturity + HighGain',
    }

    def _place_labels_sparse(ax, xs, ys, zs, labels, fontsize, alpha=0.75, dist_thr=0.035):
        # Place labels by checking overlap in the final 2D projected view.
        # This matches what humans observe in saved figures.
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        zs = np.asarray(zs, dtype=float)
        x_rng = np.ptp(xs) + 1e-10
        y_rng = np.ptp(ys) + 1e-10
        z_rng = np.ptp(zs) + 1e-10

        n = len(labels)
        if n == 0:
            return

        # Ensure renderer is initialized for accurate text-size measurement.
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_bbox = ax.bbox
        text_fp = FontProperties(size=fontsize, weight='semibold')

        # Project all points once; used to penalize labels covering other points.
        point_pixels = []
        for i in range(n):
            up, vp, _ = proj3d.proj_transform(xs[i], ys[i], zs[i], ax.get_proj())
            pxi, pyi = ax.transData.transform((up, vp))
            point_pixels.append((pxi, pyi))

        # Approximate marker radius in pixels from scatter size (pt^2 -> pt -> px).
        point_r_px = max(4.0, (np.sqrt(marker_size_fixed) * fig.dpi / 72.0) * 0.55)

        xn = (xs - np.min(xs)) / x_rng
        yn = (ys - np.min(ys)) / y_rng
        zn = (zs - np.min(zs)) / z_rng

        density = np.zeros(n, dtype=float)
        for i in range(n):
            dsum = 0.0
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((xn[i] - xn[j]) ** 2 + (yn[i] - yn[j]) ** 2 + 0.35 * (zn[i] - zn[j]) ** 2)
                dsum += np.exp(-9.5 * d)
            density[i] = dsum
        place_order = np.argsort(-density)

        # Priority rule:
        # 1) Evaluate 8 planar directions with short offsets.
        # 2) If overlap remains in 2D, switch to aggressive spread (8 planar + z up/down).
        dirs_8 = [
            (1, 1, 0), (1, 0, 0), (1, -1, 0), (0, -1, 0),
            (-1, -1, 0), (-1, 0, 0), (-1, 1, 0), (0, 1, 0),
        ]
        fallback_dirs = [
            (0, -1, 0), (-1, -1, 0), (1, -1, 0), (-1, 0, 0),
            (1, 0, 0), (-1, 1, 0), (1, 1, 0), (0, 1, 0),
            (0, -1, 1), (-1, -1, 1), (1, -1, 1),
            (0, -1, -1), (-1, -1, -1), (1, -1, -1),
            (0, 0, 1), (0, 0, -1),
        ]
        primary_scales = [0.25, 0.40, 0.55, 0.75]
        fallback_scales = [1.10, 1.50, 2.00, 2.60, 3.20]
        connector_thr = 0.004
        # Stored as (x0, y0, x1, y1) in display pixels.
        placed_boxes = []

        def _bbox_overlap_area(b1, b2):
            x0 = max(b1[0], b2[0])
            y0 = max(b1[1], b2[1])
            x1 = min(b1[2], b2[2])
            y1 = min(b1[3], b2[3])
            if x1 <= x0 or y1 <= y0:
                return 0.0
            return (x1 - x0) * (y1 - y0)

        def _project_to_pixels(xv, yv, zv):
            up, vp, _ = proj3d.proj_transform(xv, yv, zv, ax.get_proj())
            px, py = ax.transData.transform((up, vp))
            return px, py

        def _make_pixel_box(px, py, text, pad_px=2.0):
            w, h, d = renderer.get_text_width_height_descent(text, text_fp, ismath=False)
            # Matplotlib text default alignment: left + baseline.
            x0 = px - pad_px
            y0 = py - d - pad_px
            x1 = px + w + pad_px
            y1 = py + (h - d) + pad_px
            return (x0, y0, x1, y1)

        def _box_intersects_point(box, px, py, r):
            cx = min(max(px, box[0]), box[2])
            cy = min(max(py, box[1]), box[3])
            return (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2

        def _candidate_metrics(lx, ly, lz, idx):
            px, py = _project_to_pixels(lx, ly, lz)
            box = _make_pixel_box(px, py, labels[idx])

            outside = 0.0
            if box[0] < axes_bbox.x0 or box[2] > axes_bbox.x1:
                outside += 1.0
            if box[1] < axes_bbox.y0 or box[3] > axes_bbox.y1:
                outside += 1.0

            overlap_area = 0.0
            for pbox in placed_boxes:
                overlap_area += _bbox_overlap_area(box, pbox)

            # Penalize labels that cover other scatter points in 2D view.
            point_cover_count = 0
            for j, (ppx, ppy) in enumerate(point_pixels):
                if j == idx:
                    continue
                if _box_intersects_point(box, ppx, ppy, point_r_px):
                    point_cover_count += 1

            return px, py, box, outside, overlap_area, point_cover_count

        for idx in place_order:
            dx, dy, dz = _label_offset(idx)

            # Count close neighbors in normalized space to scale offset adaptively.
            neighbor_count = 0
            for j in range(n):
                if j == idx:
                    continue
                d = np.sqrt((xn[idx] - xn[j]) ** 2 + (yn[idx] - yn[j]) ** 2 + 0.35 * (zn[idx] - zn[j]) ** 2)
                if d < dist_thr:
                    neighbor_count += 1

            scale = 1.0 + min(0.7, 0.12 * neighbor_count)
            # Keep labels visible across very different z-axis scales (yield vs gain-rate).
            dz_abs = np.clip(dz * z_rng, 0.03, 10.0)

            # Hard caps to prevent labels from flying far away, but allow enough room in dense center.
            max_dx = max(0.06 * x_rng, 0.018)
            max_dy = max(0.08 * y_rng, 0.65)
            max_dz = max(0.12 * z_rng, 0.12)

            px0, py0 = _project_to_pixels(xs[idx], ys[idx], zs[idx])

            best_xyz = (xs[idx], ys[idx], zs[idx])
            best_proj = (px0, py0)
            best_box = _make_pixel_box(px0, py0, labels[idx])
            best_overlap = 1e9
            best_cost = 1e9

            accepted = False
            has_overlap_pressure = False

            # Stage-1: keep labels close to points with full 8-direction search.
            for local_scale in primary_scales:
                x_step = min(abs(dx) * scale * local_scale, max_dx)
                y_step = min(abs(dy) * scale * local_scale, max_dy)
                z_step = min(dz_abs * scale * local_scale, max_dz)
                for sx, sy, sz in dirs_8:
                    lx = xs[idx] + sx * x_step
                    ly = ys[idx] + sy * y_step
                    lz = zs[idx] + 0.45 * z_step
                    px, py, box, outside, overlap_area, point_cover_count = _candidate_metrics(lx, ly, lz, idx)
                    disp_proj = np.hypot(px - px0, py - py0)
                    if outside == 0 and overlap_area <= 0.0 and point_cover_count == 0:
                        best_xyz = (lx, ly, lz)
                        best_proj = (px, py)
                        best_box = box
                        accepted = True
                        break

                    if overlap_area > 0.0 or point_cover_count > 0:
                        has_overlap_pressure = True

                    dir_penalty = 0.0
                    if sy > 0:
                        dir_penalty += 0.09
                    if sx > 0:
                        dir_penalty += 0.05
                    # Stage-1 keeps labels relatively near points, but still reacts to overlap.
                    cost = 0.35 * overlap_area + 0.08 * disp_proj + 20.0 * outside + 20.0 * point_cover_count + dir_penalty
                    if (overlap_area < best_overlap - 1e-12) or (abs(overlap_area - best_overlap) <= 1e-12 and cost < best_cost):
                        best_overlap = overlap_area
                        best_cost = cost
                        best_xyz = (lx, ly, lz)
                        best_proj = (px, py)
                        best_box = box
                if accepted:
                    break

            # Stage-2: if overlap remains, try below / left-below with short connector line.
            if not accepted:
                for local_scale in fallback_scales:
                    x_step = min(abs(dx) * scale * local_scale, max_dx)
                    y_step = min(abs(dy) * scale * local_scale, max_dy)
                    z_step = min(dz_abs * scale * local_scale, max_dz)
                    for sx, sy, sz in fallback_dirs:
                        lx = xs[idx] + sx * x_step
                        ly = ys[idx] + sy * y_step
                        lz = zs[idx] + sz * z_step
                        px, py, box, outside, overlap_area, point_cover_count = _candidate_metrics(lx, ly, lz, idx)
                        disp_proj = np.hypot(px - px0, py - py0)
                        if outside == 0 and overlap_area <= 0.0 and point_cover_count == 0:
                            best_xyz = (lx, ly, lz)
                            best_proj = (px, py)
                            best_box = box
                            accepted = True
                            break

                        dir_penalty = 0.0
                        if sy > 0:
                            dir_penalty += 0.12
                        if sx > 0:
                            dir_penalty += 0.06
                        # Stage-2 aggressively prioritizes removing overlap in 2D view.
                        if has_overlap_pressure:
                            cost = 1.20 * overlap_area + 0.035 * disp_proj + 22.0 * outside + 28.0 * point_cover_count + dir_penalty
                        else:
                            cost = 0.20 * overlap_area + 0.07 * disp_proj + 20.0 * outside + 14.0 * point_cover_count + dir_penalty
                        if (overlap_area < best_overlap - 1e-12) or (abs(overlap_area - best_overlap) <= 1e-12 and cost < best_cost):
                            best_overlap = overlap_area
                            best_cost = cost
                            best_xyz = (lx, ly, lz)
                            best_proj = (px, py)
                            best_box = box
                    if accepted:
                        break

            lx, ly, lz = best_xyz
            placed_boxes.append(best_box)

            move_norm = np.sqrt(
                ((lx - xs[idx]) / x_rng) ** 2 +
                ((ly - ys[idx]) / y_rng) ** 2 +
                0.35 * ((lz - zs[idx]) / z_rng) ** 2
            )
            if move_norm >= connector_thr:
                ax.plot(
                    [xs[idx], lx],
                    [ys[idx], ly],
                    [zs[idx], lz],
                    color='#303030',
                    linewidth=0.7,
                    alpha=0.65,
                    zorder=9,
                )

            ax.text(
                lx,
                ly,
                lz,
                labels[idx],
                fontsize=fontsize,
                alpha=alpha,
                color='#111111',
                fontweight='semibold',
                zorder=10,
                clip_on=False,
            )

    def _draw_plane_projections(ax, xs, ys, zs, colors, xlim, ylim, zlim):
        z0 = zlim[0]
        y0 = ylim[0]
        x0 = xlim[0]
        ax.scatter(xs, ys, np.full_like(xs, z0), s=24, c=colors, alpha=projection_alpha, edgecolors='none')
        ax.scatter(xs, np.full_like(xs, y0), zs, s=24, c=colors, alpha=projection_alpha, edgecolors='none')
        ax.scatter(np.full_like(xs, x0), ys, zs, s=24, c=colors, alpha=projection_alpha, edgecolors='none')

    def _apply_consistent_3d_direction(ax, elev, azim):
        """Keep 3D axis reading direction consistent across all figures.
        X: left->right increasing, Y: inside->outside increasing, Z: bottom->top increasing.
        """
        ax.view_init(elev=elev, azim=azim)
        if ax.xaxis_inverted():
            ax.invert_xaxis()
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        if ax.zaxis_inverted():
            ax.invert_zaxis()

    w_feature = 0.35
    w_gain = 0.45
    w_ndm = 0.20

    # Global limits (shared) so planes extend across full coordinate system, not only local point cloud.
    all_diffs = np.array([[genotype_data[g]['feature_diffs'][t] for g in names] for t in range(n_timepoints)], dtype=float)
    diff_min, diff_max = np.nanmin(all_diffs), np.nanmax(all_diffs)
    all_similarity = 1.0 - (all_diffs - diff_min) / (diff_max - diff_min + 1e-10)
    all_similarity_disp = np.array([_rank_spread(row) for row in all_similarity])
    sim_min, sim_max = np.nanmin(all_similarity_disp), np.nanmax(all_similarity_disp)
    ndm_min, ndm_max = np.nanmin(ndm_values), np.nanmax(ndm_values)
    y_min, y_max = np.nanmin(yields), np.nanmax(yields)
    gain_min, gain_max = np.nanmin(gain_norm), np.nanmax(gain_norm)
    gain_rate_min, gain_rate_max = np.nanmin(gain_rates), np.nanmax(gain_rates)

    pad_x = 0.08 * (sim_max - sim_min + 1e-10)
    pad_y = 0.08 * (ndm_max - ndm_min + 1e-10)
    pad_z = 0.10 * (y_max - y_min + 1e-10)
    pad_gain = 0.08 * (gain_max - gain_min + 1e-10)
    pad_gain_rate = 0.10 * (gain_rate_max - gain_rate_min + 1e-10)

    xlim_global = (sim_min - pad_x, sim_max + pad_x)
    ylim_global = (ndm_min - pad_y, ndm_max + pad_y)
    zlim_global = (max(0, y_min - pad_z), y_max + pad_z)
    zlim_gain_global = (max(0.0, gain_min - pad_gain), min(1.0, gain_max + pad_gain))
    zlim_gain_rate_global = (gain_rate_min - pad_gain_rate, gain_rate_max + pad_gain_rate)
    zlim_gain_rate_global = (min(zlim_gain_rate_global[0], -0.02), max(zlim_gain_rate_global[1], 0.02))
    gain_threshold = 0.0

    # Z8 tracking across 6 matched timepoint views.
    # Yield-Z8:  high similarity + early maturity + high NoControl yield
    # Gain-Z8:   high similarity + early maturity + high gain rate
    # Intersection-Z8: both Yield-Z8 and Gain-Z8 in the same matched view
    ideal_count_yield_map = {g: 0 for g in names}
    ideal_count_gain_map = {g: 0 for g in names}
    yield_r8_views = {g: [False] * n_timepoints for g in names}
    gain_r8_views = {g: [False] * n_timepoints for g in names}

    # 图1：FeatureSimilarity(x)-NDM(y)-Yield(z) 三维时序
    n_cols = 3
    n_rows = (n_timepoints + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(22.2, 6.0 * n_rows))
    gs_yield = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        hspace=0.06,
        wspace=0.015,
    )

    for t in range(n_timepoints):
        row = t // n_cols
        col = t % n_cols
        ax = fig.add_subplot(gs_yield[row, col], projection='3d')

        diff_data = np.array([genotype_data[g]['feature_diffs'][t] for g in names], dtype=float)
        similarity_raw_t = 1.0 - (diff_data - diff_min) / (diff_max - diff_min + 1e-10)
        x_data = _rank_spread(similarity_raw_t)
        low_diff_t = similarity_raw_t.copy()
        composite_score_t = w_feature * (1.0 - low_diff_t) + w_gain * y_norm + w_ndm * ndm_earliness

        x_med = np.nanmedian(x_data)
        y_med = np.nanmedian(ndm_values)
        z_med = np.nanmedian(yields)
        regions_t = [
            _region_id_breeding(x_data[i], ndm_values[i], yields[i], x_med, y_med, z_med)
            for i in range(len(names))
        ]
        colors_t = [region_colors[r] for r in regions_t]

        sc = ax.scatter(
            x_data,
            ndm_values,
            yields,
            s=marker_size_fixed,
            c=colors_t,
            alpha=0.78,
            edgecolors='black',
            linewidth=0.6,
        )

        ax.set_xlim(*xlim_global)
        ax.set_ylim(*ylim_global)
        ax.set_zlim(*zlim_global)
        _draw_octant_volumes(ax, x_med, y_med, z_med, xlim_global, ylim_global, zlim_global, alpha=0.048)
        ax.plot([xlim_global[0], xlim_global[1]], [y_med, y_med], [z_med, z_med], '--', color='gray', alpha=0.65, linewidth=1.1)
        ax.plot([x_med, x_med], [ylim_global[0], ylim_global[1]], [z_med, z_med], '--', color='gray', alpha=0.65, linewidth=1.1)
        ax.plot([x_med, x_med], [y_med, y_med], [zlim_global[0], zlim_global[1]], '--', color='gray', alpha=0.65, linewidth=1.1)
        _draw_median_planes(ax, x_med, y_med, z_med, xlim_global, ylim_global, zlim_global)

        # Yield-Z8 frequency at each timepoint (Z8 == region id 7)
        for i, g in enumerate(names):
            is_r8_yield = regions_t[i] == 7
            yield_r8_views[g][t] = is_r8_yield
            if is_r8_yield:
                ideal_count_yield_map[g] += 1

        # Reduce axis crowding (NDM and decimal x-axis)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.set_xlabel('Feature Similarity', fontsize=12, labelpad=8)
        ax.set_ylabel('NDM', fontsize=12, labelpad=7)
        ax.set_zlabel('NoControl Yield', fontsize=12, labelpad=6)
        ax.zaxis.label.set_clip_on(False)
        ax.text2D(
            0.50,
            -0.015,
            f'$T_{{{t+1}}}$: {date_labels[t]}',
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=12,
            fontweight='bold',
        )
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        _apply_consistent_3d_direction(ax, elev=22, azim=-58)
        _place_labels_sparse(ax, x_data, ndm_values, yields, names, fontsize=text_size_timeseries, alpha=0.76)

    region_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Z{rid+1}', markerfacecolor=region_colors[rid],
                   markeredgecolor='black', markersize=9)
        for rid in range(8)
    ]
    left_margin = 0.008
    right_margin = 0.988
    top_margin = 0.986
    bottom_margin = 0.095
    center_x = 0.5 * (left_margin + right_margin)
    fig.legend(
        handles=region_handles,
        loc='lower center',
        ncol=8,
        frameon=False,
        bbox_to_anchor=(center_x, 0.016),
        fontsize=12,
    )
    fig.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)

    save_path = analyzer.output_dir / f'feature_vs_yield_timeseries_{feature_type}.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches=None, pad_inches=0.03, facecolor='white')
    plt.close()
    print(f"✓ Timepoints 3D saved: {save_path}")

    # Average vectors are kept for scoring/ranking, but standalone average plots are disabled.
    x_data_mean_diff = np.array([genotype_data[g]['mean_diff'] for g in names], dtype=float)
    x_data_mean_raw = 1.0 - (x_data_mean_diff - diff_min) / (diff_max - diff_min + 1e-10)
    x_data_mean = _rank_spread(x_data_mean_raw)
    low_similarity_mean = 1.0 - x_data_mean_raw
    composite_score_mean = w_feature * low_similarity_mean + w_gain * y_norm + w_ndm * ndm_earliness

    x_med_avg_y = np.nanmedian(x_data_mean)
    y_med_avg_y = np.nanmedian(ndm_values)
    z_med_avg_y = np.nanmedian(yields)
    regions_avg_y = [
        _region_id_breeding(x_data_mean[i], ndm_values[i], yields[i], x_med_avg_y, y_med_avg_y, z_med_avg_y)
        for i in range(len(names))
    ]
    colors_avg_y = [region_colors[r] for r in regions_avg_y]

    xlim_avg = (
        np.nanmin(x_data_mean) - 0.08 * (np.ptp(x_data_mean) + 1e-10),
        np.nanmax(x_data_mean) + 0.08 * (np.ptp(x_data_mean) + 1e-10),
    )
    ylim_avg = ylim_global

    # 图3：FeatureSimilarity(x)-NDM(y)-NormalizedGain(z) 三维时序
    fig_gain = plt.figure(figsize=(22.2, 6.0 * n_rows))
    gs_gain = GridSpec(
        n_rows,
        n_cols,
        figure=fig_gain,
        hspace=0.06,
        wspace=0.015,
    )
    for t in range(n_timepoints):
        row = t // n_cols
        col = t % n_cols
        ax = fig_gain.add_subplot(gs_gain[row, col], projection='3d')

        diff_data = np.array([genotype_data[g]['feature_diffs'][t] for g in names], dtype=float)
        similarity_raw_t = 1.0 - (diff_data - diff_min) / (diff_max - diff_min + 1e-10)
        x_data = _rank_spread(similarity_raw_t)

        x_med = np.nanmedian(x_data)
        y_med = np.nanmedian(ndm_values)
        z_med = gain_threshold
        regions_t = [
            _region_id_breeding(x_data[i], ndm_values[i], gain_rates[i], x_med, y_med, z_med)
            for i in range(len(names))
        ]
        colors_t = [region_colors[r] for r in regions_t]

        ax.scatter(
            x_data,
            ndm_values,
            gain_rates,
            s=marker_size_fixed,
            c=colors_t,
            alpha=0.78,
            edgecolors='black',
            linewidth=0.6,
        )

        ax.set_xlim(*xlim_global)
        ax.set_ylim(*ylim_global)
        ax.set_zlim(*zlim_gain_rate_global)
        _draw_octant_volumes(ax, x_med, y_med, z_med, xlim_global, ylim_global, zlim_gain_rate_global, alpha=0.048)
        ax.plot([xlim_global[0], xlim_global[1]], [y_med, y_med], [z_med, z_med], '--', color='gray', alpha=0.65, linewidth=1.1)
        ax.plot([x_med, x_med], [ylim_global[0], ylim_global[1]], [z_med, z_med], '--', color='gray', alpha=0.65, linewidth=1.1)
        ax.plot([x_med, x_med], [y_med, y_med], [zlim_gain_rate_global[0], zlim_gain_rate_global[1]], '--', color='gray', alpha=0.65, linewidth=1.1)
        _draw_median_planes(ax, x_med, y_med, z_med, xlim_global, ylim_global, zlim_gain_rate_global)
        _draw_zero_gain_plane(ax, xlim_global, ylim_global)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax.set_xlabel('Feature Similarity', fontsize=12, labelpad=8)
        ax.set_ylabel('NDM', fontsize=12, labelpad=7)
        ax.set_zlabel('Yield Gain Rate', fontsize=12, labelpad=6)
        ax.zaxis.label.set_clip_on(False)
        ax.text2D(
            0.50,
            -0.015,
            f'$T_{{{t+1}}}$: {date_labels[t]}',
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=12,
            fontweight='bold',
        )
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        _apply_consistent_3d_direction(ax, elev=22, azim=-58)
        _place_labels_sparse(ax, x_data, ndm_values, gain_rates, names, fontsize=text_size_timeseries, alpha=0.95, dist_thr=0.03)

        # Gain-Z8 frequency at each timepoint (Z8 == region id 7)
        for i, g in enumerate(names):
            is_r8_gain = regions_t[i] == 7
            gain_r8_views[g][t] = is_r8_gain
            if is_r8_gain:
                ideal_count_gain_map[g] += 1

    fig_gain_left = 0.008
    fig_gain_right = 0.988
    fig_gain_top = 0.986
    fig_gain_bottom = 0.095
    fig_gain_center_x = 0.5 * (fig_gain_left + fig_gain_right)
    fig_gain.legend(
        handles=region_handles,
        loc='lower center',
        ncol=8,
        frameon=False,
        bbox_to_anchor=(fig_gain_center_x, 0.016),
        fontsize=12,
    )
    fig_gain.subplots_adjust(
        left=fig_gain_left,
        right=fig_gain_right,
        top=fig_gain_top,
        bottom=fig_gain_bottom,
    )
    save_path_gain = analyzer.output_dir / f'feature_vs_gain_timeseries_{feature_type}.png'
    _save_png_and_pdf(save_path_gain, dpi=300, bbox_inches=None, pad_inches=0.03, facecolor='white')
    plt.close()
    print(f"✓ Gain-rate 3D time-series saved: {save_path_gain}")

    # Average region assignment (no standalone average figure output)
    x_med_gain = np.nanmedian(x_data_mean)
    y_med_gain = np.nanmedian(ndm_values)
    z_med_gain = gain_threshold
    regions_gain_avg = [
        _region_id_breeding(x_data_mean[i], ndm_values[i], gain_rates[i], x_med_gain, y_med_gain, z_med_gain)
        for i in range(len(names))
    ]

    print('✓ Extra 3D maturity-view variants skipped.')

    results_df = pd.DataFrame({
        'genotype': names,
        'mean_feature_diff': x_data_mean_diff,
        'mean_feature_similarity': x_data_mean_raw,
        'mean_feature_similarity_display': x_data_mean,
        'control_yield': control_yields,
        'nocontrol_yield': yields,
        'yield_gain_rate': gain_rates,
        'yield_gain_rate_norm': gain_norm,
        'nocontrol_ndm': ndm_values,
        'marker_size_fixed': marker_size_fixed,
        'region_id_avg': regions_gain_avg,
        'region_code_avg': [f'Z{rid+1}' for rid in regions_gain_avg],
        'region_desc_avg': [region_desc[rid] for rid in regions_gain_avg],
        'ndm_earliness_norm': ndm_earliness,
        'score_low_similarity_norm': low_similarity_mean,
        'score_high_nocontrol_yield_norm': y_norm,
        'score_high_gain_norm': gain_norm,
        'composite_score': composite_score_mean,
        'w_feature_low_similarity': w_feature,
        'w_nocontrol_yield_high': w_gain,
        'w_gain_high': w_gain,
        'w_ndm_early_maturity': w_ndm
    })

    # Add Z8 frequency rankings (yield / gain / intersection)
    ideal_count_intersection_map = {
        g: sum(yield_r8_views[g][k] and gain_r8_views[g][k] for k in range(n_timepoints))
        for g in names
    }

    results_df['ideal_count_yield_6views'] = results_df['genotype'].map(ideal_count_yield_map)
    results_df['ideal_ratio_yield_6views'] = results_df['ideal_count_yield_6views'] / n_timepoints
    results_df['ideal_count_gain_6views'] = results_df['genotype'].map(ideal_count_gain_map)
    results_df['ideal_ratio_gain_6views'] = results_df['ideal_count_gain_6views'] / n_timepoints
    results_df['ideal_count_intersection_6views'] = results_df['genotype'].map(ideal_count_intersection_map)
    results_df['ideal_ratio_intersection_6views'] = results_df['ideal_count_intersection_6views'] / n_timepoints

    # Backward-compatible aliases
    results_df['ideal_count_6views'] = results_df['ideal_count_intersection_6views']
    results_df['ideal_ratio_6views'] = results_df['ideal_ratio_intersection_6views']

    results_df['rank_by_score'] = results_df['composite_score'].rank(ascending=False, method='min').astype(int)
    results_df['rank_by_ideal_yield_frequency'] = results_df['ideal_count_yield_6views'].rank(ascending=False, method='min').astype(int)
    results_df['rank_by_ideal_gain_frequency'] = results_df['ideal_count_gain_6views'].rank(ascending=False, method='min').astype(int)
    results_df['rank_by_ideal_intersection_frequency'] = results_df['ideal_count_intersection_6views'].rank(ascending=False, method='min').astype(int)
    results_df['rank_by_ideal_frequency'] = results_df['rank_by_ideal_intersection_frequency']

    for t in range(n_timepoints):
        results_df[f'feature_diff_t{t+1}'] = [genotype_data[g]['feature_diffs'][t] for g in names]

    # Per-view Z8 membership flags for exact cross-feature intersections.
    for t in range(n_timepoints):
        results_df[f'yield_r8_view_t{t+1}'] = [int(yield_r8_views[g][t]) for g in names]
        results_df[f'gain_r8_view_t{t+1}'] = [int(gain_r8_views[g][t]) for g in names]

    csv_path = analyzer.output_dir / f'feature_vs_yield_timeseries_data_{feature_type}.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Data saved: {csv_path}")

    # Save explicit region legend as CSV (for paper/table use)
    region_legend_rows = []
    for rid in range(8):
        region_legend_rows.append({
            'region_id': rid,
            'region_code': f'Z{rid+1}',
            'definition': region_desc[rid],
            'color_hex': region_colors[rid],
            'meaning_cn': (
                '高相似+早熟+高增产' if rid == 7 else
                '高相似+早熟+低增产' if rid == 3 else
                '高相似+晚熟+高增产' if rid == 5 else
                '高相似+晚熟+低增产' if rid == 1 else
                '低相似+早熟+高增产(理想区)' if rid == 6 else
                '低相似+早熟+低增产' if rid == 2 else
                '低相似+晚熟+高增产' if rid == 4 else
                '低相似+晚熟+低增产'
            )
        })
    region_legend_df = pd.DataFrame(region_legend_rows)
    region_legend_csv = analyzer.output_dir / f'region_legend_{feature_type}.csv'
    region_legend_df.to_csv(region_legend_csv, index=False, encoding='utf-8-sig')
    print(f"✓ Region legend saved: {region_legend_csv}")

    # Dual ranking export: method-1 score, method-2 ideal frequency
    ranking_df = results_df[[
        'genotype',
        'composite_score',
        'rank_by_score',
        'ideal_count_yield_6views',
        'ideal_ratio_yield_6views',
        'rank_by_ideal_yield_frequency',
        'ideal_count_gain_6views',
        'ideal_ratio_gain_6views',
        'rank_by_ideal_gain_frequency',
        'ideal_count_intersection_6views',
        'ideal_ratio_intersection_6views',
        'rank_by_ideal_intersection_frequency',
        'ideal_count_6views',
        'ideal_ratio_6views',
        'rank_by_ideal_frequency',
        'mean_feature_similarity',
        'nocontrol_ndm',
        'yield_gain_rate',
        'yield_gain_rate_norm',
        'nocontrol_yield',
        'region_code_avg',
        'region_desc_avg',
    ]].copy()

    ranking_score_csv = analyzer.output_dir / f'ranking_by_score_{feature_type}.csv'
    ranking_df.sort_values('rank_by_score').to_csv(ranking_score_csv, index=False, encoding='utf-8-sig')
    print(f"✓ Score ranking saved: {ranking_score_csv}")

    ranking_ideal_csv = analyzer.output_dir / f'ranking_by_ideal_frequency_{feature_type}.csv'
    ranking_df.sort_values(['rank_by_ideal_intersection_frequency', 'rank_by_score']).to_csv(ranking_ideal_csv, index=False, encoding='utf-8-sig')
    print(f"✓ Ideal-frequency ranking saved: {ranking_ideal_csv}")

    # Save region summary report as TXT
    region_report_path = analyzer.output_dir / f'region_summary_{feature_type}.txt'
    with open(region_report_path, 'w', encoding='utf-8') as f:
        f.write('=' * 90 + '\n')
        f.write(f'3D REGION SUMMARY ({feature_type.upper()})\n')
        f.write('=' * 90 + '\n\n')
        f.write('Region Definitions (median-based):\n')
        f.write('- Similarity: Low <= median (IDEAL wants low similarity)\n')
        f.write('- Maturity: Early <= median NDM\n')
        f.write('- Gain: High >= 0 (positive gain rate), Low < 0 (negative gain rate)\n\n')

        for rid in range(8):
            code = f'Z{rid+1}'
            desc = region_desc[rid]
            members = results_df[results_df['region_id_avg'] == rid].sort_values('yield_gain_rate_norm', ascending=False)
            f.write(f'{code}: {desc}\n')
            f.write(f'  Count: {len(members)}\n')
            if len(members) > 0:
                names_line = ', '.join(members['genotype'].tolist())
                f.write(f'  Genotypes: {names_line}\n')
            else:
                f.write('  Genotypes: None\n')
            f.write('\n')

        f.write('=' * 90 + '\n')
        f.write('RANKING METHOD 1: BY COMPOSITE SCORE\n')
        f.write('=' * 90 + '\n')
        m1 = ranking_df.sort_values('rank_by_score').head(30)
        for _, row in m1.iterrows():
            f.write(
                f"#{int(row['rank_by_score']):2d} {row['genotype']:<15} | "
                f"Score={row['composite_score']:.3f} | "
                f"Sim={row['mean_feature_similarity']:.3f} | "
                f"NDM={row['nocontrol_ndm']:.1f} | "
                f"GainNorm={row['yield_gain_rate_norm']:.3f} | "
                f"GainRate={row['yield_gain_rate']*100:.1f}%\n"
            )

        f.write('\n' + '=' * 90 + '\n')
        f.write('RANKING METHOD 2: BY IDEAL REGION INTERSECTION FREQUENCY (6 matched views)\n')
        f.write('=' * 90 + '\n')
        m2 = ranking_df.sort_values(['rank_by_ideal_intersection_frequency', 'rank_by_score']).head(30)
        for _, row in m2.iterrows():
            f.write(
                f"#{int(row['rank_by_ideal_intersection_frequency']):2d} {row['genotype']:<15} | "
                f"YieldR8={int(row['ideal_count_yield_6views'])}/6 ({row['ideal_ratio_yield_6views']*100:4.1f}%) | "
                f"GainR8={int(row['ideal_count_gain_6views'])}/6 ({row['ideal_ratio_gain_6views']*100:4.1f}%) | "
                f"Intersect={int(row['ideal_count_intersection_6views'])}/6 ({row['ideal_ratio_intersection_6views']*100:4.1f}%) | "
                f"ScoreRank=#{int(row['rank_by_score'])}\n"
            )

    print(f"✓ Region summary report saved: {region_report_path}")

    print(f"\n{'='*80}")
    print("Statistical Summary")
    print(f"{'='*80}")
    print(f"\nCorrelation at each time point (Feature Diff vs Gain Rate):")
    for t in range(n_timepoints):
        x_t = [genotype_data[g]['feature_diffs'][t] for g in names]
        corr, p_val = stats.pearsonr(x_t, gain_rates)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  Time {t+1} ({date_labels[t]}): r={corr:6.3f}, p={p_val:.4f} {sig}")

    print(f"\n{'='*80}\n")

    return results_df


def analyze_feature_yield_relationship(feature_type='dinov3'):
    """
    分析不同时间点的特征差异(control - nocontrol)与产量的关系
    
    Args:
        feature_type: 'dinov3', 'vi', 'fusion'
    
    生成可视化：
    1. 每个时间点的特征差异 vs 产量散点图（30个品种）
    2. 所有时间点平均特征差异 vs 产量关系图
    """
    print(f"\n{'='*80}")
    print(f"Feature-Yield Relationship Analysis ({feature_type.upper()})")
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
    
    # 获取时间点数量（假设所有品种有相同数量的时间点）
    sample_plot = list(nocontrol_features.keys())[0]
    n_timepoints = nocontrol_features[sample_plot]['features'].shape[0]
    print(f"Number of time points: {n_timepoints}")
    
    # 时间点标签（从metadata获取实际日期）
    metadata_path = analyzer.data_dir / 'dataset_metadata.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 从第一个nocontrol样本获取日期
    first_nocontrol = list(nocontrol_features.keys())[0]
    dates = [img['date'] for img in metadata[first_nocontrol]['image_sequence']]
    date_labels = [d[5:] for d in dates]  # 简化日期格式 MM-DD
    
    print(f"Time points: {dates}")
    
    # 收集所有品种的数据
    genotypes = sorted(analyzer.nocontrol_df['genotype'].unique())
    print(f"Number of genotypes: {len(genotypes)}")
    
    # 为每个品种计算特征差异和产量
    genotype_data = {}
    
    for genotype in genotypes:
        # 获取该品种的control和nocontrol plot
        control_plots = [pid for pid, info in control_features.items() 
                        if info['genotype'] == genotype]
        nocontrol_plots = [pid for pid, info in nocontrol_features.items() 
                          if info['genotype'] == genotype]
        
        if not control_plots or not nocontrol_plots:
            continue
        
        # 平均每个重复的特征
        control_feat_mean = np.mean([control_features[pid]['features'] 
                                     for pid in control_plots], axis=0)  # (n_timepoints, n_features)
        nocontrol_feat_mean = np.mean([nocontrol_features[pid]['features'] 
                                       for pid in nocontrol_plots], axis=0)
        
        # 计算每个时间点的特征差异（欧氏距离）
        feature_diffs = []
        for t in range(n_timepoints):
            diff = np.linalg.norm(control_feat_mean[t] - nocontrol_feat_mean[t])
            feature_diffs.append(diff)
        
        # 获取产量
        genotype_yield = analyzer.nocontrol_df[
            analyzer.nocontrol_df['genotype'] == genotype
        ]['grain_yield'].mean()
        
        genotype_data[genotype] = {
            'feature_diffs': feature_diffs,  # 每个时间点的特征差异
            'yield': genotype_yield
        }
    
    print(f"Valid genotypes: {len(genotype_data)}")
    
    # 创建可视化
    n_cols = 3
    n_rows = (n_timepoints + 2) // n_cols  # +1 for average plot
    fig = plt.figure(figsize=(18, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.25)
    
    # 为每个时间点创建散点图
    all_feature_diffs = []  # 用于计算均值
    
    for t in range(n_timepoints):
        row = t // n_cols
        col = t % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # 提取该时间点所有品种的数据
        x_data = [genotype_data[g]['feature_diffs'][t] for g in genotype_data.keys()]
        y_data = [genotype_data[g]['yield'] for g in genotype_data.keys()]
        
        all_feature_diffs.append(x_data)
        
        # 散点图
        ax.scatter(x_data, y_data, alpha=0.6, s=100, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # 添加趋势线
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_data), max(x_data), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # 计算相关系数
            corr, p_value = stats.pearsonr(x_data, y_data)
            ax.text(0.05, 0.95, f'r={corr:.3f}\np={p_value:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Feature Difference (L2 norm)', fontsize=11)
        ax.set_ylabel('Grain Yield (kg/ha)', fontsize=11)
        ax.set_title(f'$T_{{{t+1}}}$: {date_labels[t]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 计算所有时间点的平均特征差异
    mean_feature_diffs = []
    yields = []
    genotype_names = []
    
    for genotype, data in genotype_data.items():
        mean_diff = np.mean(data['feature_diffs'])
        mean_feature_diffs.append(mean_diff)
        yields.append(data['yield'])
        genotype_names.append(genotype)
    
    # 添加平均特征差异 vs 产量的图
    row = n_timepoints // n_cols
    col = n_timepoints % n_cols
    ax_mean = fig.add_subplot(gs[row, col])
    
    scatter = ax_mean.scatter(mean_feature_diffs, yields, alpha=0.7, s=150, 
                             c=yields, cmap='RdYlGn', edgecolors='black', linewidth=1)
    
    # 添加趋势线
    if len(mean_feature_diffs) > 1:
        z = np.polyfit(mean_feature_diffs, yields, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(mean_feature_diffs), max(mean_feature_diffs), 100)
        ax_mean.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=3, label='Trend line')
        
        # 计算相关系数
        corr, p_value = stats.pearsonr(mean_feature_diffs, yields)
        ax_mean.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {p_value:.4f}', 
                    transform=ax_mean.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 标注一些极端点
    # 找出产量最高和最低的品种
    sorted_indices = np.argsort(yields)
    top_3 = sorted_indices[-3:]
    bottom_3 = sorted_indices[:3]
    
    for idx in np.concatenate([top_3, bottom_3]):
        ax_mean.annotate(genotype_names[idx], 
                        (mean_feature_diffs[idx], yields[idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    ax_mean.set_xlabel('Mean Feature Difference (across all time points)', fontsize=12, fontweight='bold')
    ax_mean.set_ylabel('Grain Yield (kg/ha)', fontsize=12, fontweight='bold')
    ax_mean.set_title(f'Average Feature Difference vs Yield\n({len(genotype_names)} genotypes)', 
                     fontsize=13, fontweight='bold')
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend()
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax_mean)
    cbar.set_label('Grain Yield (kg/ha)', fontsize=10)
    
    # 保存图片
    save_path = analyzer.output_dir / f'feature_diff_yield_relationship_{feature_type}.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Visualization saved: {save_path}")
    
    # 保存数据到CSV
    results_df = pd.DataFrame({
        'genotype': genotype_names,
        'mean_feature_diff': mean_feature_diffs,
        'grain_yield': yields
    })
    
    # 添加每个时间点的特征差异
    for t in range(n_timepoints):
        results_df[f'feature_diff_t{t+1}_{date_labels[t]}'] = [
            genotype_data[g]['feature_diffs'][t] for g in genotype_names
        ]
    
    csv_path = analyzer.output_dir / f'feature_diff_yield_data_{feature_type}.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Data saved: {csv_path}")
    
    # 打印统计摘要
    print(f"\n{'='*80}")
    print("Statistical Summary")
    print(f"{'='*80}")
    
    overall_corr, overall_p = stats.pearsonr(mean_feature_diffs, yields)
    print(f"\nOverall correlation (mean feature diff vs yield):")
    print(f"  Pearson r = {overall_corr:.4f}")
    print(f"  p-value = {overall_p:.4f}")
    print(f"  Interpretation: {'Significant' if overall_p < 0.05 else 'Not significant'} at α=0.05")
    
    print(f"\nCorrelation at each time point:")
    for t in range(n_timepoints):
        x_data = [genotype_data[g]['feature_diffs'][t] for g in genotype_data.keys()]
        y_data = [genotype_data[g]['yield'] for g in genotype_data.keys()]
        corr, p_val = stats.pearsonr(x_data, y_data)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  Time {t+1} ({date_labels[t]}): r={corr:6.3f}, p={p_val:.4f} {sig}")

