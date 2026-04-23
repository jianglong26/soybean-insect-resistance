"""
Time-series Analysis Plots
时间序列分析可视化
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path
import json

from ..core.analyzer import MultiModalInsectResistanceAnalyzer

try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False

    fig.suptitle(f'{genotype_name}: Control vs NoControl Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    save_path = output_dir / f'image_comparison_{genotype_folder}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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
        
        nocontrol_yield = analyzer.nocontrol_df[
            analyzer.nocontrol_df['genotype'] == genotype
        ]['grain_yield'].mean()
        
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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
    时序分析：特征差异 vs NoControl产量
    为每个时间点 + 平均值生成气泡图（大小由NoControl NDM决定）
    
    Args:
        feature_type: 'dinov3', 'vi', 'fusion'
    """
    print(f"\n{'='*80}")
    print(f"Feature Difference vs Yield Time-series Analysis ({feature_type.upper()})")
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
    
    # 获取时间点和日期
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

    def _inverse_bubble_sizes(values, size_min=90.0, size_max=320.0):
        """Smaller value -> larger bubble."""
        arr = np.asarray(values, dtype=float)
        sizes = np.full(arr.shape, (size_min + size_max) / 2.0, dtype=float)
        valid = np.isfinite(arr)
        if valid.sum() >= 2:
            vmin = np.nanmin(arr[valid])
            vmax = np.nanmax(arr[valid])
            if vmax - vmin > 1e-10:
                norm = (arr[valid] - vmin) / (vmax - vmin)
                sizes[valid] = size_max - norm * (size_max - size_min)
        return sizes
    
    # 收集数据
    genotypes = sorted(analyzer.nocontrol_df['genotype'].unique())
    genotype_data = {}
    
    for genotype in genotypes:
        control_plots = [pid for pid, info in control_features.items() 
                        if info['genotype'] == genotype]
        nocontrol_plots = [pid for pid, info in nocontrol_features.items() 
                          if info['genotype'] == genotype]
        
        if not control_plots or not nocontrol_plots:
            continue
        
        # 计算特征差异
        control_feat_mean = np.mean([control_features[pid]['features'] 
                                     for pid in control_plots], axis=0)
        nocontrol_feat_mean = np.mean([nocontrol_features[pid]['features'] 
                                       for pid in nocontrol_plots], axis=0)
        
        feature_diffs = [np.linalg.norm(control_feat_mean[t] - nocontrol_feat_mean[t]) 
                        for t in range(n_timepoints)]
        
        # 获取NoControl产量
        nocontrol_yield = analyzer.nocontrol_df[
            analyzer.nocontrol_df['genotype'] == genotype
        ]['grain_yield'].mean()

        nocontrol_ndm = pd.to_numeric(
            analyzer.nocontrol_df[analyzer.nocontrol_df['genotype'] == genotype]['ndm'],
            errors='coerce'
        ).mean()
        
        genotype_data[genotype] = {
            'feature_diffs': feature_diffs,
            'nocontrol_yield': nocontrol_yield,
            'nocontrol_ndm': nocontrol_ndm,
            'mean_diff': np.mean(feature_diffs)
        }
    
    print(f"Valid genotypes: {len(genotype_data)}")
    
    # 创建图1：仅包含所有时间点（不包括平均值）
    n_cols = 3
    n_rows = (n_timepoints + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(16.5, 5.2 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.18, wspace=0.12)
    
    names = list(genotype_data.keys())
    yields = [genotype_data[g]['nocontrol_yield'] for g in names]
    ndm_values = np.array([genotype_data[g]['nocontrol_ndm'] for g in names], dtype=float)

    # NDM越小，气泡越大（突出早熟）
    bubble_size_min = 90
    bubble_size_max = 320
    bubble_sizes = np.full(len(names), (bubble_size_min + bubble_size_max) / 2.0)
    ndm_earliness = np.full(len(names), 0.5)  # 默认中性值
    valid_ndm = np.isfinite(ndm_values)
    if valid_ndm.sum() >= 2 and (np.nanmax(ndm_values[valid_ndm]) - np.nanmin(ndm_values[valid_ndm])) > 1e-10:
        ndm_min = np.nanmin(ndm_values[valid_ndm])
        ndm_max = np.nanmax(ndm_values[valid_ndm])
        ndm_norm = (ndm_values[valid_ndm] - ndm_min) / (ndm_max - ndm_min)
        bubble_sizes[valid_ndm] = bubble_size_max - ndm_norm * (bubble_size_max - bubble_size_min)
        ndm_earliness[valid_ndm] = 1.0 - ndm_norm
        print(f"NDM range (NoControl): [{ndm_min:.1f}, {ndm_max:.1f}] (smaller NDM -> larger bubble)")
    else:
        print("Warning: NDM data is missing or has no variation; using default bubble size.")

    # 颜色综合得分权重（总和=1.0）
    # 平衡逻辑：特征差异和产量占主要权重，早熟作为附加偏好
    w_feature = 0.40
    w_yield = 0.40
    w_ndm = 0.20
    
    # 为每个时间点画散点图
    for t in range(n_timepoints):
        row = t // n_cols
        col = t % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        x_data = [genotype_data[g]['feature_diffs'][t] for g in names]
        
        # 计算中位数用于象限划分
        median_x = np.median(x_data)
        median_y = np.median(yields)
        
        # 计算每个品种在当前时间点的标准误差（基于不同plot的重复测量）
        # 由于我们已经对每个品种的plots取了平均，这里使用bootstrap估计误差
        x_sem = []
        for g in names:
            control_plots = [pid for pid, info in control_features.items() if info['genotype'] == g]
            nocontrol_plots = [pid for pid, info in nocontrol_features.items() if info['genotype'] == g]
            if len(control_plots) > 1 and len(nocontrol_plots) > 1:
                # 计算不同plot组合的特征差异变异
                diffs = []
                for cp in control_plots:
                    for np_id in nocontrol_plots:
                        diff = np.linalg.norm(control_features[cp]['features'][t] - nocontrol_features[np_id]['features'][t])
                        diffs.append(diff)
                x_sem.append(np.std(diffs) / np.sqrt(len(diffs)))
            else:
                x_sem.append(0)
        
        # 计算综合得分：归一化后加权融合（低特征差异 + 高产量 + 早熟）
        x_norm = (np.array(x_data) - np.min(x_data)) / (np.max(x_data) - np.min(x_data) + 1e-10)
        y_norm = (np.array(yields) - np.min(yields)) / (np.max(yields) - np.min(yields) + 1e-10)
        composite_score = (
            w_feature * (1.0 - x_norm)
            + w_yield * y_norm
            + w_ndm * ndm_earliness
        )
        
        # 气泡图（颜色表示综合得分，大小表示早熟程度）
        scatter = ax.scatter(x_data, yields, s=bubble_sizes, alpha=0.7,
                           c=composite_score, cmap='YlGnBu',
                           edgecolors='black', linewidth=0.8)
        
        # 为所有散点添加水平误差线
        for i, (x, y, sem) in enumerate(zip(x_data, yields, x_sem)):
            if sem > 0:
                ax.errorbar(x, y, xerr=sem, fmt='none', 
                           ecolor='gray', elinewidth=0.8, capsize=2, 
                           capthick=0.8, alpha=0.4, zorder=1)
        
        # 添加象限划分线（基于中位数）
        ax.axvline(median_x, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(median_y, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # 标注所有品种
        all_indices = list(range(len(names)))
        
        # 使用adjustText优化标签位置（标注所有品种）
        if ADJUST_TEXT_AVAILABLE and len(all_indices) > 0:
            texts = []
            for i in all_indices:
                # 根据所在象限使用不同颜色（四个象限四种颜色）
                left_half = x_data[i] < median_x
                upper_half = yields[i] > median_y
                
                if left_half and upper_half:
                    # 左上象限（IDEAL Zone）：深绿色
                    color = 'darkgreen'
                    weight = 'bold'
                    fontsize = 6
                elif not left_half and upper_half:
                    # 右上象限（高产量+高特征差异）：深蓝色
                    color = 'darkblue'
                    weight = 'normal'
                    fontsize = 5.5
                elif left_half and not upper_half:
                    # 左下象限（低产量+低特征差异）：橙色
                    color = 'darkorange'
                    weight = 'normal'
                    fontsize = 5
                else:
                    # 右下象限（低产量+高特征差异）：深红色
                    color = 'darkred'
                    weight = 'normal'
                    fontsize = 5
                
                txt = ax.annotate(names[i], (x_data[i], yields[i]),
                               fontsize=fontsize, color=color, fontweight=weight,
                               alpha=0.65)
                texts.append(txt)
            
            # 自动调整标签位置避免重叠
            adjust_text(texts, ax=ax, 
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.3, alpha=0.4),
                       expand_points=(1.15, 1.15), expand_text=(1.15, 1.15),
                       force_points=(0.15, 0.25), force_text=(0.15, 0.25))
        else:
            # 降级方案：只标注上半部分，交替左右
            upper_half_indices = [i for i in all_indices if yields[i] > median_y]
            upper_half_indices.sort(key=lambda i: yields[i], reverse=True)
            for idx, i in enumerate(upper_half_indices):
                in_ideal = x_data[i] < median_x
                if in_ideal:
                    color = 'darkgreen'
                    weight = 'bold'
                else:
                    color = 'darkblue'
                    weight = 'normal'
                
                if idx % 2 == 0:
                    xytext = (4, 2)
                else:
                    xytext = (-4, 2)
                
                ax.annotate(names[i], (x_data[i], yields[i]),
                           xytext=xytext, textcoords='offset points',
                           fontsize=6, color=color, fontweight=weight,
                           alpha=0.75, ha='left' if idx % 2 == 0 else 'right')
        
        ax.set_xlabel('Feature Diff', fontsize=10)
        ax.set_ylabel('NoControl Yield (kg/ha)', fontsize=10)
        ax.set_title(f'$t_{{{t+1}}}$: {date_labels[t]}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # 总标题
    fig.suptitle(
        f'Feature Difference vs NoControl Yield - {feature_type.upper()}\n'
        f'Time-series Bubble Analysis (size by NoControl NDM: smaller NDM = larger bubble)',
        fontsize=14,
        fontweight='bold',
        y=0.975
    )
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.06, right=0.98)
    
    # 保存时间点图
    save_path = analyzer.output_dir / f'feature_vs_yield_timeseries_{feature_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Timepoints time-series saved: {save_path}")
    print(f"  - {n_timepoints} timepoint plots")

    # ====================================================================================
    # 创建图1B：成熟度(横轴) vs NoControl产量（时序）
    # 气泡大小: 特征差异（越小越大）
    # ====================================================================================
    fig_maturity = plt.figure(figsize=(16.5, 5.2 * n_rows))
    gs_maturity = GridSpec(n_rows, n_cols, figure=fig_maturity, hspace=0.18, wspace=0.12)

    # NDM固定（基于品种），每个时间点变化的是特征差异->气泡大小
    maturity_x = ndm_values
    median_maturity = np.nanmedian(maturity_x)
    median_yield = np.nanmedian(yields)

    for t in range(n_timepoints):
        row = t // n_cols
        col = t % n_cols
        ax = fig_maturity.add_subplot(gs_maturity[row, col])

        diff_t = np.array([genotype_data[g]['feature_diffs'][t] for g in names], dtype=float)
        bubble_sizes_t = _inverse_bubble_sizes(diff_t, size_min=80.0, size_max=300.0)

        # 颜色使用当期综合得分（低差异 + 高产量 + 早熟）
        composite_score_t = np.zeros_like(diff_t, dtype=float)
        valid_t = np.isfinite(diff_t)
        if valid_t.sum() >= 2 and (np.nanmax(diff_t[valid_t]) - np.nanmin(diff_t[valid_t])) > 1e-10:
            dmin = np.nanmin(diff_t[valid_t])
            dmax = np.nanmax(diff_t[valid_t])
            low_diff_t = 1.0 - (diff_t[valid_t] - dmin) / (dmax - dmin)
            y_arr = np.array(yields, dtype=float)
            y_norm = (y_arr - np.nanmin(y_arr)) / (np.nanmax(y_arr) - np.nanmin(y_arr) + 1e-10)
            composite_score_t[valid_t] = (
                w_feature * low_diff_t
                + w_yield * y_norm[valid_t]
                + w_ndm * ndm_earliness[valid_t]
            )
        else:
            composite_score_t[:] = 0.5

        scatter_m = ax.scatter(
            maturity_x,
            yields,
            s=bubble_sizes_t,
            c=composite_score_t,
            cmap='YlGnBu',
            alpha=0.75,
            edgecolors='black',
            linewidth=0.8,
        )

        ax.axvline(median_maturity, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(median_yield, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

        # 只标注高产半区，减少拥挤
        high_y_idx = [i for i in range(len(names)) if yields[i] > median_yield]
        if ADJUST_TEXT_AVAILABLE and high_y_idx:
            texts = []
            for i in high_y_idx:
                txt = ax.annotate(names[i], (maturity_x[i], yields[i]), fontsize=6, alpha=0.7)
                texts.append(txt)
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.3, alpha=0.4),
                expand_points=(1.12, 1.12),
                expand_text=(1.12, 1.12),
                force_points=(0.12, 0.20),
                force_text=(0.12, 0.20),
            )

        ax.set_xlabel('NoControl Maturity (NDM, smaller = earlier)', fontsize=10)
        ax.set_ylabel('NoControl Yield (kg/ha)', fontsize=10)
        ax.set_title(f'$t_{{{t+1}}}$: {date_labels[t]}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')

    fig_maturity.suptitle(
        f'Maturity (NDM) vs NoControl Yield - {feature_type.upper()}\n'
        f'Time-series Bubble Analysis (bubble size by Feature Difference: smaller diff = larger bubble)',
        fontsize=14,
        fontweight='bold',
        y=0.975,
    )
    fig_maturity.subplots_adjust(top=0.90, bottom=0.07, left=0.06, right=0.98)

    save_path_maturity = analyzer.output_dir / f'maturity_vs_yield_timeseries_{feature_type}.png'
    plt.savefig(save_path_maturity, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Maturity-vs-yield time-series saved: {save_path_maturity}")
    print(f"  - {n_timepoints} timepoint plots")
    
    # ====================================================================================
    # 创建图2：单独的AVERAGE图（更大、更详细）
    # ====================================================================================
    fig_avg = plt.figure(figsize=(12, 10))
    ax_mean = fig_avg.add_subplot(111)
    
    x_data_mean = [genotype_data[g]['mean_diff'] for g in names]
    bubble_sizes_avg = bubble_sizes * 1.35
    
    # 计算每个品种的特征差异标准误差（SEM = SD / sqrt(n)）
    x_sem = [np.std(genotype_data[g]['feature_diffs']) / np.sqrt(len(genotype_data[g]['feature_diffs'])) 
             for g in names]
    
    # 计算中位数用于象限划分
    median_x_mean = np.median(x_data_mean)
    median_y_mean = np.median(yields)
    
    # 计算综合得分：与时间点图一致的加权规则
    x_mean_norm = (np.array(x_data_mean) - np.min(x_data_mean)) / (np.max(x_data_mean) - np.min(x_data_mean) + 1e-10)
    y_mean_norm = (np.array(yields) - np.min(yields)) / (np.max(yields) - np.min(yields) + 1e-10)
    composite_score_mean = (
        w_feature * (1.0 - x_mean_norm)
        + w_yield * y_mean_norm
        + w_ndm * ndm_earliness
    )
    
    # 绘制气泡图（颜色表示综合得分，大小表示早熟程度）
    scatter_mean = ax_mean.scatter(x_data_mean, yields, s=bubble_sizes_avg, alpha=0.7,
                                  c=composite_score_mean, cmap='YlGnBu',
                                  edgecolors='black', linewidth=1.2)
    
    # 为所有散点添加水平误差线（特征差异的标准误差SEM）
    for i, (x, y, sem) in enumerate(zip(x_data_mean, yields, x_sem)):
        ax_mean.errorbar(x, y, xerr=sem, fmt='none', 
                       ecolor='gray', elinewidth=1.2, capsize=2.5, 
                       capthick=1.2, alpha=0.5, zorder=1)
    
    # 添加象限划分线（基于中位数）
    ax_mean.axvline(median_x_mean, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax_mean.axhline(median_y_mean, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    
    # 标注所有品种
    all_indices = list(range(len(names)))
    
    # 使用adjustText优化标签位置（标注所有品种）
    if ADJUST_TEXT_AVAILABLE and len(all_indices) > 0:
        texts = []
        for i in all_indices:
            # 根据所在象限使用不同颜色（四个象限四种颜色）
            left_half = x_data_mean[i] < median_x_mean
            upper_half = yields[i] > median_y_mean
            
            if left_half and upper_half:
                # 左上象限（IDEAL Zone）：深绿色
                color = 'darkgreen'
                weight = 'bold'
                fontsize = 8
            elif not left_half and upper_half:
                # 右上象限（高产量+高特征差异）：深蓝色
                color = 'darkblue'
                weight = 'normal'
                fontsize = 7.5
            elif left_half and not upper_half:
                # 左下象限（低产量+低特征差异）：橙色
                color = 'darkorange'
                weight = 'normal'
                fontsize = 7
            else:
                # 右下象限（低产量+高特征差异）：深红色
                color = 'darkred'
                weight = 'normal'
                fontsize = 7
            
            txt = ax_mean.annotate(names[i], (x_data_mean[i], yields[i]),
                                 fontsize=fontsize, color=color, fontweight=weight,
                                 alpha=0.75)
            texts.append(txt)
        
        # 自动调整标签位置避免重叠
        adjust_text(texts, ax=ax_mean,
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.6, alpha=0.5),
                   expand_points=(1.4, 1.4), expand_text=(1.25, 1.25),
                   force_points=(0.25, 0.35), force_text=(0.25, 0.35))
    else:
        # 降级方案：只标注上半部分
        upper_half_indices = [i for i in all_indices if yields[i] > median_y_mean]
        upper_half_indices.sort(key=lambda i: yields[i], reverse=True)
        for idx, i in enumerate(upper_half_indices):
            in_ideal = x_data_mean[i] < median_x_mean
            if in_ideal:
                color = 'darkgreen'
                weight = 'bold'
            else:
                color = 'darkblue'
                weight = 'normal'
            
            if idx % 2 == 0:
                xytext = (5, 3)
                ha = 'left'
            else:
                xytext = (-5, 3)
                ha = 'right'
            
            ax_mean.annotate(names[i], (x_data_mean[i], yields[i]),
                           xytext=xytext, textcoords='offset points',
                           fontsize=7, color=color, fontweight=weight,
                           alpha=0.8, ha=ha)
    
    ax_mean.set_xlabel('Mean Feature Difference (All Timepoints)', fontsize=13, fontweight='bold')
    ax_mean.set_ylabel('NoControl Yield (kg/ha)', fontsize=13, fontweight='bold')
    ax_mean.set_title('Feature Difference vs Real Yield Performance\nBubble size = NoControl NDM (smaller NDM = larger bubble)',
                     fontsize=14, fontweight='bold', color='darkblue', pad=15)
    ax_mean.grid(True, alpha=0.3, linestyle=':')
    
    # 颜色条
    cbar = plt.colorbar(scatter_mean, ax=ax_mean)
    cbar.set_label(
        'Composite Score\n(0.40*LowDiff + 0.40*HighYield + 0.20*EarlyMaturity)',
        fontsize=11
    )
    
    # 总标题
    fig_avg.suptitle(f'AVERAGE (All Timepoints) - {feature_type.upper()}', 
                    fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存AVERAGE图
    save_path_avg = analyzer.output_dir / f'feature_vs_yield_average_{feature_type}.png'
    plt.savefig(save_path_avg, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Average time-series saved: {save_path_avg}")
    print(f"  - Average of all {n_timepoints} timepoints")

    # ====================================================================================
    # 创建图2B：成熟度(横轴) vs NoControl产量（均值）
    # 气泡大小: 均值特征差异（越小越大）
    # ====================================================================================
    fig_maturity_avg = plt.figure(figsize=(12, 10))
    ax_maturity_avg = fig_maturity_avg.add_subplot(111)

    mean_diffs = np.array(x_data_mean, dtype=float)
    bubble_sizes_mean = _inverse_bubble_sizes(mean_diffs, size_min=120.0, size_max=430.0)

    scatter_m_avg = ax_maturity_avg.scatter(
        ndm_values,
        yields,
        s=bubble_sizes_mean,
        c=composite_score_mean,
        cmap='YlGnBu',
        alpha=0.75,
        edgecolors='black',
        linewidth=1.1,
    )

    median_ndm = np.nanmedian(ndm_values)
    median_y = np.nanmedian(yields)
    ax_maturity_avg.axvline(median_ndm, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax_maturity_avg.axhline(median_y, color='gray', linestyle='--', alpha=0.5, linewidth=2)

    top_y_idx = np.argsort(np.array(yields))[::-1][:12]
    if ADJUST_TEXT_AVAILABLE and len(top_y_idx) > 0:
        texts = []
        for i in top_y_idx:
            texts.append(ax_maturity_avg.annotate(names[i], (ndm_values[i], yields[i]), fontsize=7, alpha=0.75))
        adjust_text(
            texts,
            ax=ax_maturity_avg,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.45),
            expand_points=(1.25, 1.25),
            expand_text=(1.18, 1.18),
            force_points=(0.2, 0.3),
            force_text=(0.2, 0.3),
        )

    ax_maturity_avg.set_xlabel('NoControl Maturity (NDM, smaller = earlier)', fontsize=13, fontweight='bold')
    ax_maturity_avg.set_ylabel('NoControl Yield (kg/ha)', fontsize=13, fontweight='bold')
    ax_maturity_avg.set_title(
        'Maturity vs Real Yield Performance\n'
        'Bubble size = Mean Feature Difference (smaller diff = larger bubble)',
        fontsize=14,
        fontweight='bold',
        color='darkblue',
        pad=15,
    )
    ax_maturity_avg.grid(True, alpha=0.3, linestyle=':')

    cbar_m_avg = plt.colorbar(scatter_m_avg, ax=ax_maturity_avg)
    cbar_m_avg.set_label('Composite Score (0.40*LowDiff + 0.40*HighYield + 0.20*EarlyMaturity)', fontsize=11)

    fig_maturity_avg.suptitle(f'AVERAGE Maturity vs Yield - {feature_type.upper()}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    save_path_maturity_avg = analyzer.output_dir / f'maturity_vs_yield_average_{feature_type}.png'
    plt.savefig(save_path_maturity_avg, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Maturity-vs-yield average saved: {save_path_maturity_avg}")
    
    # 保存数据
    results_df = pd.DataFrame({
        'genotype': names,
        'mean_feature_diff': x_data_mean,
        'nocontrol_yield': yields,
        'nocontrol_ndm': ndm_values,
        'bubble_size': bubble_sizes,
        'bubble_size_mean_diff_inverse': bubble_sizes_mean,
        'ndm_earliness_norm': ndm_earliness,
        'w_feature_low_diff': w_feature,
        'w_yield_high': w_yield,
        'w_ndm_early_maturity': w_ndm
    })
    
    for t in range(n_timepoints):
        results_df[f'feature_diff_t{t+1}'] = [genotype_data[g]['feature_diffs'][t] for g in names]
    
    csv_path = analyzer.output_dir / f'feature_vs_yield_timeseries_data_{feature_type}.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Data saved: {csv_path}")
    
    # 统计摘要
    print(f"\n{'='*80}")
    print("Statistical Summary")
    print(f"{'='*80}")
    
    print(f"\nCorrelation at each time point:")
    for t in range(n_timepoints):
        x_t = [genotype_data[g]['feature_diffs'][t] for g in names]
        corr, p_val = stats.pearsonr(x_t, yields)
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
        ax.set_title(f'$t_{{{t+1}}}$: {date_labels[t]}', fontsize=12, fontweight='bold')
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
    
    # 总标题
    fig.suptitle(f'Feature Difference (Control - NoControl) vs Grain Yield\nFeature Type: {feature_type.upper()}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # 保存图片
    save_path = analyzer.output_dir / f'feature_diff_yield_relationship_{feature_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
