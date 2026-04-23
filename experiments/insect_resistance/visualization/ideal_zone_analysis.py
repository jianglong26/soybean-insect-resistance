"""
IDEAL Zone Analysis
分析并可视化在IDEAL zone（左上角象限：低特征差异+高产量）中的品种
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from pathlib import Path

from ..core.analyzer import MultiModalInsectResistanceAnalyzer

# Improve CJK rendering and overall readability across figures.
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False


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


def analyze_ideal_zone_genotypes(feature_types=['dinov3', 'vi', 'fusion']):
    """
    分析并可视化在IDEAL zone（左上角象限：低特征差异+高产量）中的品种
    对比三种特征类型，找出在所有时间点和均值中都位于IDEAL zone的品种
    
    Args:
        feature_types: 特征类型列表，默认['dinov3', 'vi', 'fusion']
    
    Returns:
        comparison_df: 对比数据框
        all_results: 每种特征类型的详细结果
    """
    print(f"\n{'='*80}")
    print(f"IDEAL Zone Genotypes Analysis - Cross-Feature Comparison")
    print(f"{'='*80}\n")
    
    # 存储每种特征类型的结果
    all_results = {}
    
    for feature_type in feature_types:
        print(f"\nProcessing {feature_type.upper()}...")
        
        try:
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
            else:
                print(f"  ✗ Unknown feature type: {feature_type}")
                continue
            
            # 时间点数量
            first_control = list(control_features.keys())[0]
            n_timepoints = len(control_features[first_control]['features'])
            
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
                
                genotype_data[genotype] = {
                    'feature_diffs': feature_diffs,
                    'nocontrol_yield': nocontrol_yield,
                    'mean_diff': np.mean(feature_diffs)
                }
            
            # 统计每个品种在IDEAL zone出现的次数
            names = list(genotype_data.keys())
            yields = [genotype_data[g]['nocontrol_yield'] for g in names]
            
            ideal_zone_counts = {}
            ideal_zone_details = {}
            
            for genotype in names:
                count = 0
                timepoint_status = []
                
                # 检查每个时间点
                for t in range(n_timepoints):
                    x_data = [genotype_data[g]['feature_diffs'][t] for g in names]
                    median_x = np.median(x_data)
                    median_y = np.median(yields)
                    
                    g_idx = names.index(genotype)
                    if x_data[g_idx] < median_x and yields[g_idx] > median_y:
                        count += 1
                        timepoint_status.append(True)
                    else:
                        timepoint_status.append(False)
                
                # 检查均值
                x_data_mean = [genotype_data[g]['mean_diff'] for g in names]
                median_x_mean = np.median(x_data_mean)
                median_y_mean = np.median(yields)
                
                g_idx = names.index(genotype)
                mean_in_ideal = x_data_mean[g_idx] < median_x_mean and yields[g_idx] > median_y_mean
                if mean_in_ideal:
                    count += 1
                
                ideal_zone_counts[genotype] = count
                ideal_zone_details[genotype] = {
                    'timepoints': timepoint_status,
                    'mean': mean_in_ideal,
                    'total_opportunities': n_timepoints + 1,  # timepoints + mean
                    'count': count,
                    'yield': genotype_data[genotype]['nocontrol_yield'],
                    'mean_diff': genotype_data[genotype]['mean_diff']
                }
            
            all_results[feature_type] = {
                'ideal_zone_counts': ideal_zone_counts,
                'ideal_zone_details': ideal_zone_details,
                'n_timepoints': n_timepoints
            }
            
            print(f"  ✓ {len(ideal_zone_counts)} genotypes analyzed")
            
        except Exception as e:
            print(f"  ✗ Error processing {feature_type}: {str(e)}")
            continue
    
    if len(all_results) == 0:
        print("✗ No results to analyze!")
        return None, None
    
    # 创建对比可视化
    print(f"\n{'='*80}")
    print("Creating comparison visualizations...")
    print(f"{'='*80}\n")
    
    # 获取所有品种
    all_genotypes = set()
    for feature_type in feature_types:
        if feature_type in all_results:
            all_genotypes.update(all_results[feature_type]['ideal_zone_counts'].keys())
    all_genotypes = sorted(list(all_genotypes))
    
    # 创建对比数据框
    comparison_data = []
    for genotype in all_genotypes:
        row = {'genotype': genotype}
        for feature_type in feature_types:
            if feature_type in all_results:
                details = all_results[feature_type]['ideal_zone_details'].get(genotype, {})
                n_total = all_results[feature_type]['n_timepoints'] + 1
                count = details.get('count', 0)
                row[f'{feature_type}_count'] = count
                row[f'{feature_type}_ratio'] = count / n_total if n_total > 0 else 0
                row[f'{feature_type}_yield'] = details.get('yield', 0)
                row[f'{feature_type}_total'] = n_total
        
        # 计算平均出现率
        ratios = [row.get(f'{ft}_ratio', 0) for ft in feature_types if f'{ft}_ratio' in row]
        row['avg_ratio'] = np.mean(ratios) if ratios else 0
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('avg_ratio', ascending=False)
    
    # 保存数据
    module_dir = Path(__file__).parent.parent
    output_dir = module_dir / 'outputs' / 'results' / 'cross_feature_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'ideal_zone_genotypes_comparison.csv'
    comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Comparison data saved: {csv_path}")
    
    # 找出在所有特征中都表现优异的品种（阈值：>=70%的时间在IDEAL zone）
    threshold = 0.7
    consistent_performers = comparison_df[
        (comparison_df['avg_ratio'] >= threshold)
    ].copy()
    
    if len(consistent_performers) > 0:
        print(f"\n{'='*80}")
        print(f"Consistent IDEAL Zone Performers (>={threshold*100}% of time):")
        print(f"{'='*80}")
        for _, row in consistent_performers.iterrows():
            print(f"  {row['genotype']}: {row['avg_ratio']*100:.1f}% avg")
        print(f"{'='*80}\n")
    
    # 创建可视化（2行布局）
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25,
                  height_ratios=[1.2, 1])
    
    # 子图1：热图 - 所有30个品种在不同特征中的IDEAL zone出现次数
    ax1 = fig.add_subplot(gs[0, :])
    
    # 准备热图数据 - 显示所有30个品种
    heatmap_data = []
    heatmap_labels = []
    for _, row in comparison_df.iterrows():  # 所有品种
        counts = [row.get(f'{ft}_count', 0) for ft in feature_types if f'{ft}_count' in row]
        heatmap_data.append(counts)
        heatmap_labels.append(row['genotype'])
    
    heatmap_data = np.array(heatmap_data).T
    
    im = ax1.imshow(heatmap_data, cmap='YlGn', aspect='auto')
    
    ax1.set_xticks(np.arange(len(heatmap_labels)))
    ax1.set_xticklabels(heatmap_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(np.arange(len([ft for ft in feature_types if ft in all_results])))
    ax1.set_yticklabels([ft.upper() for ft in feature_types if ft in all_results], fontsize=12, fontweight='bold')
    ax1.set_title('(A) All 30 Genotypes: IDEAL Zone Frequency Heatmap\n(Number of times in IDEAL zone: Timepoints + Mean)',
                 fontsize=14, fontweight='bold', pad=15)
    
    # 添加数值标注（只标注>0的值）
    for i in range(len([ft for ft in feature_types if ft in all_results])):
        for j in range(len(heatmap_labels)):
            if i < heatmap_data.shape[0] and j < heatmap_data.shape[1]:
                value = heatmap_data[i, j]
                if value > 0:
                    # 获取对应特征的总机会数
                    ft = [f for f in feature_types if f in all_results][i]
                    total = all_results[ft]['n_timepoints'] + 1
                    text = ax1.text(j, i, f'{int(value)}',
                                   ha="center", va="center", color="black", 
                                   fontsize=7, fontweight='bold')
    
    # 添加网格线
    ax1.set_xticks(np.arange(len(heatmap_labels)) - 0.5, minor=True)
    ax1.set_yticks(np.arange(len([ft for ft in feature_types if ft in all_results])) - 0.5, minor=True)
    ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('IDEAL Zone Count', fontsize=11)
    
    # 子图2：条形图 - 所有30个品种的平均IDEAL zone出现率
    ax2 = fig.add_subplot(gs[1, 0])
    
    y_pos = np.arange(len(comparison_df))
    colors = plt.cm.RdYlGn(comparison_df['avg_ratio'] / comparison_df['avg_ratio'].max() if comparison_df['avg_ratio'].max() > 0 else [0.5]*len(comparison_df))
    
    bars = ax2.barh(y_pos, comparison_df['avg_ratio'] * 100, color=colors, 
                   edgecolor='black', linewidth=0.6, alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(comparison_df['genotype'], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel('Average IDEAL Zone Ratio (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) All 30 Genotypes by Average IDEAL Zone Ratio\n(Across all features)', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x', linestyle=':')
    ax2.axvline(threshold*100, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Threshold ({threshold*100}%)')
    ax2.legend(fontsize=10)
    
    # 添加数值标签（只标注前20个）
    for i, (idx, row) in enumerate(comparison_df.head(20).iterrows()):
        ratio = row['avg_ratio'] * 100
        ax2.text(ratio + 1, i, f'{ratio:.1f}%', va='center', fontsize=7, fontweight='bold')
    
    # 子图3：特征类型对比 - 每个特征的IDEAL zone分布
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 为每种特征绘制分布直方图
    x_offset = 0
    bar_width = 0.25
    colors_features = ['#4CAF50', '#FF9800', '#2196F3']
    
    for i, feature_type in enumerate([ft for ft in feature_types if ft in all_results]):
        ratios = []
        for _, row in comparison_df.iterrows():
            if f'{feature_type}_ratio' in row:
                ratios.append(row[f'{feature_type}_ratio'] * 100)
        
        # 创建直方图数据
        hist, bins = np.histogram(ratios, bins=[0, 30, 50, 70, 100])
        x_pos = np.arange(len(hist)) + i * bar_width
        
        ax3.bar(x_pos, hist, width=bar_width, label=feature_type.upper(),
               color=colors_features[i], alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # 添加数值标签
        for j, count in enumerate(hist):
            if count > 0:
                ax3.text(x_pos[j], count + 0.3, str(count),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_xticks(np.arange(4) + bar_width)
    ax3.set_xticklabels(['<30%', '30-50%', '50-70%', '≥70%'], fontsize=10)
    ax3.set_xlabel('IDEAL Zone Ratio Range', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Genotypes', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Distribution Comparison Across Features\n(All 30 Genotypes)',
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # 保存
    save_path = output_dir / 'ideal_zone_cross_feature_comparison.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Visualization saved: {save_path}")
    
    # Cross-feature standalone scatter output is disabled by current configuration.
    print("\n📊 Cross-feature standalone scatter skipped by configuration.")
    
    # 生成详细报告
    report_path = output_dir / 'ideal_zone_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IDEAL ZONE GENOTYPES ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("Analysis Summary:\n")
        f.write(f"  - Feature types analyzed: {', '.join([ft.upper() for ft in feature_types if ft in all_results])}\n")
        f.write(f"  - Total genotypes: {len(all_genotypes)}\n")
        f.write(f"  - IDEAL zone threshold: {threshold*100}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOP PERFORMERS (Average IDEAL Zone Ratio >= 70%)\n")
        f.write("="*80 + "\n\n")
        
        if len(consistent_performers) > 0:
            for _, row in consistent_performers.iterrows():
                f.write(f"{row['genotype']}:\n")
                f.write(f"  Average IDEAL Zone Ratio: {row['avg_ratio']*100:.1f}%\n")
                for ft in feature_types:
                    if f'{ft}_count' in row and ft in all_results:
                        n_total = row.get(f'{ft}_total', 0)
                        f.write(f"  {ft.upper():8s}: {int(row[f'{ft}_count'])}/{n_total} ({row[f'{ft}_ratio']*100:.1f}%)\n")
                f.write("\n")
        else:
            f.write(f"No genotypes meet the {threshold*100}% threshold.\n\n")
        
        f.write("="*80 + "\n")
        f.write("ALL GENOTYPES RANKING\n")
        f.write("="*80 + "\n\n")
        
        for rank, (_, row) in enumerate(comparison_df.iterrows(), 1):
            f.write(f"{rank:2d}. {row['genotype']:15s} | Avg: {row['avg_ratio']*100:5.1f}% | ")
            for ft in feature_types:
                if f'{ft}_ratio' in row and ft in all_results:
                    f.write(f"{ft.upper()}: {row[f'{ft}_ratio']*100:5.1f}% | ")
            f.write("\n")
    
    print(f"✓ Report saved: {report_path}")
    
    print(f"\n{'='*80}")
    print("IDEAL Zone Analysis Complete!")
    print(f"{'='*80}\n")
    
    return comparison_df, all_results


def plot_standalone_scatter_with_all_labels(sorted_names, sorted_ratios, sorted_yields,
                                           sorted_ndm, sorted_composite_scores,
                                           feature_type, output_dir):
    """
    创建独立的散点图，标注所有30个品种（使用adjustText自动调整位置）
    
    Parameters:
    -----------
    sorted_names : array
        品种名称数组（按IDEAL zone频率排序）
    sorted_ratios : array
        IDEAL zone频率数组
    sorted_yields : array
        产量数组
    feature_type : str
        特征类型
    output_dir : str
        输出目录
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ndm_arr = np.asarray(sorted_ndm, dtype=float)
    bubble_sizes = np.full(len(sorted_names), 150.0)
    valid_ndm = np.isfinite(ndm_arr)
    if valid_ndm.sum() >= 2 and (np.nanmax(ndm_arr[valid_ndm]) - np.nanmin(ndm_arr[valid_ndm])) > 1e-12:
        ndm_min = np.nanmin(ndm_arr[valid_ndm])
        ndm_max = np.nanmax(ndm_arr[valid_ndm])
        ndm_norm = (ndm_arr[valid_ndm] - ndm_min) / (ndm_max - ndm_min)
        bubble_sizes[valid_ndm] = 260.0 - ndm_norm * (260.0 - 90.0)

    # 绘制散点图（颜色=综合分，气泡大小=早熟程度）
    scatter = ax.scatter(sorted_ratios * 100, sorted_yields, 
                        s=bubble_sizes, c=sorted_composite_scores, cmap='RdYlGn',
                        edgecolors='black', linewidth=1.5, alpha=0.8,
                        zorder=3)
    
    # 准备所有30个品种的标注（无背景框，避免遮挡气泡）
    texts = []
    for i in range(len(sorted_names)):
        # 根据频率≥70%判断是否为高表现者
        is_high_performer = sorted_ratios[i] >= 0.7
        
        # 高表现者使用更大字体和粗体
        fontsize = 11 if is_high_performer else 9
        fontweight = 'bold' if is_high_performer else 'normal'
        
        txt = ax.text(
            sorted_ratios[i] * 100,
            sorted_yields[i],
            sorted_names[i],
            fontsize=fontsize,
            fontweight=fontweight,
            color='black',
            ha='center',
            va='center',
            zorder=5
        )
        # 白色描边保证文字可读，同时不遮挡气泡
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white', alpha=0.9)])
        texts.append(txt)
    
    # 使用adjustText自动调整标签位置避免重叠
    if ADJUST_TEXT_AVAILABLE:
        adjust_text(
            texts,
            ax=ax,
            expand_points=(1.8, 2.0),
            expand_text=(1.4, 1.6),
            force_points=(0.5, 0.7),
            force_text=(0.7, 0.9),
            only_move={'points': 'y', 'text': 'xy'},
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.45)
        )
    
    # 添加70%阈值线
    ax.axvline(70, color='red', linestyle='--', linewidth=2, alpha=0.7, 
              label='70% Threshold', zorder=2)
    
    # 设置标签和标题
    ax.set_xlabel('IDEAL Zone Frequency (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('NoControl Yield (kg/ha)', fontsize=14, fontweight='bold')
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle=':', zorder=1)
    ax.legend(fontsize=12, loc='upper left')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=40)
    cbar.set_label('Composite Score (Frequency + Yield + Early Maturity)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f'yield_vs_ideal_zone_scatter_{feature_type}.png')
    _save_png_and_pdf(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Standalone scatter plot saved: {output_path}")
    plt.close()


def save_statistical_summary(sorted_names, sorted_ratios, sorted_yields,
                            ideal_zone_status, feature_type, n_timepoints, output_dir):
    """
    将统计摘要保存为独立的文本文件
    
    Parameters:
    -----------
    sorted_names : array
        品种名称数组
    sorted_ratios : array
        IDEAL zone频率数组
    sorted_yields : array
        产量数组
    ideal_zone_status : DataFrame
        IDEAL zone状态表格
    feature_type : str
        特征类型
    n_timepoints : int
        时间点数量
    output_dir : str
        输出目录
    """
    output_path = os.path.join(output_dir, f'ideal_zone_statistical_summary_{feature_type}.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"IDEAL ZONE ANALYSIS - STATISTICAL SUMMARY ({feature_type.upper()})\n")
        f.write("="*80 + "\n\n")
        
        # 基本信息
        f.write("ANALYSIS PARAMETERS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Genotypes Analyzed: {len(sorted_names)}\n")
        f.write(f"Time Points: {n_timepoints} + 1 (Mean) = {n_timepoints + 1} total\n")
        f.write(f"IDEAL Zone Definition: Feature Difference < Median AND Yield > Median\n")
        f.write(f"High Performer Threshold: ≥70% (≥{int(0.7 * (n_timepoints + 1))} out of {n_timepoints + 1})\n\n")
        
        # 分布统计
        f.write("DISTRIBUTION STATISTICS:\n")
        f.write("-"*80 + "\n")
        high_performers = (sorted_ratios >= 0.7).sum()
        medium_performers = ((sorted_ratios >= 0.4) & (sorted_ratios < 0.7)).sum()
        low_performers = (sorted_ratios < 0.4).sum()
        
        f.write(f"High Performers (≥70%):   {high_performers:2d} genotypes ({high_performers/len(sorted_names)*100:5.1f}%)\n")
        f.write(f"Medium Performers (40-70%): {medium_performers:2d} genotypes ({medium_performers/len(sorted_names)*100:5.1f}%)\n")
        f.write(f"Low Performers (<40%):    {low_performers:2d} genotypes ({low_performers/len(sorted_names)*100:5.1f}%)\n\n")
        
        # Top 10品种详细信息
        f.write("TOP 10 GENOTYPES (by IDEAL Zone Frequency):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6}{'Genotype':<20}{'Frequency':<15}{'Yield (kg/ha)':<15}{'Status'}\n")
        f.write("-"*80 + "\n")
        
        for i in range(min(10, len(sorted_names))):
            frequency_str = f"{sorted_ratios[i]*100:.1f}%"
            yield_str = f"{sorted_yields[i]:.1f}"
            status = "⭐ High" if sorted_ratios[i] >= 0.7 else "Medium"
            f.write(f"{i+1:<6}{sorted_names[i]:<20}{frequency_str:<15}{yield_str:<15}{status}\n")
        
        f.write("\n")
        
        # Bottom 5品种
        f.write("BOTTOM 5 GENOTYPES:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6}{'Genotype':<20}{'Frequency':<15}{'Yield (kg/ha)'}\n")
        f.write("-"*80 + "\n")
        
        for i in range(max(0, len(sorted_names)-5), len(sorted_names)):
            rank = i + 1
            frequency_str = f"{sorted_ratios[i]*100:.1f}%"
            yield_str = f"{sorted_yields[i]:.1f}"
            f.write(f"{rank:<6}{sorted_names[i]:<20}{frequency_str:<15}{yield_str}\n")
        
        f.write("\n")
        
        # 产量统计
        f.write("YIELD STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Yield:   {sorted_yields.mean():.2f} kg/ha\n")
        f.write(f"Median Yield: {np.median(sorted_yields):.2f} kg/ha\n")
        f.write(f"Max Yield:    {sorted_yields.max():.2f} kg/ha ({sorted_names[sorted_yields.argmax()]})\n")
        f.write(f"Min Yield:    {sorted_yields.min():.2f} kg/ha ({sorted_names[sorted_yields.argmin()]})\n")
        f.write(f"Std Dev:      {sorted_yields.std():.2f} kg/ha\n\n")
        
        # IDEAL Zone频率统计
        f.write("IDEAL ZONE FREQUENCY STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Frequency:   {sorted_ratios.mean()*100:.2f}%\n")
        f.write(f"Median Frequency: {np.median(sorted_ratios)*100:.2f}%\n")
        f.write(f"Max Frequency:    {sorted_ratios.max()*100:.2f}% ({sorted_names[0]})\n")
        f.write(f"Min Frequency:    {sorted_ratios.min()*100:.2f}% ({sorted_names[-1]})\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Statistical summary saved: {output_path}")


def plot_cross_feature_scatter_with_all_labels(comparison_df, feature_types, output_dir, threshold=0.7):
    """
    创建跨特征对比的独立散点图，标注所有30个品种（每种特征）
    
    Parameters:
    -----------
    comparison_df : DataFrame
        跨特征对比数据
    feature_types : list
        特征类型列表
    output_dir : Path
        输出目录
    threshold : float
        阈值（默认0.7）
    """
    colors_features = {'dinov3': '#FF6B6B', 'vi': '#4ECDC4', 'fusion': '#95E1D3'}
    
    fig, axes = plt.subplots(1, len(feature_types), figsize=(18, 8), sharey=True)
    if len(feature_types) == 1:
        axes = [axes]
    
    for idx, feature_type in enumerate(feature_types):
        # 检查是否有该特征类型的数据
        if f'{feature_type}_ratio' not in comparison_df.columns:
            continue
            
        ax = axes[idx]
        
        # 提取数据
        ratios = []
        yields = []
        genotypes = []
        
        for _, row in comparison_df.iterrows():
            if f'{feature_type}_ratio' in row and f'{feature_type}_yield' in row:
                if pd.notna(row[f'{feature_type}_ratio']) and pd.notna(row[f'{feature_type}_yield']):
                    ratios.append(row[f'{feature_type}_ratio'] * 100)
                    yields.append(row[f'{feature_type}_yield'])
                    genotypes.append(row['genotype'])
        
        # 转换为numpy数组
        ratios = np.array(ratios)
        yields = np.array(yields)
        
        # 绘制散点图
        scatter = ax.scatter(ratios, yields, s=150, alpha=0.7,
                           color=colors_features.get(feature_type, '#999999'),
                           edgecolors='black', linewidth=1.5, zorder=3)
        
        # 标注所有30个品种
        texts = []
        for i in range(len(genotypes)):
            # Top 10品种使用更大字体
            fontsize = 10 if i < 10 or ratios[i] >= threshold * 100 else 8
            fontweight = 'bold' if ratios[i] >= threshold * 100 else 'normal'
            
            texts.append(ax.text(ratios[i], yields[i], genotypes[i],
                               fontsize=fontsize, fontweight=fontweight,
                               ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='yellow' if ratios[i] >= threshold * 100 else 'lightblue',
                                       alpha=0.6, edgecolor='black', linewidth=0.5)))
        
        # 使用adjustText自动调整标签位置
        if ADJUST_TEXT_AVAILABLE:
            adjust_text(texts, ax=ax,
                       expand_points=(1.5, 1.5),
                       expand_text=(1.3, 1.3),
                       force_points=(0.3, 0.4),
                       force_text=(0.3, 0.4),
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))
        
        # 添加70%阈值线
        ax.axvline(threshold * 100, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'{threshold*100:.0f}% Threshold', zorder=2)
        
        # 设置标签和标题
        ax.set_xlabel('IDEAL Zone Frequency (%)', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('NoControl Yield (kg/ha)', fontsize=12, fontweight='bold')
        panel_tag = chr(65 + idx)
        ax.set_title(f'({panel_tag}) {feature_type.upper()} Features\nAll 30 Genotypes Labeled',
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle=':', zorder=1)
        ax.legend(fontsize=10, loc='upper left')
        
        # 添加统计信息
        high_count = (ratios >= threshold * 100).sum()
        stats_text = f'High Performers: {high_count}/30'
        ax.text(0.98, 0.02, stats_text,
               transform=ax.transAxes, fontsize=9,
               ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # 保存图像
    output_path = output_dir / 'yield_vs_ideal_zone_scatter_cross_feature.png'
    _save_png_and_pdf(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Cross-feature standalone scatter plot saved: {output_path}")
    plt.close()


def save_statistical_summary(sorted_names, sorted_ratios, sorted_yields,
                            ideal_zone_status, feature_type, n_timepoints, output_dir):
    """
    将统计摘要保存为独立的文本文件
    
    Parameters:
    -----------
    sorted_names : array
        品种名称数组
    sorted_ratios : array
        IDEAL zone频率数组
    sorted_yields : array
        产量数组
    ideal_zone_status : DataFrame
        IDEAL zone状态表格
    feature_type : str
        特征类型
    n_timepoints : int
        时间点数量
    output_dir : str
        输出目录
    """
    output_path = os.path.join(output_dir, f'ideal_zone_statistical_summary_{feature_type}.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"IDEAL ZONE ANALYSIS - STATISTICAL SUMMARY ({feature_type.upper()})\n")
        f.write("="*80 + "\n\n")
        
        # 基本信息
        f.write("ANALYSIS PARAMETERS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Genotypes Analyzed: {len(sorted_names)}\n")
        f.write(f"Time Points: {n_timepoints} + 1 (Mean) = {n_timepoints + 1} total\n")
        f.write(f"IDEAL Zone Definition: Feature Difference < Median AND Yield > Median\n")
        f.write(f"High Performer Threshold: ≥70% (≥{int(0.7 * (n_timepoints + 1))} out of {n_timepoints + 1})\n\n")
        
        # 分布统计
        f.write("DISTRIBUTION STATISTICS:\n")
        f.write("-"*80 + "\n")
        high_performers = (sorted_ratios >= 0.7).sum()
        medium_performers = ((sorted_ratios >= 0.4) & (sorted_ratios < 0.7)).sum()
        low_performers = (sorted_ratios < 0.4).sum()
        
        f.write(f"High Performers (≥70%):   {high_performers:2d} genotypes ({high_performers/len(sorted_names)*100:5.1f}%)\n")
        f.write(f"Medium Performers (40-70%): {medium_performers:2d} genotypes ({medium_performers/len(sorted_names)*100:5.1f}%)\n")
        f.write(f"Low Performers (<40%):    {low_performers:2d} genotypes ({low_performers/len(sorted_names)*100:5.1f}%)\n\n")
        
        # Top 10品种详细信息
        f.write("TOP 10 GENOTYPES (by IDEAL Zone Frequency):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6}{'Genotype':<20}{'Frequency':<15}{'Yield (kg/ha)':<15}{'Status'}\n")
        f.write("-"*80 + "\n")
        
        for i in range(min(10, len(sorted_names))):
            frequency_str = f"{sorted_ratios[i]*100:.1f}%"
            yield_str = f"{sorted_yields[i]:.1f}"
            status = "⭐ High" if sorted_ratios[i] >= 0.7 else "Medium"
            f.write(f"{i+1:<6}{sorted_names[i]:<20}{frequency_str:<15}{yield_str:<15}{status}\n")
        
        f.write("\n")
        
        # Bottom 5品种
        f.write("BOTTOM 5 GENOTYPES:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6}{'Genotype':<20}{'Frequency':<15}{'Yield (kg/ha)'}\n")
        f.write("-"*80 + "\n")
        
        for i in range(max(0, len(sorted_names)-5), len(sorted_names)):
            rank = i + 1
            frequency_str = f"{sorted_ratios[i]*100:.1f}%"
            yield_str = f"{sorted_yields[i]:.1f}"
            f.write(f"{rank:<6}{sorted_names[i]:<20}{frequency_str:<15}{yield_str}\n")
        
        f.write("\n")
        
        # 产量统计
        f.write("YIELD STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Yield:   {sorted_yields.mean():.2f} kg/ha\n")
        f.write(f"Median Yield: {np.median(sorted_yields):.2f} kg/ha\n")
        f.write(f"Max Yield:    {sorted_yields.max():.2f} kg/ha ({sorted_names[sorted_yields.argmax()]})\n")
        f.write(f"Min Yield:    {sorted_yields.min():.2f} kg/ha ({sorted_names[sorted_yields.argmin()]})\n")
        f.write(f"Std Dev:      {sorted_yields.std():.2f} kg/ha\n\n")
        
        # IDEAL Zone频率统计
        f.write("IDEAL ZONE FREQUENCY STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Frequency:   {sorted_ratios.mean()*100:.2f}%\n")
        f.write(f"Median Frequency: {np.median(sorted_ratios)*100:.2f}%\n")
        f.write(f"Max Frequency:    {sorted_ratios.max()*100:.2f}% ({sorted_names[0]})\n")
        f.write(f"Min Frequency:    {sorted_ratios.min()*100:.2f}% ({sorted_names[-1]})\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Statistical summary saved: {output_path}")


def visualize_single_feature_ideal_zone(feature_type='dinov3'):
    """
    为单个特征类型生成详细的IDEAL zone分析
    显示所有30个品种在每个时间点+均值中的IDEAL zone状态
    
    Args:
        feature_type: 'dinov3', 'vi', 'fusion'
    """
    print(f"\n{'='*80}")
    print(f"Detailed IDEAL Zone Analysis - {feature_type.upper()}")
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
    else:
        print(f"✗ Unknown feature type: {feature_type}")
        return
    
    # 时间点数量
    first_control = list(control_features.keys())[0]
    n_timepoints = len(control_features[first_control]['features'])
    
    # 获取日期标签
    import json
    metadata_path = analyzer.data_dir / 'dataset_metadata.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    first_nocontrol = list(nocontrol_features.keys())[0]
    dates = [img['date'] for img in metadata[first_nocontrol]['image_sequence']]
    date_labels = [d[5:] for d in dates]  # '01-28' format
    
    # 收集数据
    genotypes = sorted(analyzer.nocontrol_df['genotype'].unique())
    genotype_data = {}

    def _safe_minmax(values, inverse=False, fill_value=0.5):
        arr = np.asarray(values, dtype=float)
        out = np.full(arr.shape, fill_value, dtype=float)
        valid = np.isfinite(arr)
        if valid.sum() >= 2:
            vmin = np.nanmin(arr[valid])
            vmax = np.nanmax(arr[valid])
            if vmax - vmin > 1e-12:
                scaled = (arr[valid] - vmin) / (vmax - vmin)
                out[valid] = 1.0 - scaled if inverse else scaled
        return out
    
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
            'mean_diff': np.mean(feature_diffs),
            'nocontrol_ndm': nocontrol_ndm
        }
    
    # 统计每个品种在IDEAL zone的状态
    names = list(genotype_data.keys())
    yields = [genotype_data[g]['nocontrol_yield'] for g in names]
    
    # 存储每个品种在各个时间点+均值的状态
    ideal_zone_matrix = np.zeros((len(names), n_timepoints + 1), dtype=int)  # +1 for mean
    
    # 检查每个时间点
    for t in range(n_timepoints):
        x_data = [genotype_data[g]['feature_diffs'][t] for g in names]
        median_x = np.median(x_data)
        median_y = np.median(yields)
        
        for i, genotype in enumerate(names):
            if x_data[i] < median_x and yields[i] > median_y:
                ideal_zone_matrix[i, t] = 1
    
    # 检查均值
    x_data_mean = [genotype_data[g]['mean_diff'] for g in names]
    median_x_mean = np.median(x_data_mean)
    median_y_mean = np.median(yields)
    
    for i, genotype in enumerate(names):
        if x_data_mean[i] < median_x_mean and yields[i] > median_y_mean:
            ideal_zone_matrix[i, n_timepoints] = 1
    
    # 计算总次数和频率
    ideal_counts = ideal_zone_matrix.sum(axis=1)
    ideal_ratios = ideal_counts / (n_timepoints + 1)

    ndm_values = np.array([genotype_data[g]['nocontrol_ndm'] for g in names], dtype=float)
    ndm_earliness = _safe_minmax(ndm_values, inverse=True)

    ratio_norm = ideal_ratios
    yield_norm = _safe_minmax(np.array(yields, dtype=float), inverse=False)
    w_ratio = 0.50
    w_yield = 0.30
    w_maturity = 0.20
    composite_scores = w_ratio * ratio_norm + w_yield * yield_norm + w_maturity * ndm_earliness
    
    # 排序规则（用于排名展示）：按IDEAL出现频率从高到低。
    # 同频时依次按出现次数、产量、早熟性和综合分打破并列。
    sorted_indices = sorted(
        range(len(names)),
        key=lambda i: (
            ideal_ratios[i],
            ideal_counts[i],
            yields[i],
            ndm_earliness[i],
            composite_scores[i],
        ),
        reverse=True,
    )
    sorted_names = [names[i] for i in sorted_indices]
    sorted_matrix = ideal_zone_matrix[sorted_indices]
    sorted_counts = ideal_counts[sorted_indices]
    sorted_ratios = ideal_ratios[sorted_indices]
    sorted_yields = [yields[i] for i in sorted_indices]
    sorted_ndm = [ndm_values[i] for i in sorted_indices]
    sorted_composite_scores = [composite_scores[i] for i in sorted_indices]

    # Scheme-1 for publication: plot only genotypes with IDEAL count > 0.
    nonzero_plot_idx = [i for i, c in enumerate(sorted_counts) if c > 0]
    zero_count_idx = [i for i, c in enumerate(sorted_counts) if c == 0]

    sorted_names_plot = [sorted_names[i] for i in nonzero_plot_idx]
    sorted_counts_plot = sorted_counts[nonzero_plot_idx]
    sorted_ratios_plot = sorted_ratios[nonzero_plot_idx]
    sorted_yields_plot = [sorted_yields[i] for i in nonzero_plot_idx]
    sorted_ndm_plot = [sorted_ndm[i] for i in nonzero_plot_idx]
    sorted_composite_scores_plot = [sorted_composite_scores[i] for i in nonzero_plot_idx]
    
    # 保存详细数据
    module_dir = Path(__file__).parent.parent
    output_dir = module_dir / 'outputs' / 'results' / feature_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV数据
    detailed_data = []
    for i, genotype in enumerate(sorted_names):
        row = {
            'Rank': i + 1,
            'Genotype': genotype,
            'IDEAL_Count': int(sorted_counts[i]),
            'IDEAL_Ratio': f'{sorted_ratios[i]*100:.1f}%',
            'NoControl_Yield': f'{sorted_yields[i]:.1f}',
            'NoControl_NDM': f'{sorted_ndm[i]:.1f}' if np.isfinite(sorted_ndm[i]) else 'NA',
            'Composite_Score': f'{sorted_composite_scores[i]*100:.1f}'
        }
        # 添加每个时间点的状态
        for t in range(n_timepoints):
            row[f'Time{t+1}_{date_labels[t]}'] = 'Yes' if sorted_matrix[i, t] == 1 else 'No'
        row['Mean'] = 'Yes' if sorted_matrix[i, n_timepoints] == 1 else 'No'
        detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    csv_path = output_dir / f'ideal_zone_detailed_{feature_type}.csv'
    detailed_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Detailed data saved: {csv_path}")
    
    # 创建2子图可视化（仅保留底部两个条形图）
    fig = plt.figure(figsize=(20, 10.5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.2, wspace=0.25)

    # ========== 子图1：频率条形图（所有30个品种）==========
    ax2 = fig.add_subplot(gs[0, 0])
    
    y_pos = np.arange(len(sorted_names_plot))
    colors = plt.cm.RdYlGn(sorted_ratios_plot if len(sorted_ratios_plot) > 0 else np.array([0.5]))
    
    bars = ax2.barh(y_pos, sorted_ratios_plot * 100, color=colors, 
                   edgecolor='black', linewidth=0.6, alpha=0.9)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names_plot, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('IDEAL Zone Frequency (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Genotype', fontsize=13, fontweight='bold')
    ax2.text(0.5, -0.13, '(A)', transform=ax2.transAxes,
             ha='center', va='top', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x', linestyle=':')
    ax2.tick_params(axis='x', labelsize=12)
    
    # 添加阈值线
    ax2.axvline(70, color='red', linestyle='--', linewidth=2, alpha=0.6, label='70% Threshold')
    ax2.legend(fontsize=9, loc='lower right')
    
    # 标注数值（只标注前15个；保持文字在边框内）
    x2_left, x2_right = ax2.get_xlim()
    for i in range(min(15, len(sorted_names_plot))):
        ratio = sorted_ratios_plot[i] * 100
        count = sorted_counts_plot[i]
        label = f'{ratio:.1f}% ({int(count)}/{n_timepoints+1})'
        # If near right edge, place label inside the bar with right alignment.
        if ratio >= x2_right - 9.0:
            x_pos = max(x2_left + 1.0, ratio - 0.8)
            ha = 'right'
        else:
            x_pos = ratio + 1.0
            ha = 'left'
        ax2.text(x_pos, i, label, va='center', ha=ha, fontsize=10, fontweight='bold', clip_on=True)
    
    # ========== 子图2：次数统计条形图 ==========
    ax3 = fig.add_subplot(gs[0, 1])
    
    bars3 = ax3.barh(y_pos, sorted_counts_plot, color=colors,
                    edgecolor='black', linewidth=0.6, alpha=0.9)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(sorted_names_plot, fontsize=10)
    ax3.invert_yaxis()
    ax3.set_xlabel(f'IDEAL Zone Count (out of {n_timepoints+1})', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Genotype', fontsize=13, fontweight='bold')
    ax3.text(0.5, -0.13, '(B)', transform=ax3.transAxes,
             ha='center', va='top', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x', linestyle=':')
    ax3.set_xlim(0, n_timepoints + 1)
    ax3.tick_params(axis='x', labelsize=12)
    
    # 标注数值（只标注前15个；保持文字在边框内）
    x3_left, x3_right = ax3.get_xlim()
    for i in range(min(15, len(sorted_names_plot))):
        count = sorted_counts_plot[i]
        if count >= x3_right - 0.35:
            x_pos = max(x3_left + 0.1, count - 0.08)
            ha = 'right'
        else:
            x_pos = count + 0.1
            ha = 'left'
        ax3.text(x_pos, i, f'{int(count)}', va='center', ha=ha, fontsize=10, fontweight='bold', clip_on=True)

    # Explain omitted zero-count genotypes to keep the plot compact for papers.
    ax2.text(
        0.01,
        0.01,
        f'Omitted zero-count genotypes: {len(zero_count_idx)} / {len(sorted_names)}',
        transform=ax2.transAxes,
        fontsize=9,
        color='#444444',
        ha='left',
        va='bottom'
    )
    
    # 保存
    save_path = output_dir / f'ideal_zone_comprehensive_{feature_type}.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Comprehensive visualization saved: {save_path}")
    print(f"  - Frequency ranking (desc): non-zero IDEAL genotypes only ({len(sorted_names_plot)} shown)")
    print(f"  - Absolute count view (same order): non-zero IDEAL genotypes only ({len(sorted_names_plot)} shown)")
    print(f"  - Omitted zero-count genotypes: {len(zero_count_idx)}")
    print(f"  - Statistical summary included")
    
    # 生成独立的散点图（标注所有30个品种）
    print(f"\n📊 Generating standalone scatter plot with all 30 genotypes labeled...")
    plot_standalone_scatter_with_all_labels(
        sorted_names=sorted_names_plot if len(sorted_names_plot) > 0 else sorted_names,
        sorted_ratios=sorted_ratios_plot if len(sorted_names_plot) > 0 else sorted_ratios,
        sorted_yields=sorted_yields_plot if len(sorted_names_plot) > 0 else sorted_yields,
        sorted_ndm=sorted_ndm_plot if len(sorted_names_plot) > 0 else sorted_ndm,
        sorted_composite_scores=sorted_composite_scores_plot if len(sorted_names_plot) > 0 else sorted_composite_scores,
        feature_type=feature_type,
        output_dir=str(output_dir)
    )
    
    # 保存统计摘要为独立文本文件
    print(f"📄 Saving statistical summary as separate text file...")
    save_statistical_summary(
        sorted_names=sorted_names,
        sorted_ratios=sorted_ratios,
        sorted_yields=np.array(sorted_yields),
        ideal_zone_status=detailed_df,
        feature_type=feature_type,
        n_timepoints=n_timepoints,
        output_dir=str(output_dir)
    )
    
    print(f"\n{'='*80}")
    print(f"✅ {feature_type.upper()} Analysis Complete!")
    print(f"{'='*80}\n")
    
    return detailed_df
