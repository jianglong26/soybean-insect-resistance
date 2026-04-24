"""
Quadrant Analysis Plots
四象限分析可视化
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

try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False

from .metrics import infer_gain_stabilizer, compute_gain_rate

# Improve CJK rendering and overall readability across figures.
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12


def _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', pad_inches=0.06, facecolor=None):
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

def plot_control_bug_distribution(metadata, output_dir):
    """绘制Control和NoControl环境的虫害分布对比（包含两个环境的Top 15）"""
    
    # 提取Control和NoControl环境的虫害数据
    control_bugs = {}
    nocontrol_bugs = {}
    
    for sample_key, sample_data in metadata.items():
        bug_count = sample_data['labels'].get('Bug')
        if bug_count is not None:
            genotype = sample_data['genotype']
            
            if sample_data['environment'] == 'control':
                if genotype not in control_bugs:
                    control_bugs[genotype] = []
                control_bugs[genotype].append(bug_count)
            else:  # nocontrol
                if genotype not in nocontrol_bugs:
                    nocontrol_bugs[genotype] = []
                nocontrol_bugs[genotype].append(bug_count)
    
    # 计算每个品种的平均虫害
    control_avg = {g: np.mean(bugs) for g, bugs in control_bugs.items()}
    nocontrol_avg = {g: np.mean(bugs) for g, bugs in nocontrol_bugs.items()}
    
    # 排序
    control_sorted = sorted(control_avg.items(), key=lambda x: x[1])
    nocontrol_sorted = sorted(nocontrol_avg.items(), key=lambda x: x[1])
    
    # 绘图：1行3列
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    # 左图：箱线图对比Control vs NoControl
    all_control_bugs = []
    all_nocontrol_bugs = []
    
    for sample_key, sample_data in metadata.items():
        bug_count = sample_data['labels'].get('Bug')
        if bug_count is not None:
            if sample_data['environment'] == 'control':
                all_control_bugs.append(bug_count)
            else:
                all_nocontrol_bugs.append(bug_count)
    
    bp = ax1.boxplot([all_control_bugs, all_nocontrol_bugs], 
                      tick_labels=['Control\n(Pesticide)', 'NoControl\n(Pest Pressure)'],
                      patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#90EE90', '#FFB6C1']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Bug Count', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Bug Distribution Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    t_stat, p_value = stats.ttest_ind(all_nocontrol_bugs, all_control_bugs)
    ax1.text(0.5, 0.95, f't-test: p={p_value:.4f}', transform=ax1.transAxes,
             ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 中图：Control环境Top 15品种
    if len(control_sorted) > 0:
        top_15_control = control_sorted[:15] if len(control_sorted) > 15 else control_sorted
        genotypes_c = [x[0] for x in top_15_control]
        bugs_c = [x[1] for x in top_15_control]
        
        bars2 = ax2.barh(range(len(genotypes_c)), bugs_c, color='#90EE90', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(range(len(genotypes_c)))
        ax2.set_yticklabels(genotypes_c, fontsize=10)
        ax2.invert_yaxis()
        ax2.set_xlabel('Average Bug Count', fontsize=12, fontweight='bold')
        ax2.set_title('(B) Top 15 Genotypes - Control', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, bug) in enumerate(zip(bars2, bugs_c)):
            ax2.text(bug + 0.2, i, f'{bug:.1f}', va='center', fontsize=9)
    
    # 右图：NoControl环境Top 15品种
    if len(nocontrol_sorted) > 0:
        top_15_nocontrol = nocontrol_sorted[:15] if len(nocontrol_sorted) > 15 else nocontrol_sorted
        genotypes_nc = [x[0] for x in top_15_nocontrol]
        bugs_nc = [x[1] for x in top_15_nocontrol]
        
        bars3 = ax3.barh(range(len(genotypes_nc)), bugs_nc, color='#FFB6C1', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_yticks(range(len(genotypes_nc)))
        ax3.set_yticklabels(genotypes_nc, fontsize=10)
        ax3.invert_yaxis()
        ax3.set_xlabel('Average Bug Count', fontsize=12, fontweight='bold')
        ax3.set_title('(C) Top 15 Genotypes - NoControl', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, bug) in enumerate(zip(bars3, bugs_nc)):
            ax3.text(bug + 0.3, i, f'{bug:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = output_dir / 'control_bug_distribution.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Control对比图已保存")
    plt.close()


def create_two_rankings_all30(resistance_df, output_dir, feature_type='dinov3'):
    """创建两种排名方式的可视化 - 显示所有30个品种
    
    排名1: score_without_bug - 不含虫子指标的基础排名
    排名2: score_with_bug - 含虫子指标的完整排名（实测+预测）
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    
    # ========== 排名1: 不含虫子指标的基础排名（所有30个品种） ==========
    ax1 = axes[0]
    
    # 按score_without_bug排序
    ranking1 = resistance_df.sort_values('score_without_bug', ascending=False).copy()
    
    y_pos = np.arange(len(ranking1))
    # 使用渐变色：从深绿（高分）到浅绿（低分）
    colors1 = plt.cm.Greens(np.linspace(0.9, 0.3, len(ranking1)))
    
    bars1 = ax1.barh(y_pos, ranking1['score_without_bug'], color=colors1, 
                     edgecolor='black', linewidth=1.2, alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ranking1['genotype'], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Score (Without Bug)', fontsize=12, fontweight='bold')
    ax1.set_title(f'(A) Ranking Without Bug Indicator\\n{feature_type.upper()} - All 30 Genotypes', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, 110)
    
    # 标注分数
    for i, (idx, row) in enumerate(ranking1.iterrows()):
        score = row['score_without_bug']
        ax1.text(score + 1.5, i, f'{score:.1f}', va='center', fontsize=8, fontweight='bold')
    
    # ========== 排名2: 含虫子指标的完整排名（所有30个品种，实测+预测） ==========
    ax2 = axes[1]
    
    # 按score_with_bug排序
    ranking2 = resistance_df.sort_values('score_with_bug', ascending=False).copy()
    
    y_pos2 = np.arange(len(ranking2))
    # 根据bug_source设置颜色：实测=深橙色，预测=浅蓝色
    colors2 = ['#FF8C00' if source == 'actual' else '#4169E1' 
               for source in ranking2['bug_source']]
    
    bars2 = ax2.barh(y_pos2, ranking2['score_with_bug'], color=colors2, 
                     edgecolor='black', linewidth=1.2, alpha=0.9)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(ranking2['genotype'], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Score (With Bug)', fontsize=12, fontweight='bold')
    ax2.set_title(f'(B) Ranking With Bug Indicator\\n{feature_type.upper()} - All 30 Genotypes (Actual + Predicted)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 110)
    
    # 标注分数和虫子数据来源
    for i, (idx, row) in enumerate(ranking2.iterrows()):
        score = row['score_with_bug']
        source = row['bug_source']
        marker = '✓' if source == 'actual' else 'P'  # ✓=实测, P=预测
        ax2.text(score + 1.5, i, f'{score:.1f} [{marker}]', 
                va='center', fontsize=8, fontweight='bold')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF8C00', edgecolor='black', label='Actual Bug (✓)'),
        Patch(facecolor='#4169E1', edgecolor='black', label='Predicted Bug (P)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)
    
    
    plt.tight_layout()
    save_path = output_dir / f'two_rankings_all30_{feature_type}.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ 两种排名图已保存（所有30个品种）")
    plt.close()


def visualize_resistance_ranking(resistance_df, output_dir):
    """可视化抗性排名 - 生成2_resistance_ranking.png"""
    print(f"    ✓ 生成抗性排名可视化...")
    
    df = resistance_df.copy()
    
    # 只显示有bug数据的品种
    df_measured = df[df['nocontrol_bug_mean'].notna()].sort_values(
        'resistance_score', ascending=False
    )
    
    if len(df_measured) == 0:
        print("    ⚠ 没有足够的Bug测量数据进行排名")
        return
    
    # 创建综合排名图 (2x2布局)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 综合评分排名
    top_n = min(15, len(df_measured))
    df_top = df_measured.head(top_n)
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_top)))
    y_pos = np.arange(len(df_top))
    
    axes[0, 0].barh(y_pos, df_top['resistance_score'], color=colors)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(df_top['genotype'], fontsize=10)
    axes[0, 0].set_xlabel('Resistance Score', fontsize=12)
    axes[0, 0].set_title('Top 15 Insect-Resistant Genotypes Ranking', fontsize=14, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. 虫害数量 vs 产量损失
    axes[0, 1].scatter(df_measured['nocontrol_bug_mean'], 
                      df_measured['yield_loss_rate'],
                      s=100, alpha=0.6, c=df_measured['resistance_score'],
                      cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # 标注最好的5个品种
    for _, row in df_measured.head(5).iterrows():
        axes[0, 1].annotate(row['genotype'], 
                           (row['nocontrol_bug_mean'], row['yield_loss_rate']),
                           fontsize=8, ha='right')
    
    axes[0, 1].set_xlabel('Bug Count (NoControl)', fontsize=12)
    axes[0, 1].set_ylabel('Yield Loss Rate (%)', fontsize=12)
    axes[0, 1].set_title('Bug Count vs Yield Loss', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Resistance Score', fontsize=10)
    
    # 3. Control vs NoControl 产量对比
    axes[1, 0].scatter(df_measured['control_yield'], 
                      df_measured['nocontrol_yield'],
                      s=100, alpha=0.6, c=df_measured['resistance_score'],
                      cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # 添加对角线 (无损失线)
    max_yield = max(df_measured['control_yield'].max(), 
                   df_measured['nocontrol_yield'].max())
    axes[1, 0].plot([0, max_yield], [0, max_yield], 
                   'k--', alpha=0.5, label='No Yield Loss')
    
    axes[1, 0].set_xlabel('Control Yield (kg/ha)', fontsize=12)
    axes[1, 0].set_ylabel('NoControl Yield (kg/ha)', fontsize=12)
    axes[1, 0].set_title('Yield Comparison: Control vs NoControl', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 多指标雷达图 (top 5)
    from math import pi
    
    categories = ['Low Bug', 'High Yield', 'Tolerance', 'Agro. Value', 'Leaf Retention']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(2, 2, 4, projection='polar')
    
    for i, (_, row) in enumerate(df_measured.head(5).iterrows()):
        # 标准化各指标到0-1
        values = [
            1 - (row['nocontrol_bug_mean'] - df_measured['nocontrol_bug_mean'].min()) / 
                (df_measured['nocontrol_bug_mean'].max() - df_measured['nocontrol_bug_mean'].min()),
            (row['nocontrol_yield'] - df_measured['nocontrol_yield'].min()) /
                (df_measured['nocontrol_yield'].max() - df_measured['nocontrol_yield'].min()),
            row['tolerance_index'] / 100,
            (row['nocontrol_agronomic_value'] - df_measured['nocontrol_agronomic_value'].min()) /
                (df_measured['nocontrol_agronomic_value'].max() - df_measured['nocontrol_agronomic_value'].min()),
            (row['nocontrol_leaf_retention'] - df_measured['nocontrol_leaf_retention'].min()) /
                (df_measured['nocontrol_leaf_retention'].max() - df_measured['nocontrol_leaf_retention'].min())
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['genotype'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Genotypes Multi-Trait Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)
    
    plt.tight_layout()
    save_path = output_dir / '2_resistance_ranking.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ 抗性排名图已保存")
    plt.close()


def visualize_bug_predictions(predictions_df, resistance_df, model, output_dir, feature_type):
    """可视化虫害预测结果 - 生成3_bug_prediction.png"""
    print(f"    ✓ 生成虫害预测可视化...")
    
    # 按品种聚合预测结果
    genotype_predictions = predictions_df.groupby('genotype').agg({
        'predicted_bug': 'mean',
        'true_bug': lambda x: x.dropna().mean() if x.notna().any() else np.nan,
        'has_label': 'any'
    }).reset_index()
    genotype_predictions = genotype_predictions.sort_values('predicted_bug')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 预测 vs 实际 (有标注的样本)
    labeled_data = predictions_df[predictions_df['has_label']]
    
    if len(labeled_data) > 0:
        axes[0].scatter(labeled_data['true_bug'], labeled_data['predicted_bug'],
                      alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        # 添加对角线
        min_val = min(labeled_data['true_bug'].min(), labeled_data['predicted_bug'].min())
        max_val = max(labeled_data['true_bug'].max(), labeled_data['predicted_bug'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')
        
        # 计算R²和MAE
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(labeled_data['true_bug'], labeled_data['predicted_bug'])
        mae = mean_absolute_error(labeled_data['true_bug'], labeled_data['predicted_bug'])
        
        axes[0].text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.2f}',
                    transform=axes[0].transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    axes[0].set_xlabel('Actual Bug Count', fontsize=12)
    axes[0].set_ylabel('Predicted Bug Count', fontsize=12)
    axes[0].set_title('Prediction Accuracy (Labeled Samples)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 品种预测排名
    top_n = min(15, len(genotype_predictions))
    df_plot = genotype_predictions.head(top_n)
    
    y_pos = np.arange(len(df_plot))
    colors = ['green' if row['has_label'] else 'orange' 
             for _, row in df_plot.iterrows()]
    
    axes[1].barh(y_pos, df_plot['predicted_bug'], color=colors, alpha=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(df_plot['genotype'], fontsize=9)
    axes[1].set_xlabel('Predicted Bug Count', fontsize=12)
    axes[1].set_title(f'Top {top_n} Genotypes with Lowest Predicted Bug Count', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='With Measured Data'),
        Patch(facecolor='orange', alpha=0.7, label='Predicted Only')
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # 3. 特征重要性
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-20:][::-1]
    
    axes[2].barh(range(len(top_features)), feature_importance[top_features], 
                color='steelblue', alpha=0.7)
    axes[2].set_yticks(range(len(top_features)))
    axes[2].set_yticklabels([f'Feature {i}' for i in top_features], fontsize=9)
    axes[2].set_xlabel('Feature Importance', fontsize=12)
    axes[2].set_title(f'Top 20 Important Features ({feature_type.upper()})', fontsize=14, fontweight='bold')
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = output_dir / '3_bug_prediction.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ 预测可视化图已保存")
    plt.close()


def visualize_multi_indicator_predictions(multi_indicator_summary, indicator_models, indicator_cv_results, 
                                          output_dir, feature_type):
    """可视化多指标预测结果 - 生成4_multi_indicator_prediction.png"""
    print(f"    ✓ 生成多指标预测可视化...")
    
    indicators = {
        'leaf_retention': ('Leaf Retention Rate', '%', 'green'),
        'grain_yield': ('Grain Yield', 'kg/ha', 'orange'),
        'seed_weight': ('Seed Weight', 'g', 'purple'),
        'agronomic_value': ('Agronomic Value', 'score', 'blue')
    }
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3,
                  left=0.08, right=0.96, top=0.94, bottom=0.06)
    
    plot_idx = 0
    for ind_key, (ind_name, unit, color) in indicators.items():
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # 获取有标签的数据
        labeled = multi_indicator_summary[multi_indicator_summary['has_label']].copy()
        
        pred_col = f'predicted_{ind_key}'
        true_col = f'true_{ind_key}'
        
        # 过滤NaN
        valid_mask = ~(labeled[pred_col].isna() | labeled[true_col].isna())
        labeled = labeled[valid_mask]
        
        if len(labeled) > 0:
            pred = labeled[pred_col].values
            true = labeled[true_col].values
            
            # 散点图
            ax.scatter(true, pred, alpha=0.6, s=100, color=color, 
                      edgecolors='black', linewidth=0.8, label='Predictions')
            
            # 对角线
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                   linewidth=2, alpha=0.5, label='Perfect Prediction')
            
            # 计算指标
            from sklearn.metrics import r2_score, mean_absolute_error
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred)
            
            # CV结果
            cv_info = indicator_cv_results.get(ind_key, {})
            cv_mae = cv_info.get('mae_mean', 0)
            cv_std = cv_info.get('mae_std', 0)
            
            # 文本框
            stats_text = f'R² = {r2:.3f}\nMAE = {mae:.2f} {unit}\nCV MAE = {cv_mae:.2f}±{cv_std:.2f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(f'True {ind_name} ({unit})', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'Predicted {ind_name} ({unit})', fontsize=11, fontweight='bold')
            ax.set_title(f'({chr(65+plot_idx)}) {ind_name}', fontsize=12, fontweight='bold', loc='left')
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plot_idx += 1
    
    # 第5个子图：CV性能对比柱状图
    ax5 = fig.add_subplot(gs[1, 2])
    
    ind_names = []
    mae_means = []
    mae_stds = []
    colors_bar = []
    
    for ind_key, (ind_name, unit, color) in indicators.items():
        cv_info = indicator_cv_results.get(ind_key, {})
        if cv_info:
            ind_names.append(ind_name.replace(' Rate', '').replace(' Value', ''))
            mae_means.append(cv_info['mae_mean'])
            mae_stds.append(cv_info['mae_std'])
            colors_bar.append(color)
    
    if len(ind_names) > 0:
        x_pos = np.arange(len(ind_names))
        bars = ax5.bar(x_pos, mae_means, yerr=mae_stds, capsize=5, 
                      color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # 添加数值标签
        for i, (mean, std) in enumerate(zip(mae_means, mae_stds)):
            ax5.text(i, mean + std + max(mae_means)*0.02, f'{mean:.2f}±{std:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(ind_names, rotation=15, ha='right', fontsize=10)
        ax5.set_ylabel('CV MAE', fontsize=11, fontweight='bold')
        ax5.set_title(f'(E) Cross-Validation Performance', fontsize=12, fontweight='bold', loc='left')
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 保存
    save_path = output_dir / '4_multi_indicator_prediction.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ 多指标预测可视化图已保存")
    plt.close()




def visualize_genotype_images(genotype_name, num_timepoints=8, feature_type='dinov3'):
    """
    可视化同一品种在Control和NoControl环境下的图像对比
    
    Args:
        genotype_name: 品种名称，如 'RIL 145'
        num_timepoints: 显示的时间点数量（默认8个）
        feature_type: 特征类型（用于输出目录）
    """
    print(f"\n{'='*80}")
    print(f"Visualizing Images: {genotype_name} (Control vs NoControl)")
    print(f"{'='*80}\n")
    
    # 项目根目录（向上3级：visualization -> insect_resistance -> experiments -> root）
    project_root = Path(__file__).parent.parent.parent.parent
    image_dir = project_root / 'AnhumasPiracicaba' / 'dataset' / 'images'
    
    # 输出目录使用module内部目录
    module_dir = Path(__file__).parent.parent
    output_dir = module_dir / 'outputs' / 'results' / feature_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找该品种的图像文件
    genotype_folder = genotype_name.replace(' ', '_')
    control_dir = image_dir / 'control' / genotype_name
    nocontrol_dir = image_dir / 'nocontrol' / genotype_name
    
    if not control_dir.exists() or not nocontrol_dir.exists():
        print(f"✗ Error: Image directory not found for {genotype_name}")
        print(f"  Expected paths:")
        print(f"    Control: {control_dir}")
        print(f"    NoControl: {nocontrol_dir}")
        return
    
    # 获取所有图像文件（按文件名排序）
    all_control_images = sorted(list(control_dir.glob('*.png')))
    all_nocontrol_images = sorted(list(nocontrol_dir.glob('*.png')))
    
    if len(all_control_images) == 0 or len(all_nocontrol_images) == 0:
        print(f"✗ Error: No images found for {genotype_name}")
        return
    
    # 提取唯一的日期（同一日期可能有多个plot）
    def extract_date(img_path):
        parts = img_path.stem.split('_')
        return '_'.join(parts[-3:])  # '2025_01_28'
    
    # 获取所有唯一日期
    control_dates = {}
    for img in all_control_images:
        date = extract_date(img)
        if date not in control_dates:
            control_dates[date] = img
    
    nocontrol_dates = {}
    for img in all_nocontrol_images:
        date = extract_date(img)
        if date not in nocontrol_dates:
            nocontrol_dates[date] = img
    
    # 找到共同的日期
    common_dates = sorted(set(control_dates.keys()) & set(nocontrol_dates.keys()))
    n_times = min(len(common_dates), num_timepoints)
    
    if n_times == 0:
        print(f"✗ Error: No matching dates found for {genotype_name}")
        return
    
    # 只使用前n_times个日期
    selected_dates = common_dates[:n_times]
    
    # 按原图纵横比自适应画布，避免图像被拉伸变形
    sample_img = plt.imread(control_dates[selected_dates[0]])
    img_h, img_w = sample_img.shape[0], sample_img.shape[1]
    img_aspect = img_w / max(img_h, 1)
    row_height = 3.2
    fig_height = row_height * 2 + 0.2
    fig_width = max(8.0, n_times * row_height * img_aspect + 0.6)

    # 创建对比图（布局紧凑，给总标题预留足够空间避免重叠）
    fig, axes = plt.subplots(2, n_times, figsize=(fig_width, fig_height))
    if n_times == 1:
        axes = axes.reshape(2, 1)
    
    for i, date in enumerate(selected_dates):
        # 加载图像
        control_img = plt.imread(control_dates[date])
        nocontrol_img = plt.imread(nocontrol_dates[date])
        
        # 格式化日期显示
        date_parts = date.split('_')
        date_display = f"{date_parts[0][-2:]}-{date_parts[1]}-{date_parts[2]}"  # '25-01-28'
        
        # 显示Control图像
        axes[0, i].imshow(control_img)
        panel_top = chr(65 + i)
        axes[0, i].text(
            0.5, -0.055,
            f'({panel_top}) Control {date_display}',
            transform=axes[0, i].transAxes,
            ha='center', va='top', fontsize=9, fontweight='bold'
        )
        axes[0, i].axis('off')
        
        # 显示NoControl图像
        axes[1, i].imshow(nocontrol_img)
        panel_bottom = chr(65 + n_times + i)
        axes[1, i].text(
            0.5, -0.055,
            f'({panel_bottom}) NoControl {date_display}',
            transform=axes[1, i].transAxes,
            ha='center', va='top', fontsize=9, fontweight='bold'
        )
        axes[1, i].axis('off')

    # 紧凑布局并显著收缩列间空白，同时缩小两行之间的空白。
    plt.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.02, wspace=0.005, hspace=0.01)
    
    # 保存
    save_path = output_dir / f'image_comparison_{genotype_folder}.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Image comparison saved: {save_path}")
    print(f"  - {n_times} timepoints visualized")
    print(f"\n{'='*80}\n")


def visualize_yield_comparison(feature_type='dinov3'):
    """
    可视化30个品种的Control vs NoControl产量对比
    包含2个子图：
    1. 并排条形图（Control vs NoControl）
    2. 稳定化产量增益率（与3D时序图一致）
    
    Args:
        feature_type: 特征类型，用于输出目录
    """
    from ..core.analyzer import MultiModalInsectResistanceAnalyzer
    
    print(f"\n{'='*80}")
    print(f"Yield Comparison Visualization: Control vs NoControl")
    print(f"{'='*80}\n")
    
    # 初始化分析器
    analyzer = MultiModalInsectResistanceAnalyzer(feature_type=feature_type)
    
    # 收集数据
    genotypes = sorted(analyzer.control_df['genotype'].unique())
    data = []
    
    for genotype in genotypes:
        control_yield = analyzer.control_df[
            analyzer.control_df['genotype'] == genotype
        ]['grain_yield'].mean()
        
        nocontrol_yield = analyzer.nocontrol_df[
            analyzer.nocontrol_df['genotype'] == genotype
        ]['grain_yield'].mean()
        
        loss_rate = (control_yield - nocontrol_yield) / control_yield * 100 if control_yield > 0 else 0
        
        data.append({
            'genotype': genotype,
            'control_yield': control_yield,
            'nocontrol_yield': nocontrol_yield,
            'loss_rate': loss_rate,
            'yield_change': control_yield - nocontrol_yield
        })
    
    df = pd.DataFrame(data)
    tau_gain = infer_gain_stabilizer(df['control_yield'].values, quantile=10.0, floor=1.0)
    df['yield_gain_rate'] = compute_gain_rate(
        df['control_yield'].values,
        df['nocontrol_yield'].values,
        tau=tau_gain,
    )
    df = df.sort_values('nocontrol_yield', ascending=False)  # 按NoControl产量从高到低排序
    
    # ========================================
    # 单独保存散点图
    # ========================================
    fig_scatter = plt.figure(figsize=(10, 10))
    ax_scatter = fig_scatter.add_subplot(111)
    
    # 根据损失率着色
    colors_scatter = ['green' if lr < 0 else 'orange' if lr < 20 else 'red' for lr in df['loss_rate']]
    
    scatter = ax_scatter.scatter(df['control_yield'], df['nocontrol_yield'], 
                         s=150, c=colors_scatter, alpha=0.7,
                         edgecolors='black', linewidth=1.2)
    
    # 添加对角线（无变化线）
    min_val = min(df['control_yield'].min(), df['nocontrol_yield'].min())
    max_val = max(df['control_yield'].max(), df['nocontrol_yield'].max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', 
            alpha=0.5, linewidth=2, label='No Change Line')
    
    # 标注极端品种
    max_gain_idx = df['yield_change'].idxmax()
    ax_scatter.annotate(df.loc[max_gain_idx, 'genotype'],
                (df.loc[max_gain_idx, 'control_yield'], df.loc[max_gain_idx, 'nocontrol_yield']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
    
    max_loss_idx = df['yield_change'].idxmin()
    ax_scatter.annotate(df.loc[max_loss_idx, 'genotype'],
                (df.loc[max_loss_idx, 'control_yield'], df.loc[max_loss_idx, 'nocontrol_yield']),
                xytext=(-10, -10), textcoords='offset points',
                fontsize=10, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
    
    ax_scatter.set_xlabel('Control Yield (kg/ha)', fontsize=13, fontweight='bold')
    ax_scatter.set_ylabel('NoControl Yield (kg/ha)', fontsize=13, fontweight='bold')
    ax_scatter.grid(True, alpha=0.3, linestyle=':')
    ax_scatter.set_aspect('equal', adjustable='box')
    
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Yield Gain (Loss < 0%)'),
        Patch(facecolor='orange', alpha=0.7, label='Moderate Loss (0-20%)'),
        Patch(facecolor='red', alpha=0.7, label='High Loss (> 20%)')
    ]
    ax_scatter.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    scatter_path = analyzer.output_dir / f'yield_scatter_plot_{feature_type}.png'
    _save_png_and_pdf(scatter_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Scatter plot saved: {scatter_path}")
    
    # ========================================
    # 创建主图形：2行1列
    # ========================================
    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.25, 1.0], hspace=0.32)
    
    # ========================================
    # 子图1：并排条形图（Control vs NoControl）
    # ========================================
    ax1 = fig.add_subplot(gs[0])
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['control_yield'], width, label='Control',
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, df['nocontrol_yield'], width, label='NoControl',
                    color='coral', alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # 标注产量变化（增加或损失的具体数值）
    y_top_limit = max(df['control_yield'].max(), df['nocontrol_yield'].max()) + 220
    ax1.set_ylim(0, y_top_limit)

    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        yield_change = row['nocontrol_yield'] - row['control_yield']
        max_height = max(row['control_yield'], row['nocontrol_yield'])
        label_y = min(max_height + 45, y_top_limit - 25)
        
        if yield_change > 0:  # 产量增加
            ax1.text(idx, label_y, f'+{yield_change:.0f}',
                    ha='center', va='bottom', fontsize=9, color='green', fontweight='bold', clip_on=True)
        else:  # 产量损失
            ax1.text(idx, label_y, f'{yield_change:.0f}',
                    ha='center', va='bottom', fontsize=9, color='red', fontweight='bold', clip_on=True)
    
    ax1.set_xlabel('', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Grain Yield (kg/ha)', fontsize=18, fontweight='bold')
    ax1.set_title('', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['genotype'], rotation=45, ha='right', fontsize=12)
    ax1.tick_params(axis='y', labelsize=13)
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax1.text(
        0.5,
        -0.18,
        'a',
        transform=ax1.transAxes,
        ha='center',
        va='top',
        fontsize=16,
        fontweight='bold',
    )
    
    # 添加均值线
    ax1.axhline(df['control_yield'].mean(), color='steelblue', linestyle='--', 
               alpha=0.6, linewidth=2, label=f"Control Mean: {df['control_yield'].mean():.0f}")
    ax1.axhline(df['nocontrol_yield'].mean(), color='coral', linestyle='--', 
               alpha=0.6, linewidth=2, label=f"NoControl Mean: {df['nocontrol_yield'].mean():.0f}")
    
    # ========================================
    # 子图2：稳定化产量增益率条形图（与3D时序分析一致）
    # ========================================
    ax2 = fig.add_subplot(gs[1])
    
    # 重新排序：按稳定化增益率从大到小
    df_sorted_by_change = df.copy().sort_values('yield_gain_rate', ascending=False)
    
    x_pos = np.arange(len(df_sorted_by_change))
    
    # 根据增益率着色
    colors_change = []
    for gain_rate in df_sorted_by_change['yield_gain_rate']:
        if gain_rate > 0:
            colors_change.append('green')
        elif gain_rate > -0.10:
            colors_change.append('gold')
        elif gain_rate > -0.25:
            colors_change.append('orange')
        else:
            colors_change.append('red')
    
    # 绘制条形图
    bars = ax2.bar(x_pos, df_sorted_by_change['yield_gain_rate'], 
                   color=colors_change, alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # 添加零线
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # 给上下留白，避免数值标注超出边框。
    gain_vals = df_sorted_by_change['yield_gain_rate'].to_numpy(dtype=float)
    y_min = float(np.nanmin(gain_vals))
    y_max = float(np.nanmax(gain_vals))
    y_span = max(y_max - y_min, 0.02)
    y_pad = max(0.015, 0.10 * y_span)
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)

    # 标注数值
    y_low, y_high = ax2.get_ylim()
    for i, (_, row) in enumerate(df_sorted_by_change.iterrows()):
        gain_rate = float(row['yield_gain_rate'])
        if abs(gain_rate) > 0.03:  # 只标注较大的值
            if gain_rate >= 0:
                y_text = min(gain_rate + 0.012, y_high - 0.006)
                va = 'top' if y_text >= y_high - 0.006 else 'bottom'
            else:
                y_text = max(gain_rate - 0.012, y_low + 0.006)
                va = 'bottom' if y_text <= y_low + 0.006 else 'top'
            ax2.text(
                i,
                y_text,
                f'{gain_rate:.3f}',
                ha='center',
                va=va,
                fontsize=9,
                fontweight='bold',
                clip_on=True,
            )
    
    ax2.set_xlabel('', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Yield Gain Rate', fontsize=18, fontweight='bold')
    ax2.set_title('', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_sorted_by_change['genotype'], rotation=45, ha='right', fontsize=12)
    ax2.tick_params(axis='y', labelsize=13)
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax2.text(
        0.5,
        -0.23,
        'b',
        transform=ax2.transAxes,
        ha='center',
        va='top',
        fontsize=16,
        fontweight='bold',
    )
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Positive Gain (> 0)'),
        Patch(facecolor='gold', alpha=0.8, label='Slight Decline (0 to -0.10)'),
        Patch(facecolor='orange', alpha=0.8, label='Moderate Decline (-0.10 to -0.25)'),
        Patch(facecolor='red', alpha=0.8, label='Large Decline (< -0.25)')
    ]
    ax2.legend(handles=legend_elements, loc='lower left', fontsize=13, framealpha=0.9)

    fig.subplots_adjust(hspace=0.42)
    
    # 保存
    save_path = analyzer.output_dir / f'yield_comparison_comprehensive_{feature_type}.png'
    _save_png_and_pdf(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Yield comparison visualization saved: {save_path}")
    
    # 保存数据
    csv_path = analyzer.output_dir / f'yield_comparison_data_{feature_type}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Yield comparison data saved: {csv_path}")
    
    # 统计摘要
    print(f"\n{'='*80}")
    print("Yield Comparison Summary")
    print(f"{'='*80}")
    print(f"Total Genotypes: {len(df)}")
    print(f"\nControl Yield:")
    print(f"  Mean: {df['control_yield'].mean():.1f} kg/ha")
    print(f"  Range: [{df['control_yield'].min():.1f}, {df['control_yield'].max():.1f}]")
    print(f"\nNoControl Yield:")
    print(f"  Mean: {df['nocontrol_yield'].mean():.1f} kg/ha")
    print(f"  Range: [{df['nocontrol_yield'].min():.1f}, {df['nocontrol_yield'].max():.1f}]")
    print(f"\nStabilized Yield Gain Rate:")
    print(f"  tau (denominator stabilizer): {tau_gain:.3f}")
    print(f"  Mean: {df['yield_gain_rate'].mean():.3f}")
    print(f"  Std:  {df['yield_gain_rate'].std():.3f}")
    print(f"  Range: [{df['yield_gain_rate'].min():.3f}, {df['yield_gain_rate'].max():.3f}]")
    print(f"\n{'='*80}\n")
    
    return df

