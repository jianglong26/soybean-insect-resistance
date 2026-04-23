"""
Analysis Workflows
完整分析工作流
"""
from ..core.analyzer import MultiModalInsectResistanceAnalyzer
from ..visualization import *
import json
from pathlib import Path
import datetime
from src.config import Config

def run_single_experiment(feature_type):
    """运行单个实验"""
    print(f"\n\n{'#'*80}")
    print(f"# EXPERIMENT: {feature_type.upper()}")
    print(f"{'#'*80}\n")
    
    # 初始化分析器
    analyzer = MultiModalInsectResistanceAnalyzer(feature_type=feature_type)
    
    # 计算基础抗性指标（包含score_without_bug和初始的score_with_bug）
    resistance_df = analyzer.calculate_resistance_indices()
    
    # 预测虫害
    prediction_df, genotype_summary, model, cv_mae = analyzer.predict_bug_from_features()
    
    # 预测多个指标（叶片保持率、产量、种子重量、农艺价值）
    multi_indicator_summary, indicator_models, indicator_cv_results = analyzer.predict_multiple_indicators()
    
    # ⭐ 关键：用预测的bug数据补充完整score_with_bug
    resistance_df = analyzer.complete_score_with_predictions(resistance_df, genotype_summary)
    
    # 重新保存更新后的结果（包含两种评分）
    save_path = analyzer.output_dir / 'resistance_indices.csv'
    resistance_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✓ Resistance indices saved with both scoring methods")
    print(f"  - score_without_bug: 基础评分（不含虫子，30个品种）")
    print(f"  - score_with_bug: 完整评分（含虫子实测+预测，30个品种）")
    
    # 返回关键指标（基于不含虫子的评分排序，保持一致性）
    all_genotypes = resistance_df.sort_values('score_without_bug', ascending=False)
    top10_genotypes = all_genotypes.head(10)['genotype'].tolist()
    
    return {
        'feature_type': feature_type,
        'cv_mae': cv_mae.mean(),
        'cv_mae_std': cv_mae.std(),
        'top10_genotypes': top10_genotypes,
        'n_features': prediction_df.iloc[0]['predicted_bug'] if len(prediction_df) > 0 else 0,
        'resistance_df': resistance_df,
        'genotype_summary': genotype_summary,
        'predictions_df': prediction_df,
        'model': model,
        'multi_indicator_summary': multi_indicator_summary,
        'indicator_models': indicator_models,
        'indicator_cv_results': indicator_cv_results
    }


def generate_comparison_report(results):
    """生成三组实验对比报告"""
    print(f"\n\n{'='*80}")
    print("COMPARISON REPORT: DINOv3 vs VI vs Fusion")
    print(f"{'='*80}\n")
    
    # 对比表格
    print(f"{'Metric':<30} {'DINOv3':<20} {'VI':<20} {'Fusion':<20}")
    print("-" * 90)
    
    for metric_name, metric_key in [
        ('Feature Dimension', 'feature_dim'),
        ('CV MAE (mean)', 'cv_mae'),
        ('CV MAE (std)', 'cv_mae_std')
    ]:
        if metric_key == 'feature_dim':
            values = ['384', '24', '408']
        else:
            values = [f"{r[metric_key]:.2f}" for r in results]
        
        print(f"{metric_name:<30} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
    
    print(f"\n{'Top 10 Genotypes Overlap':<30}")
    print("-" * 90)
    
    dinov3_top10 = set(results[0]['top10_genotypes'])
    vi_top10 = set(results[1]['top10_genotypes'])
    fusion_top10 = set(results[2]['top10_genotypes'])
    
    overlap_all = dinov3_top10 & vi_top10 & fusion_top10
    overlap_dv_f = dinov3_top10 & fusion_top10
    overlap_vi_f = vi_top10 & fusion_top10
    
    print(f"{'All three agree:':<30} {len(overlap_all)}/10 genotypes")
    print(f"{'DINOv3 & Fusion:':<30} {len(overlap_dv_f)}/10 genotypes")
    print(f"{'VI & Fusion:':<30} {len(overlap_vi_f)}/10 genotypes")
    
    print(f"\n{'Common genotypes (all three):':<30}")
    if overlap_all:
        for g in sorted(overlap_all):
            print(f"  • {g}")
    else:
        print("  (None)")
    
    # 保存对比报告
    report_dir = Path(__file__).parent.parent / 'outputs' / 'results' / 'comparison'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    with open(report_dir / 'comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Multi-Modal Feature Comparison Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Experiment Design:\n")
        f.write("-" * 80 + "\n")
        f.write("1. DINOv3-only: Deep learning features (384-dim)\n")
        f.write("2. VI-only: Vegetation indices (24-dim)\n")
        f.write("3. Fusion: Combined features (408-dim)\n\n")
        
        f.write("Prediction Performance:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Feature Type':<20} {'CV MAE':<15} {'Feature Dim':<15}\n")
        f.write("-" * 80 + "\n")
        for i, r in enumerate(results):
            dim = ['384', '24', '408'][i]
            f.write(f"{r['feature_type'].upper():<20} {r['cv_mae']:.2f} ± {r['cv_mae_std']:.2f}{'':<5} {dim:<15}\n")
        
        f.write(f"\nTop 10 Genotypes Ranking:\n")
        f.write("-" * 80 + "\n")
        
        for i, r in enumerate(results):
            f.write(f"\n{r['feature_type'].upper()}:\n")
            for j, g in enumerate(r['top10_genotypes'], 1):
                f.write(f"  {j}. {g}\n")
        
        f.write(f"\nGenotype Overlap Analysis:\n")
        f.write("-" * 80 + "\n")
        f.write(f"All three agree: {len(overlap_all)}/10 genotypes\n")
        f.write(f"DINOv3 & Fusion: {len(overlap_dv_f)}/10 genotypes\n")
        f.write(f"VI & Fusion: {len(overlap_vi_f)}/10 genotypes\n")
        
        if overlap_all:
            f.write(f"\nCommon genotypes (consensus):\n")
            for g in sorted(overlap_all):
                f.write(f"  • {g}\n")
    
    print(f"\n✓ Comparison report saved: {report_dir / 'comparison_report.txt'}")
    
    return report_dir




def run_prediction_experiments():
    """
    运行三组对比实验 (DINOv3, VI, Fusion)
    
    包含虫害预测、多指标预测、抗性评分等完整分析流程。
    """
    print(f"\n{'#'*80}")
    print("# MULTI-MODAL INSECT RESISTANCE ANALYSIS")
    print("# Three-Group Comparison Experiment")
    print(f"{'#'*80}\n")
    
    results = []
    
    # 加载元数据（用于增强可视化）
    module_dir = Path(__file__).parent.parent
    metadata_path = Config.ANNOTATION_PATH
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 实验1: DINOv3特征
    result1 = run_single_experiment('dinov3')
    results.append(result1)
    
    # 为DINOv3生成所有可视化
    print(f"\n  生成DINOv3可视化...")
    output_dir = module_dir / 'outputs' / 'results' / 'dinov3'
    plot_control_bug_distribution(metadata, output_dir)
    visualize_resistance_ranking(result1['resistance_df'], output_dir)
    visualize_bug_predictions(result1['predictions_df'], result1['resistance_df'], result1['model'], output_dir, 'dinov3')
    visualize_multi_indicator_predictions(result1['multi_indicator_summary'], result1['indicator_models'], 
                                         result1['indicator_cv_results'], output_dir, 'dinov3')
    create_two_rankings_all30(result1['resistance_df'], output_dir, 'dinov3')
    
    # 实验2: 植被指数特征
    result2 = run_single_experiment('vi')
    results.append(result2)
    
    # 为VI生成所有可视化
    print(f"\n  生成VI可视化...")
    output_dir = module_dir / 'outputs' / 'results' / 'vi'
    plot_control_bug_distribution(metadata, output_dir)
    visualize_resistance_ranking(result2['resistance_df'], output_dir)
    visualize_bug_predictions(result2['predictions_df'], result2['resistance_df'], result2['model'], output_dir, 'vi')
    visualize_multi_indicator_predictions(result2['multi_indicator_summary'], result2['indicator_models'], 
                                         result2['indicator_cv_results'], output_dir, 'vi')
    create_two_rankings_all30(result2['resistance_df'], output_dir, 'vi')
    
    # 实验3: 融合特征
    result3 = run_single_experiment('fusion')
    results.append(result3)
    
    # 为Fusion生成所有可视化
    print(f"\n  生成Fusion可视化...")
    output_dir = module_dir / 'outputs' / 'results' / 'fusion'
    plot_control_bug_distribution(metadata, output_dir)
    visualize_resistance_ranking(result3['resistance_df'], output_dir)
    visualize_bug_predictions(result3['predictions_df'], result3['resistance_df'], result3['model'], output_dir, 'fusion')
    visualize_multi_indicator_predictions(result3['multi_indicator_summary'], result3['indicator_models'], 
                                         result3['indicator_cv_results'], output_dir, 'fusion')
    create_two_rankings_all30(result3['resistance_df'], output_dir, 'fusion')
    
    # 生成对比报告
    report_dir = generate_comparison_report(results)
    
    print(f"\n{'='*80}")
    print("✓ ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print(f"\nResults saved in:")
    print(f"  • DINOv3:  outputs/results/insect_resistance_dinov3/")
    print(f"      - control_bug_distribution.png (Control vs NoControl对比 + Top15)")
    print(f"      - 2_resistance_ranking.png (综合抗性排名分析)")
    print(f"      - 3_bug_prediction.png (虫害预测结果)")
    print(f"      - three_rankings_dinov3.png (三种排名策略)")
    print(f"      - resistance_indices.csv, bug_predictions.csv")
    print(f"  • VI:      outputs/results/insect_resistance_vi/")
    print(f"      - control_bug_distribution.png")
    print(f"      - 2_resistance_ranking.png")
    print(f"      - 3_bug_prediction.png")
    print(f"      - three_rankings_vi.png")
    print(f"      - resistance_indices.csv, bug_predictions.csv")
    print(f"  • Fusion:  outputs/results/insect_resistance_fusion/")
    print(f"      - control_bug_distribution.png")
    print(f"      - 2_resistance_ranking.png")
    print(f"      - 3_bug_prediction.png")
    print(f"      - three_rankings_fusion.png")
    print(f"      - resistance_indices.csv, bug_predictions.csv")
    print(f"  • Compare: {report_dir}/")
    print(f"      - comparison_report.txt, performance_summary.csv")
    print()


