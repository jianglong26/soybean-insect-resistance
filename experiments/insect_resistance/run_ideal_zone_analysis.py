"""
运行IDEAL zone分析，生成独立散点图和统计摘要文件
"""
import sys
from pathlib import Path

# 添加项目根目录和experiments目录到路径
project_root = Path(__file__).parent.parent.parent
experiments_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(experiments_dir))

from insect_resistance.visualization.ideal_zone_analysis import visualize_single_feature_ideal_zone, analyze_ideal_zone_genotypes

if __name__ == "__main__":
    print("=" * 80)
    print("IDEAL ZONE ANALYSIS WITH STANDALONE VISUALIZATIONS")
    print("=" * 80)
    print()
    print("Generating standalone scatter plots with all 30 genotypes labeled...")
    print("Saving statistical summaries as separate text files...")
    print()
    
    # 分析三种特征类型
    for feature_type in ['dinov3', 'vi', 'fusion']:
        print(f"\n{'='*80}")
        print(f"Processing {feature_type.upper()} features...")
        print(f"{'='*80}\n")
        visualize_single_feature_ideal_zone(feature_type)
    
    # 生成跨特征对比（包含独立散点图）
    print(f"\n{'='*80}")
    print("Generating cross-feature comparison...")
    print(f"{'='*80}\n")
    analyze_ideal_zone_genotypes(['dinov3', 'vi', 'fusion'])
    
    print(f"\n{'='*80}")
    print("✅ ALL ANALYSES COMPLETE!")
    print(f"{'='*80}")
    print("\nGenerated files for each feature type (dinov3, vi, fusion):")
    print("  1. ideal_zone_comprehensive_{feature}.png - Main 3-panel visualization")
    print("  2. yield_vs_ideal_zone_scatter_{feature}.png - Standalone scatter with all 30 labeled")
    print("  3. ideal_zone_statistical_summary_{feature}.txt - Detailed text summary")
    print("  4. ideal_zone_detailed_{feature}.csv - Detailed data table")
    print("\nCross-feature files:")
    print("  1. ideal_zone_cross_feature_comparison.png - Main comparison visualization")
    print("  2. yield_vs_ideal_zone_scatter_cross_feature.png - Standalone scatter (all 3 features, all 30 labeled)")
    print("  3. ideal_zone_genotypes_comparison.csv - Comparison data")
    print("  4. ideal_zone_analysis_report.txt - Cross-feature report")
    print()
