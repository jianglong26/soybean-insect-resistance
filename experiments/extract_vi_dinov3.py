"""
实验脚本2: 提取植被指数特征
从所有图像中提取传统植被指数并保存


实验脚本1: 提取DINOv3特征
从所有图像中提取DINOv3深度特征并保存

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import pickle
from tqdm import tqdm
from src.config import Config
from src.features import VegetationIndicesExtractor
import cv2
from src.features import DINOv3FeatureExtractor
from PIL import Image


def gray_world_white_balance(rgb: np.ndarray) -> np.ndarray:
    """Mild Gray-World white balance with clamped channel scaling."""
    img = rgb.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    global_mean = float(np.mean(means) + 1e-8)
    scales = np.clip(global_mean / (means + 1e-8), 0.95, 1.05)
    balanced = img * scales[None, None, :]
    return np.clip(balanced, 0, 255).astype(np.uint8)


def normalize_lighting_lab(
    rgb: np.ndarray,
    target_median: float = 125.0,
    max_delta: float = 14.0,
    shadow_gamma: float = 2.2,
    strength: float = 0.30,
) -> np.ndarray:
    """Conservative illumination normalization: lift dark tones only."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l_src = l.astype(np.float32)
    median_l = float(np.median(l_src))
    delta = float(np.clip(target_median - median_l, -max_delta, max_delta))

    shadow_weight = np.power(1.0 - (l_src / 255.0), shadow_gamma)
    l_adj = np.clip(l_src + delta * shadow_weight, 0, 255)

    alpha = float(np.clip(strength, 0.0, 1.0))
    l_mix = (1.0 - alpha) * l_src + alpha * l_adj
    l_u8 = np.clip(l_mix, 0, 255).astype(np.uint8)

    out_lab = cv2.merge([l_u8, a, b])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)


def postprocess_mask(mask: np.ndarray, min_area_ratio: float = 0.003) -> np.ndarray:
    """Morphological cleanup + connected-component area filtering."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    h, w = mask_u8.shape
    min_area = int(min_area_ratio * h * w)

    clean = np.zeros_like(mask_u8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255
    return clean


def build_foreground_mask(rgb: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """Foreground mask with gray-range gating used by current mask baseline."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lo = int(np.clip(low_threshold, 0, 255))
    hi = int(np.clip(high_threshold, 0, 255))
    if lo >= hi:
        lo = max(0, hi - 1)
    raw_mask = ((gray >= lo) & (gray < hi)).astype(np.uint8) * 255
    return postprocess_mask(raw_mask)


def extract_masked_vi_features(
    extractor: VegetationIndicesExtractor,
    image: np.ndarray,
    mask: np.ndarray,
    feature_names,
) -> np.ndarray:
    """Compute VI statistics on foreground pixels only."""
    if image.dtype == np.uint8:
        image_f = image.astype(np.float32) / 255.0
    elif image.max() > 1.0:
        image_f = image.astype(np.float32) / 255.0
    else:
        image_f = image.astype(np.float32)

    valid = (mask > 0)
    if valid.sum() < 10:
        # Fallback to full-image extraction if the mask is nearly empty.
        features = extractor.extract(image, return_stats=True)
        return np.array([features[name] for name in feature_names], dtype=np.float32)

    R = image_f[:, :, 0]
    G = image_f[:, :, 1]
    B = image_f[:, :, 2]

    sum_rgb = R + G + B + 1e-8
    r = R / sum_rgb
    g = G / sum_rgb
    b = B / sum_rgb

    features = {}
    for index_name in extractor.indices:
        index_map = extractor.available_indices[index_name](R, G, B, r, g, b)
        vals = index_map[valid]
        features[f'{index_name}_mean'] = float(np.mean(vals))
        features[f'{index_name}_std'] = float(np.std(vals))
        features[f'{index_name}_median'] = float(np.median(vals))

    features['R_mean'] = float(np.mean(R[valid]))
    features['G_mean'] = float(np.mean(G[valid]))
    features['B_mean'] = float(np.mean(B[valid]))
    features['R_std'] = float(np.std(R[valid]))
    features['G_std'] = float(np.std(G[valid]))
    features['B_std'] = float(np.std(B[valid]))

    return np.array([features[name] for name in feature_names], dtype=np.float32)


def extract_vegetation_indices():
    """提取植被指数特征并保存"""
    
    # 创建输出目录
    Config.create_dirs()
    
    # 初始化特征提取器
    print("\n" + "="*60)
    print("步骤1: 初始化植被指数提取器")
    print("="*60)
    
    extractor = VegetationIndicesExtractor(indices=Config.VEGETATION_INDICES)
    feature_names = extractor.get_feature_names(return_stats=True)
    
    print(f"计算的植被指数: {Config.VEGETATION_INDICES}")
    print(f"特征维度: {extractor.get_feature_dim(return_stats=True)}")
    print(f"特征名称: {feature_names[:5]}... (共{len(feature_names)}个)")
    
    # 加载元数据
    print("\n" + "="*60)
    print("步骤2: 加载数据集元数据")
    print("="*60)
    
    with open(Config.ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"总样本数: {len(metadata)}")
    
    # 按环境分别处理
    for environment in Config.ENVIRONMENTS:
        print("\n" + "="*60)
        print(f"步骤3: 提取 {environment.upper()} 环境的特征")
        print("="*60)
        
        # 筛选该环境的样本
        env_samples = {k: v for k, v in metadata.items() if v['environment'] == environment}
        print(f"{environment} 样本数: {len(env_samples)}")
        
        # 存储特征
        features_dict = {}
        
        # 遍历每个样本
        for sample_key, sample_data in tqdm(env_samples.items(), desc=f"提取{environment}植被指数"):
            image_sequence = sample_data['image_sequence']
            genotype = sample_data['genotype']
            block = sample_data['block']
            
            # 提取时序特征
            timeseries_features = []
            for img_info in image_sequence:
                img_path = Config.IMAGE_ROOT / img_info['path']
                
                try:
                    # 加载图像
                    image = cv2.imread(str(img_path))
                    if image is None:
                        raise ValueError(f"无法加载图像: {img_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # 按当前阈值流程做背景去除后，仅在前景上提取VI统计特征。
                    image_light = normalize_lighting_lab(gray_world_white_balance(image))
                    foreground_mask = build_foreground_mask(image_light, low_threshold=50, high_threshold=150)
                    feature_vector = extract_masked_vi_features(
                        extractor=extractor,
                        image=image_light,
                        mask=foreground_mask,
                        feature_names=feature_names,
                    )
                    timeseries_features.append(feature_vector)
                    
                except Exception as e:
                    print(f"\n⚠️ 处理失败: {img_path}")
                    print(f"   错误: {e}")
                    continue
            
            # 保存该样本的时序特征
            if timeseries_features:
                features_dict[sample_key] = {
                    'features': np.array(timeseries_features),  # shape: (T, D)
                    'feature_names': feature_names,
                    'genotype': genotype,
                    'block': block,
                    'dates': [img['date'] for img in image_sequence],
                    'labels': sample_data['labels']
                }
        
        # 保存特征
        output_path = Config.get_feature_path('vegetation_indices', environment)
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f)
        
        print(f"\n✓ {environment} 特征已保存到: {output_path}")
        print(f"  样本数: {len(features_dict)}")
        print(f"  特征维度: {len(feature_names)}")
        print(f"  示例特征shape: {list(features_dict.values())[0]['features'].shape}")
    
    print("\n" + "="*60)
    print("✓ 所有植被指数特征提取完成！")
    print("="*60)
    print(f"\n特征保存位置: {Config.FEATURE_ROOT / 'vegetation_indices'}")
    print("\n下一步:")
    print("  - 使用这些特征进行统计分析")
    print("  - 或与DINOv3特征融合进行深度学习")





def extract_dinov3_features():
    """提取DINOv3特征并保存

        实验脚本1: 提取DINOv3特征
        从所有图像中提取DINOv3深度特征并保存

    """
    
    # 创建输出目录
    Config.create_dirs()
    
    # 初始化特征提取器
    print("\n" + "="*60)
    print("步骤1: 初始化DINOv3特征提取器")
    print("="*60)
    
    extractor = DINOv3FeatureExtractor(
        model_name=Config.DINOV3_MODEL,
        checkpoint_path=str(Config.DINOV3_CHECKPOINT) if Config.DINOV3_CHECKPOINT.exists() else None,
        device=Config.DEVICE,
        image_size=Config.DINOV3_IMAGE_SIZE
    )
    
    # 加载元数据
    print("\n" + "="*60)
    print("步骤2: 加载数据集元数据")
    print("="*60)
    
    with open(Config.ANNOTATION_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"总样本数: {len(metadata)}")
    
    # 按环境分别处理
    for environment in Config.ENVIRONMENTS:
        print("\n" + "="*60)
        print(f"步骤3: 提取 {environment.upper()} 环境的特征")
        print("="*60)
        
        # 筛选该环境的样本
        env_samples = {k: v for k, v in metadata.items() if v['environment'] == environment}
        print(f"{environment} 样本数: {len(env_samples)}")
        
        # 存储特征
        features_dict = {}
        
        # 遍历每个样本
        for sample_key, sample_data in tqdm(env_samples.items(), desc=f"提取{environment}特征"):
            image_sequence = sample_data['image_sequence']
            genotype = sample_data['genotype']
            block = sample_data['block']
            
            # 提取时序特征
            timeseries_features = []
            for img_info in image_sequence:
                img_path = Config.IMAGE_ROOT / img_info['path']
                
                try:
                    # 加载图像
                    image = Image.open(img_path).convert('RGB')
                    
                    # 提取特征
                    feature = extractor.extract(image)
                    timeseries_features.append(feature)
                    
                except Exception as e:
                    print(f"\n⚠️ 处理失败: {img_path}")
                    print(f"   错误: {e}")
                    continue
            
            # 保存该样本的时序特征
            if timeseries_features:
                features_dict[sample_key] = {
                    'features': np.array(timeseries_features),  # shape: (T, D)
                    'genotype': genotype,
                    'block': block,
                    'dates': [img['date'] for img in image_sequence],
                    'labels': sample_data['labels']
                }
        
        # 保存特征
        output_path = Config.get_feature_path('dinov3', environment)
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f)
        
        print(f"\n✓ {environment} 特征已保存到: {output_path}")
        print(f"  样本数: {len(features_dict)}")
        print(f"  特征维度: {extractor.feature_dim}")
        print(f"  示例特征shape: {list(features_dict.values())[0]['features'].shape}")
    
    print("\n" + "="*60)
    print("✓ 所有DINOv3特征提取完成！")
    print("="*60)
    print(f"\n特征保存位置: {Config.FEATURE_ROOT / 'dinov3'}")
    print("\n下一步:")
    print("  - 运行 extract_vegetation_indices.py 提取植被指数")
    print("  - 或直接使用这些特征进行分析")

    


if __name__ == "__main__":
    extract_vegetation_indices()
    extract_dinov3_features()
