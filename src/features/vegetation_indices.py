"""
植被指数特征提取器
从RGB图像计算多种植被指数
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class VegetationIndicesExtractor:
    """植被指数提取器"""
    
    def __init__(self, indices: List[str] = None):
        """
        初始化
        
        Args:
            indices: 要计算的植被指数列表，如果为None则计算所有指数
        """
        self.available_indices = {
            'VDVI': self._compute_vdvi,
            'ExG': self._compute_exg,
            'ExGR': self._compute_exgr,
            'CIVE': self._compute_cive,
            'NGRDI': self._compute_ngrdi,
            'RGBVI': self._compute_rgbvi,
            'GLI': self._compute_gli,
            'VARI': self._compute_vari,
        }
        
        if indices is None:
            self.indices = list(self.available_indices.keys())
        else:
            self.indices = [idx for idx in indices if idx in self.available_indices]
            if len(self.indices) < len(indices):
                missing = set(indices) - set(self.indices)
                print(f"⚠️ 警告: 以下指数不可用: {missing}")
    
    def extract(self, image: np.ndarray, return_stats: bool = True) -> Dict[str, float]:
        """
        从RGB图像提取植被指数
        
        Args:
            image: RGB图像，shape=(H, W, 3), dtype=uint8 或 float [0-255]
            return_stats: 是否返回统计特征（均值、标准差）
            
        Returns:
            特征字典 {index_name: value}
        """
        # 确保图像是浮点数 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image / 255.0
        
        # 分离通道
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
        
        # 归一化颜色
        sum_rgb = R + G + B + 1e-8
        r = R / sum_rgb
        g = G / sum_rgb
        b = B / sum_rgb
        
        features = {}
        
        # 计算每个指数
        for index_name in self.indices:
            index_map = self.available_indices[index_name](R, G, B, r, g, b)
            
            if return_stats:
                # 返回均值和标准差
                features[f'{index_name}_mean'] = float(np.mean(index_map))
                features[f'{index_name}_std'] = float(np.std(index_map))
                features[f'{index_name}_median'] = float(np.median(index_map))
            else:
                # 只返回均值
                features[index_name] = float(np.mean(index_map))
        
        # 添加RGB统计特征
        if return_stats:
            features['R_mean'] = float(np.mean(R))
            features['G_mean'] = float(np.mean(G))
            features['B_mean'] = float(np.mean(B))
            features['R_std'] = float(np.std(R))
            features['G_std'] = float(np.std(G))
            features['B_std'] = float(np.std(B))
        
        return features
    
    def extract_from_path(self, image_path: str, return_stats: bool = True) -> Dict[str, float]:
        """
        从图像路径提取特征
        
        Args:
            image_path: 图像路径
            return_stats: 是否返回统计特征
            
        Returns:
            特征字典
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.extract(image, return_stats)
    
    # ==================== 各种植被指数计算 ====================
    
    @staticmethod
    def _compute_vdvi(R, G, B, r, g, b):
        """可见光差异植被指数 (Visible-band Difference Vegetation Index)"""
        return (2*G - R - B) / (2*G + R + B + 1e-8)
    
    @staticmethod
    def _compute_exg(R, G, B, r, g, b):
        """超绿指数 (Excess Green)"""
        return 2*g - r - b
    
    @staticmethod
    def _compute_exgr(R, G, B, r, g, b):
        """超绿减超红指数 (Excess Green minus Excess Red)"""
        ExG = 2*g - r - b
        ExR = 1.4*r - g
        return ExG - ExR
    
    @staticmethod
    def _compute_cive(R, G, B, r, g, b):
        """色彩植被指数 (Color Index of Vegetation Extraction)"""
        return 0.441*r - 0.811*g + 0.385*b + 18.78
    
    @staticmethod
    def _compute_ngrdi(R, G, B, r, g, b):
        """归一化绿红差异指数 (Normalized Green-Red Difference Index)"""
        return (G - R) / (G + R + 1e-8)
    
    @staticmethod
    def _compute_rgbvi(R, G, B, r, g, b):
        """RGB植被指数 (RGB Vegetation Index)"""
        return (G**2 - B*R) / (G**2 + B*R + 1e-8)
    
    @staticmethod
    def _compute_gli(R, G, B, r, g, b):
        """绿叶指数 (Green Leaf Index)"""
        return (2*G - R - B) / (2*G + R + B + 1e-8)
    
    @staticmethod
    def _compute_vari(R, G, B, r, g, b):
        """可见光大气阻抗植被指数 (Visible Atmospherically Resistant Index)"""
        return (G - R) / (G + R - B + 1e-8)
    
    def get_feature_names(self, return_stats: bool = True) -> List[str]:
        """
        获取特征名称列表
        
        Args:
            return_stats: 是否包含统计特征
            
        Returns:
            特征名称列表
        """
        names = []
        if return_stats:
            for idx in self.indices:
                names.extend([f'{idx}_mean', f'{idx}_std', f'{idx}_median'])
            names.extend(['R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std'])
        else:
            names = self.indices.copy()
        return names
    
    def get_feature_dim(self, return_stats: bool = True) -> int:
        """获取特征维度"""
        return len(self.get_feature_names(return_stats))


def extract_timeseries_features(images: List[np.ndarray], 
                                extractor: VegetationIndicesExtractor) -> np.ndarray:
    """
    从时序图像提取特征
    
    Args:
        images: 时序图像列表，每个元素shape=(H, W, 3)
        extractor: 植被指数提取器
        
    Returns:
        特征矩阵，shape=(T, D)，T为时间步数，D为特征维度
    """
    features_list = []
    for img in images:
        features = extractor.extract(img, return_stats=True)
        # 转为向量
        feat_vector = np.array([features[name] for name in extractor.get_feature_names(True)])
        features_list.append(feat_vector)
    
    return np.array(features_list)


if __name__ == "__main__":
    # 测试代码
    print("测试植被指数提取器...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 初始化提取器
    extractor = VegetationIndicesExtractor()
    
    # 提取特征
    features = extractor.extract(test_image)
    
    print(f"\n特征数量: {len(features)}")
    print(f"特征维度: {extractor.get_feature_dim()}")
    print(f"\n特征名称: {extractor.get_feature_names()}")
    print(f"\n特征示例:")
    for i, (name, value) in enumerate(list(features.items())[:5]):
        print(f"  {name}: {value:.4f}")
    
    print("\n✓ 测试通过！")
