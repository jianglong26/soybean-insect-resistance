"""
大豆抗虫数据集类
加载时序图像和表型标签
"""
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
import torch


class SoybeanDataset(Dataset):
    """大豆抗虫数据集"""
    
    def __init__(self,
                 metadata_path: str,
                 image_root: str,
                 environments: List[str] = None,
                 target_labels: List[str] = None,
                 transform=None,
                 use_timeseries: bool = True):
        """
        初始化数据集
        
        Args:
            metadata_path: 元数据JSON文件路径
            image_root: 图像根目录
            environments: 使用的环境列表 ['control', 'nocontrol']
            target_labels: 目标标签列表
            transform: 图像转换
            use_timeseries: 是否使用时序数据
        """
        self.metadata_path = Path(metadata_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.use_timeseries = use_timeseries
        
        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 筛选环境
        if environments:
            self.metadata = {
                k: v for k, v in self.metadata.items()
                if v['environment'] in environments
            }
        
        # 样本键列表
        self.sample_keys = list(self.metadata.keys())
        
        # 目标标签
        self.target_labels = target_labels if target_labels else []
        
        print(f"数据集初始化完成:")
        print(f"  总样本数: {len(self.sample_keys)}")
        print(f"  环境: {set(v['environment'] for v in self.metadata.values())}")
        print(f"  品种数: {len(set(v['genotype'] for v in self.metadata.values()))}")
        if target_labels:
            print(f"  目标标签: {target_labels}")
    
    def __len__(self) -> int:
        return len(self.sample_keys)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        Returns:
            {
                'images': tensor, shape=(T, C, H, W) 或 (C, H, W)
                'labels': dict of tensors
                'genotype': str
                'environment': str
                'plot_number': int
            }
        """
        sample_key = self.sample_keys[idx]
        sample_data = self.metadata[sample_key]
        
        # 加载图像序列
        if self.use_timeseries:
            images = self._load_image_sequence(sample_data)
        else:
            # 只加载最后一个时间点
            images = self._load_single_image(sample_data)
        
        # 加载标签
        labels = self._load_labels(sample_data)
        
        # 返回样本
        return {
            'images': images,
            'labels': labels,
            'genotype': sample_data['genotype'],
            'environment': sample_data['environment'],
            'block': sample_data['block'],
            'sample_key': sample_key
        }
    
    def _load_image_sequence(self, sample_data: Dict) -> torch.Tensor:
        """
        加载时序图像
        
        Returns:
            tensor, shape=(T, C, H, W)
        """
        image_sequence = sample_data['image_sequence']
        images = []
        
        for img_info in image_sequence:
            img_path = self.image_root / img_info['path']
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"无法加载图像: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用转换
            if self.transform:
                image = self.transform(image)
            else:
                # 默认转换：resize到统一大小
                image = cv2.resize(image, (224, 224))  # 统一大小
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            images.append(image)
        
        # 堆叠为时序
        return torch.stack(images)  # (T, C, H, W)
    
    def _load_single_image(self, sample_data: Dict) -> torch.Tensor:
        """
        加载单张图像（最后一个时间点）
        
        Returns:
            tensor, shape=(C, H, W)
        """
        image_sequence = sample_data['image_sequence']
        last_img_info = image_sequence[-1]  # 取最后一个时间点
        
        img_path = self.image_root / last_img_info['path']
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法加载图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换：resize到统一大小
            image = cv2.resize(image, (224, 224))  # 统一大小
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image
    
    def _load_labels(self, sample_data: Dict) -> Dict[str, torch.Tensor]:
        """
        加载标签
        
        Returns:
            标签字典
        """
        labels_dict = sample_data['labels']
        labels = {}
        
        for label_name in self.target_labels:
            value = labels_dict.get(label_name, None)
            if value is None:
                # 缺失值填充为NaN
                labels[label_name] = torch.tensor(float('nan'))
            else:
                labels[label_name] = torch.tensor(float(value))
        
        return labels
    
    def get_genotype_list(self) -> List[str]:
        """获取所有品种列表"""
        return sorted(set(v['genotype'] for v in self.metadata.values()))
    
    def get_environment_list(self) -> List[str]:
        """获取所有环境列表"""
        return sorted(set(v['environment'] for v in self.metadata.values()))
    
    def split_by_genotype(self, test_genotypes: List[str]) -> Tuple['SoybeanDataset', 'SoybeanDataset']:
        """
        按品种划分数据集（Leave-One-Genotype-Out）
        
        Args:
            test_genotypes: 测试集品种列表
            
        Returns:
            (train_dataset, test_dataset)
        """
        train_keys = []
        test_keys = []
        
        for key in self.sample_keys:
            genotype = self.metadata[key]['genotype']
            if genotype in test_genotypes:
                test_keys.append(key)
            else:
                train_keys.append(key)
        
        # 创建子集
        train_dataset = self._create_subset(train_keys)
        test_dataset = self._create_subset(test_keys)
        
        print(f"按品种划分:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_dataset, test_dataset
    
    def _create_subset(self, sample_keys: List[str]) -> 'SoybeanDataset':
        """创建子数据集"""
        subset = SoybeanDataset.__new__(SoybeanDataset)
        subset.metadata_path = self.metadata_path
        subset.image_root = self.image_root
        subset.transform = self.transform
        subset.use_timeseries = self.use_timeseries
        subset.metadata = {k: self.metadata[k] for k in sample_keys}
        subset.sample_keys = sample_keys
        subset.target_labels = self.target_labels
        return subset


def create_dataloaders(config, train_dataset, val_dataset=None, test_dataset=None):
    """
    创建数据加载器
    
    Args:
        config: 配置对象
        train_dataset: 训练集
        val_dataset: 验证集
        test_dataset: 测试集
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    from ..config import Config
    
    Config.create_dirs()
    
    print("测试数据集加载...")
    dataset = SoybeanDataset(
        metadata_path=Config.ANNOTATION_PATH,
        image_root=Config.IMAGE_ROOT,
        environments=['control', 'nocontrol'],
        target_labels=['Bug', 'Tolerance (TOL)', 'Leaf Retention (FR)'],
        use_timeseries=True
    )
    
    print(f"\n获取第一个样本...")
    sample = dataset[0]
    print(f"  图像shape: {sample['images'].shape}")
    print(f"  品种: {sample['genotype']}")
    print(f"  环境: {sample['environment']}")
    print(f"  标签: {sample['labels']}")
    
    print("\n✓ 测试通过！")
