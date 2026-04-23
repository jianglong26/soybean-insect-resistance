"""
DINOv3特征提取器
使用预训练的DINOv3模型提取深度特征
支持两种加载方式：
1. HuggingFace Transformers（推荐）
2. PyTorch Hub（本地权重）
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List
from pathlib import Path
from PIL import Image


class DINOv3FeatureExtractor:
    """DINOv3特征提取器"""
    
    def __init__(self, 
                 model_name: str = "dinov3_vits16",
                 checkpoint_path: str = None,
                 device: str = "cuda",
                 image_size: int = 224):
        """
        初始化DINOv3特征提取器
        
        Args:
            model_name: 模型名称 (dinov3_vits16, dinov3_vitb16, dinov3_vitl16)
            checkpoint_path: 本地权重路径
            device: 设备 (cuda/cpu)
            image_size: 输入图像大小
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        
        print(f"初始化DINOv3特征提取器...")
        print(f"  模型: {model_name}")
        print(f"  设备: {self.device}")
        print(f"  加载方式: PyTorch Hub (本地权重)")
        
        # 加载模型
        self.model = self._load_from_pytorch_hub(checkpoint_path)
        self.processor = None
        
        # 图像预处理
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.model.eval()
        
        # 获取特征维度
        self.feature_dim = self._get_feature_dim()
        print(f"  特征维度: {self.feature_dim}")
    
    def _load_from_pytorch_hub(self, checkpoint_path: str = None):
        """从PyTorch Hub加载模型（使用本地权重）"""
        try:
            # 优先使用提供的checkpoint
            if checkpoint_path and Path(checkpoint_path).exists():
                print(f"  从本地加载权重: {checkpoint_path}")
                return self._load_local_checkpoint(checkpoint_path)
            
            # 尝试默认路径
            default_checkpoint = Path('checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
            if default_checkpoint.exists():
                print(f"  从默认路径加载权重: {default_checkpoint}")
                return self._load_local_checkpoint(str(default_checkpoint))
            
            # 如果没有本地权重，尝试从torch hub下载
            print(f"  本地权重未找到，从torch hub下载模型...")
            model = torch.hub.load('facebookresearch/dinov3', self.model_name, pretrained=True)
            model = model.to(self.device)
            print(f"  ✓ 成功从torch hub加载模型")
            return model
            
        except Exception as e:
            raise ValueError(f"PyTorch Hub加载失败: {e}")
    
    def _load_local_checkpoint(self, checkpoint_path: str):
        """从本地checkpoint加载模型（参考v3kmeans.py的实现）"""
        # 首先从torch hub获取模型架构（不加载权重）
        print(f"  获取模型架构...")
        model = torch.hub.load('facebookresearch/dinov3', self.model_name, pretrained=False)
        
        # 加载本地权重
        print(f"  加载权重文件: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理可能的键名差异
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        
        print(f"  ✓ 成功加载本地模型权重")
        return model
    
    def _get_feature_dim(self):
        """获取特征维度"""
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        with torch.no_grad():
            # 使用DINOv3的forward_features方法
            if hasattr(self.model, 'forward_features'):
                output = self.model.forward_features(dummy_input)
                # 使用CLS token作为全局特征
                features = output['x_norm_clstoken']
            else:
                outputs = self.model(dummy_input)
                if isinstance(outputs, dict):
                    features = outputs.get('pooler_output', outputs.get('last_hidden_state'))
                else:
                    features = outputs
            feature_dim = features.shape[-1]
        
        return feature_dim
    
    @torch.no_grad()
    def extract(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
        """
        提取单张图像的特征
        
        Args:
            image: 输入图像
                - numpy数组: shape=(H, W, 3), RGB格式
                - PIL.Image: RGB格式
                - torch.Tensor: shape=(3, H, W)
                
        Returns:
            特征向量，shape=(feature_dim,)
        """
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            if image.max() > 1.0:
                image = (image).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            from torchvision import transforms as T
            image = T.ToPILImage()(image)
        
        # 提取特征 - 使用transform
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 使用DINOv3的forward_features方法
        if hasattr(self.model, 'forward_features'):
            output = self.model.forward_features(img_tensor)
            # 使用CLS token作为全局特征
            features = output['x_norm_clstoken'].squeeze().cpu().numpy()
        else:
            outputs = self.model(img_tensor)
            # 处理不同输出格式
            if isinstance(outputs, dict):
                features = outputs.get('pooler_output', outputs.get('last_hidden_state'))
            else:
                features = outputs
            
            if isinstance(features, torch.Tensor):
                if len(features.shape) == 3:  # (B, T, D)
                    features = features[:, 0]  # 取CLS token
                features = features.squeeze().cpu().numpy()
        
        return features
    
    @torch.no_grad()
    def extract_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            images: 图像列表
            
        Returns:
            特征矩阵，shape=(N, feature_dim)
        """
        # 转换为PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.max() > 1.0:
                    img = (img).astype(np.uint8)
                img = Image.fromarray(img)
            pil_images.append(img)
        
        if self.use_huggingface:
            # HuggingFace方式：使用processor批量处理
            inputs = self.processor(images=pil_images, return_tensors="pt")
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.device)
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output.cpu().numpy()
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0].cpu().numpy()
            else:
                raise ValueError(f"Unexpected model output format: {type(outputs)}")
        else:
            # PyTorch Hub方式
            img_tensors = [self.transform(img) for img in pil_images]
            batch = torch.stack(img_tensors).to(self.device)
            features = self.model(batch)
            features = features.cpu().numpy()
        
        return features
    
    def extract_from_path(self, image_path: str) -> np.ndarray:
        """
        从图像路径提取特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            特征向量
        """
        image = Image.open(image_path).convert('RGB')
        return self.extract(image)
    
    def extract_timeseries(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        提取时序图像特征
        
        Args:
            images: 时序图像列表，长度为T
            
        Returns:
            时序特征矩阵，shape=(T, feature_dim)
        """
        return self.extract_batch(images)


def create_dinov3_extractor(config=None):
    """
    创建DINOv3提取器的工厂函数
    
    Args:
        config: 配置对象，如果为None则使用默认配置
        
    Returns:
        DINOv3FeatureExtractor实例
    """
    if config is None:
        from ..config import Config
        config = Config
    
    return DINOv3FeatureExtractor(
        model_name=config.DINOV3_MODEL,
        checkpoint_path=str(config.DINOV3_CHECKPOINT) if config.DINOV3_CHECKPOINT.exists() else None,
        device=config.DEVICE,
        image_size=config.DINOV3_IMAGE_SIZE
    )


if __name__ == "__main__":
    # 测试代码
    print("测试DINOv3特征提取器...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    try:
        # 初始化提取器
        extractor = DINOv3FeatureExtractor(
            model_name="dinov3_vits16",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 提取特征
        features = extractor.extract(test_image)
        
        print(f"\n特征维度: {features.shape}")
        print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")
        print(f"特征均值: {features.mean():.4f}")
        
        # 测试批量提取
        batch_images = [test_image] * 4
        batch_features = extractor.extract_batch(batch_images)
        print(f"\n批量特征维度: {batch_features.shape}")
        
        print("\n✓ 测试通过！")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("提示: 需要安装torch和torchvision，并且能访问torch.hub或提供本地权重")
