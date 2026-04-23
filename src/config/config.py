"""
配置文件：统一管理所有超参数和路径
"""
import os
from pathlib import Path


class Config:
    """配置类"""
    
    # ==================== 路径配置 ====================
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # 数据路径（支持通过环境变量外置大数据目录）
    DATA_ROOT = Path(os.getenv("SOY_DATA_ROOT", str(PROJECT_ROOT / "AnhumasPiracicaba")))
    DATASET_ROOT = DATA_ROOT / "dataset"
    IMAGE_ROOT = DATASET_ROOT / "images"
    ANNOTATION_PATH = DATASET_ROOT / "annotations" / "dataset_metadata.json"
    
    # 输出路径（可重定向到外部磁盘）
    OUTPUT_ROOT = Path(os.getenv("SOY_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs")))
    FEATURE_ROOT = OUTPUT_ROOT / "features"
    MODEL_ROOT = OUTPUT_ROOT / "models"
    RESULT_ROOT = OUTPUT_ROOT / "results"
    
    # 预训练模型权重（支持外部路径）
    CHECKPOINT_ROOT = Path(os.getenv("SOY_CHECKPOINT_ROOT", str(PROJECT_ROOT / "checkpoints")))
    DINOV3_CHECKPOINT = CHECKPOINT_ROOT / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    
    # ==================== DINOv3配置 ====================
    DINOV3_MODEL = "dinov3_vits16"  # vits16, vitb16, vitl16, vitg14
    DINOV3_PATCH_SIZE = 16
    DINOV3_EMBED_DIM = 384  # vits16: 384, vitb16: 768, vitl16: 1024
    DINOV3_IMAGE_SIZE = 224  # 输入图像大小
    DINOV3_BATCH_SIZE = 32
    
    # ==================== 植被指数配置 ====================
    VEGETATION_INDICES = [
        'VDVI',   # 可见光差异植被指数
        'ExG',    # 超绿指数
        'ExGR',   # 超绿减超红
        'CIVE',   # 色彩植被指数
        'NGRDI',  # 归一化绿红差异指数
        'RGBVI',  # RGB植被指数
    ]
    
    # ==================== 数据集配置 ====================
    ENVIRONMENTS = ['control', 'nocontrol']
    NUM_GENOTYPES = 30
    NUM_BLOCKS = 3
    NUM_TIMEPOINTS = 6
    
    # 标签列名
    LABEL_COLUMNS = [
        'Bug', 'Specie', 'Nymph', 'Worm', 'Coleoptera', 'Other',  # 虫害
        'Tolerance (TOL)', 'Leaf Retention (FR)', 'Agronomic Value (VA)',  # 抗性
        'Grain Yield - GY (kg/ha)', 'Healthy Seed Weight (HSW)', 'Filling Period (PEG)'  # 产量
    ]
    
    # 回归任务目标
    REGRESSION_TARGETS = [
        'Bug', 'Nymph', 'Worm', 'Coleoptera',
        'Tolerance (TOL)', 'Leaf Retention (FR)',
        'Grain Yield - GY (kg/ha)', 'Healthy Seed Weight (HSW)'
    ]
    
    # ==================== 训练配置 ====================
    # 数据划分
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # 训练超参数
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    EARLY_STOP_PATIENCE = 15
    
    # 优化器
    OPTIMIZER = "AdamW"
    WEIGHT_DECAY = 1e-4
    
    # 学习率调度
    LR_SCHEDULER = "CosineAnnealingLR"
    LR_MIN = 1e-6
    
    # 正则化
    DROPOUT = 0.3
    LABEL_SMOOTHING = 0.1
    
    # ==================== 模型配置 ====================
    # 融合模型
    FUSION_HIDDEN_DIM = 256
    FUSION_NUM_LAYERS = 2
    
    # 时序模型
    TEMPORAL_MODEL = "transformer"  # transformer, lstm, gru
    TEMPORAL_HIDDEN_DIM = 256
    TEMPORAL_NUM_HEADS = 8
    TEMPORAL_NUM_LAYERS = 3
    
    # ==================== 实验配置 ====================
    # 实验模式
    EXPERIMENT_MODE = "multi_task"  # single_task, multi_task
    
    # 消融实验
    USE_DINOV3 = True
    USE_VEG_INDICES = True
    USE_TEMPORAL = True
    
    # ==================== 设备配置 ====================
    DEVICE = os.getenv("SOY_DEVICE", "cuda")  # cuda, cpu
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        for dir_path in [
            cls.OUTPUT_ROOT,
            cls.FEATURE_ROOT,
            cls.MODEL_ROOT,
            cls.RESULT_ROOT,
            cls.FEATURE_ROOT / "dinov3",
            cls.FEATURE_ROOT / "vegetation_indices",
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_feature_path(cls, feature_type, environment):
        """获取特征保存路径"""
        return cls.FEATURE_ROOT / feature_type / f"{environment}_features.pkl"
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("="*60)
        print("配置信息")
        print("="*60)
        print(f"数据根目录: {cls.DATA_ROOT}")
        print(f"图像根目录: {cls.IMAGE_ROOT}")
        print(f"标注文件: {cls.ANNOTATION_PATH}")
        print(f"输出目录: {cls.OUTPUT_ROOT}")
        print(f"DINOv3模型: {cls.DINOV3_MODEL}")
        print(f"设备: {cls.DEVICE}")
        print("="*60)


if __name__ == "__main__":
    Config.create_dirs()
    Config.print_config()
