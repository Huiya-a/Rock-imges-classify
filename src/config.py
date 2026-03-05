# -*- coding: utf-8 -*-
"""
配置文件 - 管理所有超参数和设置
"""

import torch

class Config:
    """配置类，包含所有训练和模型参数"""
    
    # 数据路径
    DATA_ROOT = '../data/rock-data'
    TRAIN_DIR = '../data/rock-data/train'
    TEST_DIR = '../data/rock-data/test'
    VALID_DIR = '../data/rock-data/valid'
    
    # 模型保存路径
    MODEL_SAVE_DIR = 'models'
    RESULTS_DIR = 'results'
    
    # 数据参数
    IMAGE_SIZE = 299
    BATCH_SIZE = 32   # 减小batch size以适应更大的模型
    NUM_WORKERS = 4
    NUM_CLASSES = 9
    
    # 训练参数
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # 学习率调度
    LR_SCHEDULER = 'cosine'  # 'step', 'cosine', 'plateau'
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.1
    
    # 早停参数
    EARLY_STOPPING = True
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # 设备设置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据增强参数
    AUGMENTATION = {
        'rotation': 30,
        'horizontal_flip': 0.5,
        'vertical_flip': 0.3,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1,
        'gaussian_blur': 0.3,
        'random_crop': 0.8,
        'color_jitter': 0.4
    }
    
    # 模型参数
    MODEL_TYPE = 'resnet50'  # 'custom_cnn', 'resnet18', 'resnet50', 'efficientnet'
    PRETRAINED = True
    DROPOUT_RATE = 0.5
    
    # 集成学习参数
    ENSEMBLE = True
    ENSEMBLE_MODELS = ['resnet50', 'inception_v3', 'efficientnet_b0']
    
    # 损失函数
    LOSS_FUNCTION = 'cross_entropy'  # 'cross_entropy', 'focal_loss', 'label_smoothing'
    LABEL_SMOOTHING = 0.1
    
    # 优化器
    OPTIMIZER = 'adamw'  # 'adam', 'adamw', 'sgd'
    MOMENTUM = 0.9  # for SGD
    
    # 混合精度训练
    MIXED_PRECISION = True
    
    # 可视化参数
    PLOT_INTERVAL = 5
    SAVE_PLOTS = True
    
    # 验证参数
    VALIDATION_SPLIT = 0.2  # 如果没有单独的验证集
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("配置信息:")
        print("=" * 50)
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                print(f"{attr}: {getattr(cls, attr)}")
        print("=" * 50)
