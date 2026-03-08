# -*- coding: utf-8 -*-
"""
数据加载和预处理模块

功能说明：
    1. 数据集加载：从目录加载训练集、验证集和测试集
    2. 数据增强：应用多种变换增加数据多样性
    3. 数据预处理：标准化、归一化等预处理操作
    4. 批量处理：创建DataLoader进行高效的批量数据加载

使用说明：
    - 训练集使用数据增强以提高模型泛化能力
    - 验证集和测试集不使用数据增强，确保评估的公平性
    - 使用ImageNet标准化均值和标准差，因为使用预训练模型
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from config import Config

class DataManager:
    """
    数据管理类 - 负责数据加载、预处理和增强

    主要功能：
        - 加载训练集、验证集、测试集
        - 应用数据增强策略
        - 创建批量数据加载器
        - 提供数据集信息查询

    数据增强策略：
        - 训练集：应用多种增强（旋转、翻转、颜色调整等）
        - 验证集/测试集：仅进行基础预处理（调整大小、标准化）

    标准化说明：
        - 使用ImageNet的均值和标准差
        - 原因：使用预训练模型时，需要与预训练时使用相同的标准化
        - 均值：[0.485, 0.456, 0.406]
        - 标准差：[0.229, 0.224, 0.225]
    """

    def __init__(self, config=Config):
        """
        初始化数据管理器

        参数：
            config (Config): 配置对象，包含所有数据相关参数

        初始化内容：
            - 训练数据变换（包含数据增强）
            - 测试数据变换（不包含数据增强）
        """
        self.config = config
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    def _get_train_transform(self):
        """
        构建训练数据变换流程

        变换步骤（按顺序）：
            1. Resize: 调整图像到指定尺寸
            2. RandomRotation: 随机旋转（±30度）
            3. RandomHorizontalFlip: 随机水平翻转（50%概率）
            4. RandomVerticalFlip: 随机垂直翻转（30%概率）
            5. ColorJitter: 随机调整亮度、对比度、饱和度、色调
            6. RandomResizedCrop: 随机裁剪和缩放
            7. GaussianBlur (可选): 随机高斯模糊
            8. ToTensor: 转换为PyTorch张量（值域0-1）
            9. Normalize: 标准化（均值0.485,0.456,0.406, 标准差0.229,0.224,0.225）

        返回：
            transforms.Compose: 组合的变换对象
        """
        transform_list = [
            # 1. 调整图像尺寸
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),

            # 2. 几何变换
            transforms.RandomRotation(self.config.AUGMENTATION['rotation']),
            transforms.RandomHorizontalFlip(p=self.config.AUGMENTATION['horizontal_flip']),
            transforms.RandomVerticalFlip(p=self.config.AUGMENTATION['vertical_flip']),

            # 3. 颜色变换
            transforms.ColorJitter(
                brightness=self.config.AUGMENTATION['brightness'],
                contrast=self.config.AUGMENTATION['contrast'],
                saturation=self.config.AUGMENTATION['saturation'],
                hue=self.config.AUGMENTATION['hue']
            ),

            # 4. 随机裁剪
            transforms.RandomResizedCrop(
                self.config.IMAGE_SIZE,
                scale=(self.config.AUGMENTATION['random_crop'], 1.0)
            ),

            # 5. 转换为张量（在标准化之前）
            transforms.ToTensor(),

            # 6. ImageNet标准化（与预训练模型保持一致）
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB通道均值
                std=[0.229, 0.224, 0.225]   # RGB通道标准差
            )
        ]

        # 可选：添加高斯模糊（在ToTensor之前，RandomApply确保概率应用）
        if self.config.AUGMENTATION['gaussian_blur'] > 0:
            transform_list.insert(-2, transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=self.config.AUGMENTATION['gaussian_blur']))

        return transforms.Compose(transform_list)

    def _get_test_transform(self):
        """
        构建测试/验证数据变换流程

        变换步骤（按顺序）：
            1. Resize: 调整图像到指定尺寸
            2. ToTensor: 转换为PyTorch张量
            3. Normalize: 标准化（使用ImageNet均值和标准差）

        注意：
            - 测试数据不使用数据增强
            - 确保评估的一致性和公平性
            - 所有测试样本使用相同的预处理

        返回：
            transforms.Compose: 组合的变换对象
        """
        return transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_data_loaders(self):
        """
        获取训练、验证和测试数据加载器

        数据集处理逻辑：
            1. 检查是否存在独立的验证集目录
            2. 如果存在：直接加载训练集、验证集、测试集
            3. 如果不存在：从训练集划分出验证集（80%训练，20%验证）
            4. 为验证集应用测试变换（不使用数据增强）

        DataLoader参数：
            - batch_size: 批次大小（来自配置）
            - shuffle: 训练集打乱，验证/测试集不打乱
            - num_workers: 数据加载工作进程数
            - pin_memory: GPU训练时启用，加速数据传输

        返回：
            train_loader (DataLoader): 训练数据加载器
            valid_loader (DataLoader): 验证数据加载器
            test_loader (DataLoader): 测试数据加载器
            classes (list): 类别名称列表
        """
        # 检查是否存在单独的验证集
        if os.path.exists(self.config.VALID_DIR):
            # 场景1: 使用独立的验证集
            train_dataset = datasets.ImageFolder(
                root=self.config.TRAIN_DIR,
                transform=self.train_transform
            )
            valid_dataset = datasets.ImageFolder(
                root=self.config.VALID_DIR,
                transform=self.test_transform
            )
            test_dataset = datasets.ImageFolder(
                root=self.config.TEST_DIR,
                transform=self.test_transform
            )
        else:
            # 场景2: 从训练集划分验证集
            full_train_dataset = datasets.ImageFolder(
                root=self.config.TRAIN_DIR,
                transform=self.train_transform
            )

            # 计算划分比例
            total_size = len(full_train_dataset)
            valid_size = int(total_size * self.config.VALIDATION_SPLIT)
            train_size = total_size - valid_size

            # 随机划分数据集（固定随机种子确保可复现）
            train_dataset, valid_dataset = random_split(
                full_train_dataset,
                [train_size, valid_size],
                generator=torch.Generator().manual_seed(42)
            )

            # 为验证集创建新的数据集对象，使用测试变换
            # 这样验证集不会应用数据增强
            valid_dataset.dataset = datasets.ImageFolder(
                root=self.config.TRAIN_DIR,
                transform=self.test_transform
            )

            test_dataset = datasets.ImageFolder(
                root=self.config.TEST_DIR,
                transform=self.test_transform
            )

        # 创建训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,  # 训练集需要打乱
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )

        # 创建验证数据加载器
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,  # 验证集不打乱
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )

        # 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,  # 测试集不打乱
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )

        return train_loader, valid_loader, test_loader, train_dataset.classes

    def get_class_names(self):
        """
        获取数据集的类别名称

        返回：
            list: 类别名称列表（按文件夹名称排序）
        """
        dataset = datasets.ImageFolder(root=self.config.TRAIN_DIR)
        return dataset.classes

    def get_dataset_info(self):
        """
        获取数据集的详细信息

        返回信息：
            - num_classes: 类别数量
            - class_names: 类别名称列表
            - train_size: 训练集样本数
            - test_size: 测试集样本数
            - valid_size: 验证集样本数（如果存在）

        返回：
            dict: 数据集信息字典
        """
        train_dataset = datasets.ImageFolder(root=self.config.TRAIN_DIR)
        test_dataset = datasets.ImageFolder(root=self.config.TEST_DIR)

        info = {
            'num_classes': len(train_dataset.classes),
            'class_names': train_dataset.classes,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset)
        }

        # 如果存在独立验证集，添加验证集信息
        if os.path.exists(self.config.VALID_DIR):
            valid_dataset = datasets.ImageFolder(root=self.config.VALID_DIR)
            info['valid_size'] = len(valid_dataset)

        return info

    def get_optimized_data_loaders(batch_size=32, num_workers=4, mode='full'):
        """
        获取优化的数据加载器（根据不同模式调整数据增强强度）

        模式说明：
            - quick: 快速测试模式，最简单的数据增强
            - fast: 快速训练模式，中等强度的数据增强
            - full: 完整训练模式，强数据增强

        参数：
            batch_size (int): 批次大小
            num_workers (int): 数据加载工作进程数
            mode (str): 训练模式（'quick', 'fast', 'full'）

        返回：
            train_loader, valid_loader, test_loader, classes: 数据加载器和类别列表
        """
        # 根据模式调整数据增强强度
        if mode == 'quick':
            # 快速模式：最简单的增强（仅水平翻转）
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif mode == 'fast':
            # 快速训练：中等增强（翻转+轻微旋转+颜色调整）
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # 完整模式：强数据增强（所有变换）
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # 测试变换（所有模式相同）
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 加载数据集
        train_dataset = datasets.ImageFolder('../data/rock-data/train', transform=train_transform)
        valid_dataset = datasets.ImageFolder('../data/rock-data/valid', transform=test_transform)
        test_dataset = datasets.ImageFolder('../data/rock-data/test', transform=test_transform)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=torch.cuda.is_available())
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=torch.cuda.is_available())

        return train_loader, valid_loader, test_loader, train_dataset.classes
