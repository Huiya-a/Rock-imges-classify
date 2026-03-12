# -*- coding: utf-8 -*-
"""
模型定义模块

功能说明：
    1. 自定义CNN模型：从零构建的卷积神经网络
    2. 迁移学习模型：基于预训练模型的微调
    3. 模型工厂函数：统一接口创建不同类型的模型

支持的模型架构：
    - 自定义：ImprovedCNN（手工设计的CNN）
    - ResNet系列：ResNet18/34/50/101（残差网络）
    - EfficientNet系列：B0/B1/B2（高效网络）
    - InceptionV3：Inception架构（多尺度特征）
    - VGG系列：VGG11/13/16（经典深度网络）

迁移学习策略：
    - 使用ImageNet预训练权重
    - 冻结大部分层，仅训练最后几层
    - 替换分类器以适应目标任务

性能特点：
    - ResNet：平衡性能和计算成本
    - EfficientNet：高效轻量，适合移动端
    - InceptionV3：多尺度特征，适合复杂图像
    - VGG：参数量大，计算成本高
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import Config

class ImprovedCNN(nn.Module):
    """
    改进的CNN模型（从零训练）

    架构特点：
        - 4个卷积块，逐步增加通道数（64→128→256→512）
        - 每个卷积块包含：卷积层→批归一化→ReLU激活→池化
        - 全连接分类器（4096→4096→9）
        - Dropout防止过拟合

    优势：
        - 轻量级，训练速度快
        - 适合快速实验和原型开发

    劣势：
        - 从零训练，需要大量数据
        - 性能通常低于预训练模型

    使用场景：
        - 快速验证代码逻辑
        - 数据量较小的简单任务
        - 不依赖预训练权重的实验
    """

    def __init__(self, num_classes=9, dropout_rate=0.5):
        """
        初始化改进的CNN模型

        参数：
            num_classes (int): 分类类别数量
            dropout_rate (float): Dropout丢弃率，防止过拟合
        """
        super(ImprovedCNN, self).__init__()

        # ==================== 卷积层特征提取 ====================
        # 卷积块1：64个通道，提取低级特征（边缘、纹理）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 3→64通道
            nn.BatchNorm2d(64),  # 批归一化，加速收敛
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 保持64通道
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样，减小空间尺寸
        )

        # 卷积块2：128个通道，提取中级特征
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 卷积块3：256个通道，提取高级特征（3个卷积层）
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 卷积块4：512个通道，提取最深层特征（3个卷积层）
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ==================== 全局平均池化 ====================
        # 将特征图转换为固定尺寸（7x7），适应不同输入尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # ==================== 分类器 ====================
        # 全连接层：将特征映射到类别
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout，防止过拟合
            nn.Linear(512 * 7 * 7, 4096),  # 25088→4096
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),  # 4096→4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)  # 4096→9（类别数）
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化网络权重

        初始化策略：
            - 卷积层：Kaiming正态初始化（适合ReLU激活）
            - 批归一化：权重=1，偏置=0
            - 全连接层：正态分布（均值=0，标准差=0.01）
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数：
            x (Tensor): 输入图像张量，形状(batch_size, 3, H, W)

        返回：
            Tensor: 分类logits，形状(batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # 展平为向量
        x = self.classifier(x)
        return x

class ResNetModel(nn.Module):
    """
    基于ResNet的迁移学习模型

    架构特点：
        - 残差连接，解决梯度消失问题
        - 深度网络（最多101层）
        - 使用ImageNet预训练权重

    迁移学习策略：
        - 冻结layer1-3（保留ImageNet特征）
        - 微调layer4（适应目标任务）
        - 替换全连接层（9分类）

    性能：
        - ResNet50：最佳平衡点（准确率~73%）
        - ResNet18：轻量级，训练快
        - ResNet101：更高精度，训练慢

    使用场景：
        - 标准迁移学习任务
        - 平衡性能和计算成本
        - 集成学习的基础模型
    """

    def __init__(self, model_name='resnet50', num_classes=9, pretrained=True, dropout_rate=0.5):
        """
        初始化ResNet迁移学习模型

        参数：
            model_name (str): ResNet版本（'resnet18', 'resnet34', 'resnet50', 'resnet101'）
            num_classes (int): 分类类别数量
            pretrained (bool): 是否使用ImageNet预训练权重
            dropout_rate (float): Dropout丢弃率
        """
        super(ResNetModel, self).__init__()

        # 加载预训练模型
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # ==================== 迁移学习策略 ====================
        # 冻结前3个层（保留ImageNet学到的通用特征）
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻layer4（微调高级特征以适应岩石分类）
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # ==================== 替换分类器 ====================
        # 将ImageNet的1000分类替换为岩石的9分类
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),  # 2048→512
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)  # 512→9
        )

        # 确保分类器参数可训练
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

class EfficientNetModel(nn.Module):
    """
    基于EfficientNet的迁移学习模型

    架构特点：
        - 高效的复合缩放策略（深度、宽度、分辨率）
        - MBConv模块（深度可分离卷积）
        - 参数少，性能高

    迁移学习策略：
        - 冻结大部分MBConv块
        - 微调最后3个MBConv块
        - 替换分类器

    性能：
        - EfficientNet-B0：最轻量（准确率~70%）
        - EfficientNet-B1：平衡
        - EfficientNet-B2：更高精度

    使用场景：
        - 资源受限环境
        - 移动端部署
        - 快速训练和推理
    """

    def __init__(self, model_name='efficientnet_b0', num_classes=9, pretrained=True, dropout_rate=0.5):
        """
        初始化EfficientNet迁移学习模型

        参数：
            model_name (str): EfficientNet版本（'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'）
            num_classes (int): 分类类别数量
            pretrained (bool): 是否使用ImageNet预训练权重
            dropout_rate (float): Dropout丢弃率
        """
        super(EfficientNetModel, self).__init__()

        # 加载预训练模型
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
        elif model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # ==================== 迁移学习策略 ====================
        # 解冻最后3个MBConv块（保留大部分ImageNet特征，微调高层特征）
        total_blocks = len(self.backbone.features)
        blocks_to_unfreeze = 3
        for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

        # ==================== 替换分类器 ====================
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # 确保分类器参数可训练
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

class Inception(nn.Module):
    """
    基于InceptionV3的迁移学习模型

    架构特点：
        - Inception模块（多尺度特征提取）
        - 辅助分类器（加速训练）
        - 输入尺寸要求：299x299

    迁移学习策略：
        - 禁用辅助分类器
        - 冻结大部分Inception块
        - 微调最后两个Inception块（Mixed_7b, Mixed_7c）

    性能：
        - 多尺度特征，适合复杂图像
        - 准确率~72%
        - 计算成本较高

    使用场景：
        - 复杂纹理的岩石分类
        - 需要多尺度特征的场景
        - 集成学习的多样性来源
    """

    def __init__(self, model_name='inception_v3', num_classes=9, pretrained=True, dropout_rate=0.5):
        """
        初始化InceptionV3迁移学习模型

        参数：
            model_name (str): 模型名称（目前仅支持'inception_v3'）
            num_classes (int): 分类类别数量
            pretrained (bool): 是否使用ImageNet预训练权重
            dropout_rate (float): Dropout丢弃率
        """
        super(Inception, self).__init__()

        # 加载预训练模型
        self.backbone = models.inception_v3(pretrained=pretrained)
        num_features = self.backbone.fc.in_features

        # 禁用辅助分类器（简化训练过程）
        self.backbone.aux_logits = False

        # ==================== 迁移学习策略 ====================
        # 冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 仅解冻最后两个Inception块（最高级特征）
        for param in self.backbone.Mixed_7b.parameters():
            param.requires_grad = True
        for param in self.backbone.Mixed_7c.parameters():
            param.requires_grad = True

        # ==================== 替换分类器 ====================
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # 确保分类器参数可训练
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

class VGG(nn.Module):
    """
    基于VGG的迁移学习模型

    架构特点：
        - 简单的堆叠卷积结构
        - 大量参数（计算成本高）
        - 深度网络（最多19层）

    迁移学习策略：
        - 冻结前3个卷积块
        - 微调最后2个卷积块
        - 替换分类器

    性能：
        - VGG16：标准性能（准确率~71%）
        - 参数量大，显存占用高
        - 训练速度较慢

    使用场景：
        - 基准对比实验
        - 简单迁移学习示例
        - 不推荐用于生产环境
    """

    def __init__(self, model_name='vgg16', num_classes=9, pretrained=True, dropout_rate=0.5):
        """
        初始化VGG迁移学习模型

        参数：
            model_name (str): VGG版本（'vgg11', 'vgg13', 'vgg16'）
            num_classes (int): 分类类别数量
            pretrained (bool): 是否使用ImageNet预训练权重
            dropout_rate (float): Dropout丢弃率
        """
        super(VGG, self).__init__()

        # 加载预训练模型
        if model_name == 'vgg11':
            self.backbone = models.vgg11(pretrained=pretrained)
            num_features = self.backbone.classifier[6].in_features
        elif model_name == 'vgg13':
            self.backbone = models.vgg13(pretrained=pretrained)
            num_features = self.backbone.classifier[6].in_features
        elif model_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            num_features = self.backbone.classifier[6].in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # ==================== 迁移学习策略 ====================
        # 冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 只解冻最后两个卷积块（块4和块5）
        # VGG features索引：块4为17-23，块5为24-30
        for param in self.backbone.features[17:].parameters():
            param.requires_grad = True

        # ==================== 替换分类器 ====================
        self.backbone.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, num_classes)
        )

        # 确保分类器参数可训练
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

class DenseNetModel(nn.Module):
    """
    基于DenseNet的迁移学习模型

    架构特点：
        - 密集连接：每一层都与前面所有层连接
        - 特征重用：特征可以被多次使用
        - 参数效率：参数量少，性能高

    迁移学习策略：
        - 冻结大部分DenseBlock
        - 微调最后一个DenseBlock
        - 替换分类器

    性能：
        - DenseNet121：轻量级，参数量少，适合小数据集

    使用场景：
        - 小数据集（参数少，不易过拟合）
        - 特征重用需求高的任务
        - 集成学习的基模型
    """

    def __init__(self, model_name='densenet121', num_classes=9, pretrained=True, dropout_rate=0.5):
        """
        初始化DenseNet迁移学习模型

        参数：
            model_name (str): DenseNet版本（'densenet121'）
            num_classes (int): 分类类别数量
            pretrained (bool): 是否使用ImageNet预训练权重
            dropout_rate (float): Dropout丢弃率
        """
        super(DenseNetModel, self).__init__()

        # 加载预训练模型
        if model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # ==================== 迁移学习策略 ====================
        # 冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻最后一个DenseBlock（微调最高级特征）
        for param in self.backbone.features.denseblock4.parameters():
            param.requires_grad = True

        # ==================== 替换分类器 ====================
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # 确保分类器参数可训练
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

class ConvNeXtModel(nn.Module):
    """
    基于ConvNeXt的迁移学习模型

    架构特点：
        - 现代CNN架构，融合Transformer设计理念
        - 大卷积核（7x7）代替传统3x3卷积
        - 层归一化（LayerNorm）代替批归一化
        - 倒瓶颈结构，参数效率高

    迁移学习策略：
        - 冻结大部分Stage
        - 微调最后一个Stage
        - 替换分类器

    性能：
        - ConvNeXt Tiny：高效，准确率~75%
        - ConvNeXt Base：更高精度，准确率~77%

    使用场景：
        - 需要现代化CNN架构
        - 平衡性能和效率
        - 集成学习的多样性来源
    """

    def __init__(self, model_name='convnext_tiny', num_classes=9, pretrained=True, dropout_rate=0.5):
        """
        初始化ConvNeXt迁移学习模型

        参数：
            model_name (str): ConvNeXt版本（'convnext_tiny', 'convnext_base'）
            num_classes (int): 分类类别数量
            pretrained (bool): 是否使用ImageNet预训练权重
            dropout_rate (float): Dropout丢弃率
        """
        super(ConvNeXtModel, self).__init__()

        try:
            import timm
        except ImportError:
            raise ImportError("请先安装timm库: pip install timm")

        # 加载预训练模型
        if model_name == 'convnext_tiny':
            self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained)
            num_features = self.backbone.head.fc.in_features
        elif model_name == 'convnext_base':
            self.backbone = timm.create_model('convnext_base', pretrained=pretrained)
            num_features = self.backbone.head.fc.in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # ==================== 迁移学习策略 ====================
        # 冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻最后一个Stage（微调高级特征）
        for param in self.backbone.stages[-1].parameters():
            param.requires_grad = True

        # ==================== 替换分类器 ====================
        self.backbone.head.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # 确保分类器参数可训练
        for param in self.backbone.head.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

class ViTModel(nn.Module):
    """
    基于Vision Transformer的迁移学习模型

    架构特点：
        - 完全基于注意力机制，没有卷积层
        - 图像分块作为"词"，通过Transformer处理
        - 全局上下文建模能力强
        - 可解释性好（注意力权重可视化）

    迁移学习策略：
        - 冻结大部分Transformer块
        - 微调最后几个Transformer块
        - 替换分类器

    性能：
        - ViT Base：全局建模能力强，准确率~78%
        - ViT Small：轻量级，准确率~76%

    使用场景：
        - 复杂纹理识别
        - 需要全局上下文的任务
        - 可解释性要求高的场景
    """

    def __init__(self, model_name='vit_base_patch16_224', num_classes=9, pretrained=True, dropout_rate=0.5):
        """
        初始化Vision Transformer迁移学习模型

        参数：
            model_name (str): ViT版本（'vit_base_patch16_224', 'vit_small_patch16_224'）
            num_classes (int): 分类类别数量
            pretrained (bool): 是否使用ImageNet预训练权重
            dropout_rate (float): Dropout丢弃率
        """
        super(ViTModel, self).__init__()

        try:
            import timm
        except ImportError:
            raise ImportError("请先安装timm库: pip install timm")

        # 加载预训练模型
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.backbone.head.in_features

        # ==================== 迁移学习策略 ====================
        # 冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻最后几个Transformer块（微调高级特征）
        for param in self.backbone.blocks[-4:].parameters():
            param.requires_grad = True

        # ==================== 替换分类器 ====================
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # 确保分类器参数可训练
        for param in self.backbone.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

def get_model(model_type='resnet50', num_classes=9, pretrained=True, dropout_rate=0.5):
    """
    模型工厂函数 - 根据类型创建模型

    支持的模型类型：
        - 'custom_cnn': 自定义CNN模型
        - 'resnet18/34/50/101': ResNet系列
        - 'efficientnet_b0/b1/b2': EfficientNet系列
        - 'inception_v3': InceptionV3
        - 'vgg11/13/16': VGG系列
        - 'densenet121/161/169': DenseNet系列（新增）
        - 'convnext_tiny/base': ConvNeXt系列（新增）
        - 'vit_base_patch16_224': Vision Transformer（新增）

    参数：
        model_type (str): 模型类型名称
        num_classes (int): 分类类别数量
        pretrained (bool): 是否使用预训练权重
        dropout_rate (float): Dropout丢弃率

    返回：
        nn.Module: 创建的模型实例

    异常：
        ValueError: 不支持的模型类型
    """
    if model_type == 'custom_cnn':
        return ImprovedCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type.startswith('resnet'):
        return ResNetModel(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_type.startswith('efficientnet'):
        return EfficientNetModel(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_type.startswith('inception'):
        return Inception(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_type.startswith('vgg'):
        return VGG(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_type.startswith('densenet'):
        return DenseNetModel(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_type.startswith('convnext'):
        return ConvNeXtModel(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_type.startswith('vit'):
        return ViTModel(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
