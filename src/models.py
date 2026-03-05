# -*- coding: utf-8 -*-
"""
模型定义模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import Config

class ImprovedCNN(nn.Module):
    """改进的CNN模型"""
    def __init__(self, num_classes=9, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第三个卷积块
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
        
        # 第四个卷积块
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
        
        # 自适应平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNetModel(nn.Module):
    """基于ResNet的迁移学习模型"""
    
    def __init__(self, model_name='resnet50', num_classes=9, pretrained=True, dropout_rate=0.5):
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
        
        # 冻结模型的前三层
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        # 替换并解冻最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetModel(nn.Module):
    """基于EfficientNet的迁移学习模型"""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=9, pretrained=True, dropout_rate=0.5):
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
        
        # 解冻最后3个MBConv块
        total_blocks = len(self.backbone.features)
        blocks_to_unfreeze = 3
        for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True
        
        # 替换并解冻分类器
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

class Inception(nn.Module):
    """基于 Inception 的迁移学习模型"""
    def __init__(self, model_name='inception_v3', num_classes=9, pretrained=True, dropout_rate=0.5):
        super(Inception, self).__init__()

        # 加载预训练模型
        self.backbone = models.inception_v3(pretrained=pretrained)
        num_features = self.backbone.fc.in_features

        # 禁用辅助分类器
        self.backbone.aux_logits = False

        # 仅解冻最后两个Inception块 (Mixed_7b 和 Mixed_7c)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.Mixed_7b.parameters():
            param.requires_grad = True
        for param in self.backbone.Mixed_7c.parameters():
            param.requires_grad = True

        # 替换并解冻分类器
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

class VGG(nn.Module):
    """基于 VGG 的迁移学习模型"""
    def __init__(self, model_name='vgg16', num_classes=9, pretrained=True, dropout_rate=0.5):
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
        
        # 只解冻最后两个卷积块
        # 块4（索引17-23）和块5（索引24-30）
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.features[17:].parameters():
            param.requires_grad = True

        # 替换并解冻分类器
        self.backbone.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, num_classes)
        )
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

def get_model(model_type='resnet50', num_classes=9, pretrained=True, dropout_rate=0.5):
    """获取指定类型的模型"""
    
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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
