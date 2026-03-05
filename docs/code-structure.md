# 岩石图像分类项目代码结构说明

## 项目目录结构

```
Rock-images-classify-with-CNN/
├── requirements.txt             # 项目依赖包列表
├── README.md                    # 项目概述和介绍
├── 项目报告.pdf                 # 项目最终报告
├── doc/                         # 项目文档目录
│   ├── 使用指南.md              # 详细使用说明
│   ├── 代码结构说明.md          # 代码架构介绍(本文件)
│   └── result/                  # 分类报告和结果图表
├── src/                         # 源代码目录
│   ├── main.py                  # 主训练脚本
|   ├── test.py                  # 测试已训练模型脚本
│   ├── config.py                # 配置管理模块
│   ├── data_loader.py           # 数据加载和预处理
│   ├── models.py                # 模型定义模块
│   ├── trainer.py               # 训练器模块
│   ├── ensemble.py              # 集成学习模块
│   ├── utils.py                 # 工具函数模块
│   ├── Rock Data/               # 岩石图像数据集
│   │   ├── train/               # 训练集 (3,687张图片)
│   │   ├── valid/               # 验证集 (351张图片)
│   │   └── test/                # 测试集 (174张图片)
│   ├── models/                  # 模型保存目录
│   ├── results/                 # 训练结果目录
│   └── test_models/             # 待测试模型目录
```

## 核心文件详解

### 1. 主训练脚本 (`main.py`)

**功能**: 项目的核心入口，整合了所有训练功能
**特点**: 
- 一个脚本解决所有训练需求
- 支持命令行参数配置
- 多种训练模式 (quick/fast/full)
- 单模型和集成学习支持

**主要功能**:
```python
def parse_arguments()          # 解析命令行参数
def update_config()            # 更新配置参数
def create_efficient_model()   # 创建优化模型
def get_optimized_data_loaders() # 获取数据加载器
def train_single_model()       # 单模型训练
def train_ensemble_models()    # 集成学习训练
def main()                     # 主函数入口
```

**使用示例**:
```bash
python main.py --mode fast              # 快速训练
python main.py --ensemble --epochs 30   # 集成学习
python main.py --model resnet50 --epochs 50  # 自定义训练
```

### 2. 配置管理 (`config.py`)

**功能**: 统一管理所有超参数和系统配置
**特点**: 
- 集中式配置管理
- 详细的参数说明
- 易于修改和扩展

**主要配置类别**:
```python
class Config:
    # 数据配置
    IMAGE_SIZE = 224          # 图像尺寸
    BATCH_SIZE = 32           # 批次大小
    NUM_CLASSES = 9           # 类别数量
    
    # 训练配置
    EPOCHS = 100              # 训练轮数
    LEARNING_RATE = 0.001     # 学习率
    OPTIMIZER = 'adamw'       # 优化器
    
    # 模型配置
    MODEL_TYPE = 'resnet50'   # 模型类型
    PRETRAINED = True         # 预训练权重
    DROPOUT_RATE = 0.5        # Dropout率
    
    # 系统配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = True    # 混合精度训练
```

### 3. 数据处理 (`data_loader.py`)

**功能**: 负责数据加载、预处理和增强
**特点**: 
- 智能数据增强策略
- 支持多种数据集结构
- 高效的批量处理

**核心类**:
```python
class DataManager:
    def __init__(self, config)              # 初始化配置
    def _get_train_transform(self)          # 训练数据变换
    def _get_test_transform(self)           # 测试数据变换
    def get_data_loaders(self)              # 获取数据加载器
    def get_class_names(self)               # 获取类别名称
    def get_dataset_info(self)              # 获取数据集信息
```

### 4. 模型定义 (`models.py`)

**功能**: 定义和创建各种深度学习模型
**特点**: 
- 支持多种架构
- 模块化设计，易于扩展
- 针对岩石分类优化

**支持的模型**:
1. 自定义 CNN
2. ResNet 系列预训练模型
3. EfficientNet 系列预训练模型
4. Inception_V3 预训练模型 
5. VGG 系列预训练模型

### 5. 训练器 (`trainer.py`)

**功能**: 管理完整的训练流程
**特点**: 
- 完整的训练循环
- 实时性能监控
- 自动模型保存
- 智能早停机制

**核心功能**:
```python
class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, config)
    def train_epoch(self)              # 训练一个epoch
    def validate_epoch(self)           # 验证一个epoch
    def test(self)                     # 测试模型
    def train(self)                    # 完整训练流程
    def save_model(self, filename)     # 保存模型
    def load_model(self, filename)     # 加载模型
    def plot_training_history(self)    # 绘制训练历史
```

**训练特性**:
- **混合精度训练**: 加速训练，节省显存
- **学习率调度**: 余弦退火、阶梯衰减等
- **早停机制**: 防止过拟合
- **自动保存**: 保存最佳模型权重

### 6. 模型测试 (`test,py`)
**功能**：加载已训练的模型文件进行测试，验证实验结果
**特点**: 
- 测试模型方便快捷
- 可单独测试指定模型，也可测试集成模型

### 7. 工具函数 (`utils.py`)

**功能**: 提供评估、可视化和辅助功能
**特点**: 
- 丰富的评估指标
- 专业的可视化图表
- 实用的辅助函数

**主要功能**:
```python
def evaluate_model()           # 模型性能评估
def plot_confusion_matrix()    # 绘制混淆矩阵
def plot_class_performance()   # 绘制类别性能图
def save_predictions()         # 保存预测结果
def calculate_model_size()     # 计算模型大小
def print_model_info()         # 打印模型信息
```

## 扩展指南

### 添加新模型
1. 在`models.py`中定义新模型类
2. 在`get_model()`函数中注册新模型
3. 在`main.py`的参数选项中添加新模型

### 添加新的数据增强
1. 在`data_loader.py`中修改变换流程
2. 在`config.py`中添加相关参数
3. 测试新增强的效果

### 添加新的训练策略
1. 在`trainer.py`中实现新的训练逻辑
2. 在`config.py`中添加相关配置
3. 在`main.py`中集成新策略

### 添加新的评估指标
1. 在`utils.py`中实现新的评估函数
2. 在`trainer.py`中调用新的评估
3. 在可视化中展示新指标

## 代码统计

### 文件规模
| 文件 | 行数 | 主要功能 |
|------|------|----------|
| main.py | ~540 | 统一入口，训练控制 |
| trainer.py | ~420 | 训练流程管理 |
| test.py   | ~220  | 加载模型测试 |
| models.py | ~300 | 模型定义 |
| data_loader.py | ~200 | 数据处理 |
| utils.py | ~270 | 工具函数 |
| config.py | ~100 | 配置管理 |
