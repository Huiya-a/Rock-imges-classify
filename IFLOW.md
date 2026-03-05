# 岩石图像分类项目 - iFlow 上下文文件

## 项目概述

这是一个基于深度学习的岩石图像分类系统，使用卷积神经网络（CNN）对9种不同类型的岩石进行自动分类。项目采用迁移学习、数据增强和集成学习等技术，当前最高准确率达到77%。

### 核心技术栈
- **深度学习框架**: PyTorch 2.0+
- **计算机视觉**: torchvision
- **数据处理**: numpy, pandas, Pillow
- **可视化**: matplotlib, seaborn
- **评估工具**: scikit-learn

### 项目目标
- 实现岩石图像的自动分类，准确率达到80%以上
- 当前成果：准确率已达到77%，接近目标

## 项目结构

```
Rock-imges-classify-with-CNN/
├── requirements.txt              # 项目依赖包列表
├── README.md                     # 项目概述和介绍
├── project-report.pdf            # 项目最终报告
├── docs/                         # 项目文档目录
│   ├── user-guide.md             # 详细使用说明
│   ├── code-structure.md         # 代码架构介绍
│   └── results/                  # 分类报告和结果图表
├── data/                         # 数据集目录
│   └── rock-data/                # 岩石图像数据集
│       ├── train/                # 训练集 (3,687张图片)
│       ├── valid/                # 验证集 (351张图片)
│       └── test/                 # 测试集 (174张图片)
├── src/                          # 源代码目录（主要工作目录）
│   ├── main.py                   # 主训练脚本
│   ├── test.py                   # 测试已训练模型脚本
│   ├── config.py                 # 配置管理模块
│   ├── data_loader.py            # 数据加载和预处理
│   ├── models.py                 # 模型定义模块
│   ├── trainer.py                # 训练器模块
│   ├── utils.py                  # 工具函数模块
│   ├── models/                   # 模型保存目录
│   ├── results/                  # 训练结果目录
│   └── test_models/              # 待测试模型目录（包含3个预训练模型）
│       ├── efficientnet_b0.pth
│       ├── inception_v3.pth
│       └── resnet50.pth
└── outputs/                      # 输出文件集中管理目录
```

## 数据集信息

### 岩石类别（9类）
1. **Basalt** (玄武岩) - 火成岩
2. **Clay** (粘土) - 沉积岩
3. **Conglomerate** (砾岩) - 沉积岩
4. **Diatomite** (硅藻土) - 沉积岩
5. **Shale-(Mudstone)** (页岩-泥岩) - 沉积岩
6. **Siliceous-sinter** (硅质烧结岩) - 化学沉积岩
7. **chert** (燧石) - 沉积岩
8. **gypsum** (石膏) - 蒸发岩
9. **olivine-basalt** (橄榄石玄武岩) - 火成岩

### 数据分布
- **训练集**: 3,687张图片
- **验证集**: 351张图片
- **测试集**: 174张图片
- **总计**: 4,212张图片
- **图像格式**: JPG

## 构建和运行

### 环境准备

```bash
# 安装基础依赖
pip install -r requirements.txt

# 推荐：安装GPU版本（大幅提升训练速度）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 快速开始

**重要**: 所有命令都需要在 `src` 目录下运行

```bash
cd src

# 快速测试（5轮训练，约10分钟）
python main.py --mode quick

# 快速训练（15轮训练，约25分钟）
python main.py --mode fast

# 完整训练（30轮训练）
python main.py --model resnet50 --epochs 30

# 集成学习（最高精度，约60分钟）
python main.py --ensemble --epochs 30
```

### 测试预训练模型

```bash
cd src

# 测试单个模型
python test.py --model resnet50 --file resnet50.pth

# 测试集成模型（使用test_models目录下的3个预训练模型）
python test.py --ensemble
```

**预期结果**: 集成模型的测试准确率为77.01%

### 主要命令参数

| 参数 | 默认值 | 说明 | 可选值 |
|------|--------|------|--------|
| `--model` | resnet50 | 模型架构 | resnet18/34/50/101, efficientnet_b0/b1/b2, inception_v3, vgg11/13/16 |
| `--epochs` | 30 | 训练轮数 | 1-200 |
| `--batch_size` | 32 | 批次大小 | 8-128（根据显存调整） |
| `--lr` | 0.001 | 学习率 | 0.0001-0.1 |
| `--ensemble` | False | 启用集成学习 | - |
| `--mode` | full | 训练模式 | quick(5轮), fast(15轮), full(完整) |
| `--optimizer` | adamw | 优化器 | adam/adamw/sgd |
| `--scheduler` | cosine | 学习率调度 | step/cosine/plateau |

## 核心模块说明

### 1. main.py - 主训练脚本
- 功能：项目的核心入口，整合所有训练功能
- 特点：支持命令行参数配置、多种训练模式、单模型和集成学习
- 关键函数：
  - `parse_arguments()`: 解析命令行参数
  - `train_single_model()`: 单模型训练
  - `train_ensemble_models()`: 集成学习训练

### 2. config.py - 配置管理
- 功能：统一管理所有超参数和系统配置
- 主要配置：
  - 数据配置：IMAGE_SIZE=224, BATCH_SIZE=32, NUM_CLASSES=9
  - 训练配置：EPOCHS=100, LEARNING_RATE=0.001
  - 模型配置：MODEL_TYPE='resnet50', PRETRAINED=True
  - 集成学习：ENSEMBLE_MODELS=['resnet50', 'inception_v3', 'efficientnet_b0']

### 3. models.py - 模型定义
- 支持的模型：
  - 自定义CNN (ImprovedCNN)
  - ResNet系列 (ResNet18/34/50/101)
  - EfficientNet系列 (B0/B1/B2)
  - InceptionV3
  - VGG系列 (VGG11/13/16)
- 关键函数：`get_model(model_type, num_classes, pretrained, dropout_rate)`

### 4. trainer.py - 训练器
- 功能：管理完整的训练流程
- 特性：
  - 混合精度训练（GPU加速）
  - 学习率调度（余弦退火、阶梯衰减）
  - 早停机制（防止过拟合）
  - 自动保存最佳模型
- 关键类：`Trainer`

### 5. data_loader.py - 数据处理
- 功能：数据加载、预处理和增强
- 特性：智能数据增强、高效批量处理
- 关键类：`DataManager`

### 6. test.py - 模型测试
- 功能：加载已训练模型进行测试
- 特点：支持单模型和集成模型测试
- 使用：`python test.py --ensemble` 测试预训练的集成模型

### 7. utils.py - 工具函数
- 功能：评估、可视化和辅助功能
- 主要函数：
  - `evaluate_model()`: 模型性能评估
  - `plot_confusion_matrix()`: 绘制混淆矩阵
  - `plot_class_performance()`: 绘制类别性能图

## 开发规范

### 代码风格
- 使用UTF-8编码
- 遵循PEP 8规范
- 函数和类使用文档字符串说明
- 关键参数有详细的注释

### 模型保存
- 模型保存在 `src/models/` 目录
- 使用PyTorch的checkpoint格式保存：`model_state_dict`, `optimizer_state_dict`, `config`, `history`
- 最佳模型自动保存为 `best_model.pth`

### 结果输出
- 训练结果保存在 `src/results/` 目录
- 包含图表：训练曲线、混淆矩阵、类别性能图
- 分类报告：`classification_report.txt`

### 配置管理
- 所有配置集中在 `config.py` 的 `Config` 类中
- 通过命令行参数可以覆盖默认配置
- 推荐使用命令行参数进行实验，而不是直接修改config.py

## 常见问题

### 1. 内存不足
```bash
# 减小批次大小
python main.py --batch_size 16

# 使用更小的模型
python main.py --model resnet18
```

### 2. 训练速度慢
```bash
# 使用快速模式
python main.py --mode fast

# 安装GPU版本PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 准确率不理想
```bash
# 尝试集成学习
python main.py --ensemble

# 增加训练轮数
python main.py --epochs 100

# 使用更大的模型
python main.py --model resnet50
```

### 4. CUDA相关错误
```bash
# 检查CUDA是否可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 强制使用CPU
python main.py --no_mixed_precision
```

## 项目成果

### 技术成果
- 最高准确率达到77%
- 完整的端到端解决方案
- 模块化、可扩展的代码架构
- 支持多种预训练模型和集成学习

### 性能指标
- 单模型最佳准确率：~73%
- 集成模型准确率：77.01%
- 训练时间：集成模型约60分钟（GPU）
- 支持的模型架构：10+种

## 扩展建议

### 添加新模型
1. 在 `models.py` 中定义新模型类
2. 在 `get_model()` 函数中注册新模型
3. 在 `main.py` 的参数选项中添加新模型

### 添加新的数据增强
1. 在 `data_loader.py` 中修改变换流程
2. 在 `config.py` 中添加相关参数
3. 测试新增强的效果

### 添加新的训练策略
1. 在 `trainer.py` 中实现新的训练逻辑
2. 在 `config.py` 中添加相关配置
3. 在 `main.py` 中集成新策略

## 重要提示

1. **工作目录**: 所有命令必须在 `src` 目录下运行
2. **数据集**: 确保 `Rock Data` 文件夹在 `src` 目录下
3. **GPU训练**: 推荐使用GPU训练，速度可提升10-20倍
4. **模型文件**: 预训练模型位于 `src/test_models/` 目录
5. **结果验证**: 使用 `python test.py --ensemble` 验证集成模型结果

## 联系和参考

- 项目GitHub: https://github.com/Patience-Pei/Rock-imges-classify-with-CNN.git
- 详细文档：查看 `docs/` 目录下的文档
- 项目报告：`项目报告.pdf`