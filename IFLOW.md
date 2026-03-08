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

---

# 📝 iFlow 上下文同步记录

## 用户当前状态（2026年3月8日）

### 掌握程度
- ✅ **基本了解**: 了解了项目的整体目标和功能
- ✅ **运行方式**: 理解了quick/fast/full/ensemble四种训练模式
- ✅ **目录结构**: 基本了解了项目的文件组织
- ⚠️ **需要加强**: 对项目的深层架构和具体实现细节还不够熟悉
- ⚠️ **需要加强**: 对训练流程、模型保存、测试流程的具体步骤不够清晰
- ⚠️ **需要加强**: 对迁移学习、集成学习等技术原理的理解需要深化

### 已完成的工作
- 优化了所有代码文件的注释，添加了详细的文档字符串
- 理解了训练和测试的区别和流程
- 掌握了基本的Git操作（add、commit、push）
- 理解了不同运行模式的区别

### 下次工作的重点
1. 深入理解迁移学习策略的具体实现
2. 掌握训练流程的每个步骤和参数含义
3. 理解集成学习的工作原理和实现细节
4. 学习如何调优模型性能（超参数调整）
5. 了解如何添加新的模型或改进现有模型

---

## 项目核心工作流程详解

### 训练流程（单模型）

```
1. 准备阶段
   ├── 加载配置（config.py）
   ├── 创建数据加载器（data_loader.py）
   │   ├── 训练集：应用数据增强
   │   ├── 验证集：仅标准化
   │   └── 测试集：仅标准化
   └── 创建模型（models.py）
       ├── 加载ImageNet预训练权重（网络下载，仅一次）
       └── 替换分类器（1000类 → 9类）

2. 训练阶段（trainer.py）
   ├── 初始化训练组件
   │   ├── 损失函数（CrossEntropyLoss）
   │   ├── 优化器（AdamW）
   │   └── 学习率调度器（Cosine Annealing）
   ├── 训练循环（每个epoch）
   │   ├── 前向传播：计算预测
   │   ├── 计算损失
   │   ├── 反向传播：计算梯度
   │   └── 更新参数
   ├── 验证：在验证集上评估
   │   ├── 计算验证损失和准确率
   │   └── 早停检查
   └── 保存模型
       ├── 保存最佳模型（best_model.pth）
       └── 定期保存检查点（checkpoint_epoch_N.pth）

3. 测试阶段
   ├── 加载最佳模型
   ├── 在测试集上评估
   └── 生成评估报告和可视化
```

### 测试流程

```
1. 加载阶段（test.py）
   ├── 从 test_models/ 加载已训练模型
   ├── 不需要优化器（不训练）
   └── 模型设为评估模式（禁用dropout）

2. 预测阶段
   ├── 前向传播（不计算梯度）
   ├── 获取预测结果
   └── 计算准确率

3. 评估阶段（utils.py）
   ├── 计算评估指标（准确率、精确率、召回率、F1）
   ├── 生成混淆矩阵
   └── 生成可视化图表
```

### 集成学习流程

```
1. 训练多个模型
   ├── ResNet50：训练30轮 → 保存 best_resnet50.pth
   ├── InceptionV3：训练30轮 → 保存 best_inception_v3.pth
   └── EfficientNet-B0：训练30轮 → 保存 best_efficientnet_b0.pth

2. 集成预测
   ├── 加载所有训练好的模型
   ├── 对每个测试样本：
   │   ├── 模型1预测：[0.1, 0.2, 0.7, ...]
   │   ├── 模型2预测：[0.05, 0.15, 0.8, ...]
   │   ├── 模型3预测：[0.15, 0.25, 0.6, ...]
   │   └── 平均概率：[0.1, 0.2, 0.7, ...]
   └── 选择概率最高的类别作为最终预测

3. 结果
   └── 集成准确率：77.01%（高于单模型）
```

---

## 关键概念详解

### 迁移学习
**核心思想**：利用在大规模数据集（ImageNet）上预训练的模型，迁移到小规模数据集（岩石分类）上。

**具体实现**：
1. 加载预训练权重（网络自动下载ImageNet权重）
2. 冻结大部分层（保留通用特征）
3. 微调最后几层（适应目标任务）
4. 替换分类器（1000类 → 9类）

**优势**：
- 减少训练时间（1-2小时 vs 几天）
- 提高准确率（70%+ vs 50%）
- 减少数据需求（4000张 vs 数十万张）

### 数据增强
**目的**：增加数据多样性，防止过拟合

**训练集增强**：
- 旋转（±30度）
- 水平/垂直翻转
- 颜色调整（亮度、对比度、饱和度）
- 随机裁剪
- 高斯模糊

**验证/测试集**：仅标准化，不增强

### 早停机制
**目的**：防止过拟合，提前停止训练

**工作原理**：
1. 监控验证集损失
2. 如果验证损失连续N轮不降低，停止训练
3. 恢复到验证损失最低的模型

### 学习率调度
**目的**：动态调整学习率，帮助模型收敛

**Cosine退火**：
- 初始学习率：0.001
- 每轮按余弦曲线递减
- 最终学习率：接近0

---

## 目录结构详细说明

### models/ 目录
**内容**：训练过程中自动保存的模型文件

**文件说明**：
- `best_model.pth`：当前训练的最佳模型（自动覆盖）
- `best_resnet50.pth`：ResNet50的最佳模型
- `best_inception_v3.pth`：InceptionV3的最佳模型
- `best_efficientnet_b0.pth`：EfficientNet-B0的最佳模型
- `checkpoint_epoch_N.pth`：第N轮的训练检查点

**特点**：
- 训练时自动保存
- 包含模型权重、优化器状态、训练历史
- 大小约200-230MB

### test_models/ 目录
**内容**：手动管理的测试用模型文件

**文件说明**：
- `resnet50.pth`：用于测试的ResNet50模型
- `inception_v3.pth`：用于测试的InceptionV3模型
- `efficientnet_b0.pth`：用于测试的EfficientNet-B0模型

**特点**：
- 需要手动从 models/ 复制（代码不会自动复制）
- 用于最终的测试和验证
- 测试准确率：77.01%

### results/ 目录
**内容**：训练结果和可视化图表

**文件说明**：
- `training_history.png`：训练曲线（损失、准确率）
- `confusion_matrix.png`：混淆矩阵热力图
- `class_performance.png`：各类别性能对比图
- `classification_report.txt`：详细分类报告

---

## 常见疑问解答

### Q1: models/ 和 test_models/ 有什么区别？
**A**:
- `models/`：训练时自动保存，是中间产物
- `test_models/`：手动管理，存放精选的最佳模型，用于最终测试

### Q2: 预训练权重是从哪里来的？
**A**:
- 从PyTorch Hub网络下载（第一次运行时）
- 位置：`C:\Users\<用户名>\.cache\torch\hub\checkpoints\`
- 下载后永久缓存，无需重复下载

### Q3: 为什么训练后还要测试？
**A**:
- 训练时使用训练集和验证集
- 测试时使用独立的测试集
- 确保模型在未见过的数据上也能良好表现

### Q4: 单次运行能训练所有模型吗？
**A**:
- 单模型模式：只能训练1个模型
- 集成模式：可以训练3个模型（依次训练）
- 不能同时训练多个模型

### Q5: 如何选择运行模式？
**A**:
- `quick`：首次运行，验证环境
- `fast`：快速实验，查看效果
- `full`：正式训练，获得最佳单模型
- `ensemble`：追求最高准确率

---

## 下次工作建议

### 短期目标
1. **深入理解代码**：逐行阅读关键文件，理解实现细节
2. **实践训练**：运行一次完整训练，观察输出
3. **调参实验**：尝试修改学习率、批次大小等参数
4. **模型对比**：训练不同模型，对比性能

### 中期目标
1. **添加新模型**：尝试添加新的预训练模型（如DenseNet）
2. **改进数据增强**：尝试新的增强策略
3. **优化超参数**：使用网格搜索或贝叶斯优化
4. **可视化训练**：使用TensorBoard监控训练过程

### 长期目标
1. **提升准确率**：通过优化达到80%以上目标
2. **模型部署**：将模型部署为Web服务
3. **移动端适配**：使用TensorFlow Lite部署到移动设备
4. **论文撰写**：基于项目成果撰写论文

---

## 重要提醒

### 工作习惯
1. **先理解再修改**：修改代码前先理解其作用
2. **小步快跑**：每次只做一个小的改动，及时测试
3. **记录实验**：记录每次实验的参数和结果
4. **版本控制**：定期提交代码，保留历史记录

### 注意事项
1. **路径问题**：所有命令必须在 `src` 目录下运行
2. **数据依赖**：确保数据集位于正确位置
3. **GPU使用**：优先使用GPU训练，大幅提升速度
4. **模型大小**：模型文件约200MB，注意磁盘空间

---

## 快速参考命令

```bash
# 基本操作
cd src                                    # 进入工作目录
git status                                # 查看修改
git add .                                 # 添加所有修改
git commit -m "message"                   # 提交
git push                                  # 推送到远程

# 训练命令
python main.py --mode quick               # 快速测试
python main.py --mode fast                # 快速训练
python main.py --model resnet50 --epochs 30  # 完整训练
python main.py --ensemble --epochs 30     # 集成学习

# 测试命令
python test.py --model resnet50 --file resnet50.pth  # 单模型测试
python test.py --ensemble                # 集成模型测试

# 调试命令
python -c "import torch; print('CUDA:', torch.cuda.is_available())"  # 检查GPU
python -c "import sys; print(sys.path)"  # 检查Python路径
```

---

**最后更新**: 2026年3月8日
**更新人**: iFlow CLI
**目的**: 记录项目理解和工作状态，便于下次快速同步