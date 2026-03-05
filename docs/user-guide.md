# 岩石图像分类系统运行指南

## 项目运行

### 1. 代码环境准备
```bash
# 安装基础依赖
pip install -r requirements.txt

# 推荐：安装GPU版本 (大幅提升训练速度)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 添加数据集

请按照代码结构说明文档中项目目录结构，向 `src\Rock Data` 目录中添加 Rock Data 数据集。

### 3. 验证环境
```bash
cd src
python main.py --mode quick
```
**预期结果**: 5轮训练，约10分钟，准确率~68%

### 4. 快速训练
```bash
python main.py --mode fast
```
**预期结果**: 15轮训练，约25分钟，准确率~71%

### 5. 高精度训练
```bash
python main.py --model resnet50 --epochs 50
```
**预期结果**: 50轮训练，约2小时，准确率~73%

### 6. 集成学习 (最高精度)
```bash
python main.py --ensemble --epochs 30
```
**预期结果**: 训练3个模型，约60分钟，准确率~75%

### 7. 测试模型
```bash
python test.py --model resnet50 --file best_resnet50.pth
python test.py --ensemble
```
**输出结果**：模型的测试准确率和结果图表

## 结果验证

实验中使用集成模型得到的最高正确率为77.01%。相应的单模型文件放置在 `src\test_models` 目录下。在 `src` 目录下运行命令：
```bash
python test.py --ensemble
```
即可对结果进行验证。程序会输出准确率结果和相应的分析图表。

## 详细参数说明

### 基本参数
| 参数 | 默认值 | 说明 | 可选值 |
|------|--------|------|--------|
| `--model` | resnet50 | 模型架构 | resnet18/34/50/101, efficientnet_b0/b1, ... |
| `--epochs` | 30 | 训练轮数 | 1-200 |
| `--batch_size` | 32 | 批次大小 | 8-128 (根据显存调整) |
| `--lr` | 0.001 | 学习率 | 0.0001-0.1 |

### 高级选项
```bash
--ensemble                    # 启用集成学习
--optimizer adamw             # 优化器: adam/adamw/sgd
--scheduler cosine            # 学习率调度: step/cosine/plateau
--mixed_precision             # 混合精度训练 (GPU推荐)
--num_workers 4               # 数据加载进程数
--no_plots                    # 不保存训练图表
```

## 故障排除

### 常见问题及解决方案

#### 问题1: 内存不足 (Out of Memory)
```bash
# 解决方案1: 减小批次大小
python main.py --batch_size 16

# 解决方案2: 使用更小的模型
python main.py --model resnet18

# 解决方案3: 减少工作进程
python main.py --num_workers 2
```

#### 问题2: 训练速度太慢
```bash
# 解决方案1: 使用快速模式
python main.py --mode fast

# 解决方案2: 安装GPU版本PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 解决方案3: 增加工作进程 (如果CPU核心多)
python main.py --num_workers 8
```

#### 问题3: 准确率不理想
```bash
# 解决方案1: 尝试集成学习
python main.py --ensemble

# 解决方案2: 增加训练轮数
python main.py --epochs 100

# 解决方案3: 使用更大的模型
python main.py --model resnet50

# 解决方案4: 调整学习率
python main.py --lr 0.01  # 或 --lr 0.0001
```

#### 问题4: CUDA相关错误
```bash
# 解决方案1: 检查CUDA是否可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 解决方案2: 强制使用CPU
python main.py --no_mixed_precision

# 解决方案3: 重新安装PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 问题5: 数据集路径错误
```bash
# 确保在src目录下运行
cd src
python main.py --mode quick

# 检查数据集结构
ls "Rock Data/train"  # 应该显示9个类别文件夹
```

### 环境检查命令
```bash
# 检查Python环境
python --version

# 检查PyTorch版本
python -c "import torch; print('PyTorch:', torch.__version__)"

# 检查GPU状态
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

# 检查数据集
python -c "import os; print('Train classes:', len(os.listdir('Rock Data/train')))"
```
