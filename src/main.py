# -*- coding: utf-8 -*-
"""
岩石图像分类 - 主训练脚本

功能说明：
    1. 单模型训练：训练单个模型进行岩石图像分类
    2. 集成学习训练：训练多个模型并进行集成预测
    3. 模式选择：提供快速测试、快速训练、完整训练三种模式
    4. 命令行参数：支持灵活的参数配置

训练模式：
    - quick（快速测试）：5轮训练，ResNet18，小批次，约10分钟
    - fast（快速训练）：15轮训练，ResNet18，大批次，约25分钟
    - full（完整训练）：30轮训练，ResNet50，标准批次，约30-60分钟

使用示例:
    # 快速测试（验证代码运行）
    python main.py --mode quick

    # 快速训练（快速迭代实验）
    python main.py --mode fast

    # 完整训练（标准单模型训练）
    python main.py --model resnet50 --epochs 30

    # 集成学习（最高精度）
    python main.py --ensemble --epochs 30

主要参数：
    --model: 模型架构（resnet18/34/50/101, efficientnet_b0/b1/b2, inception_v3, vgg11/13/16）
    --epochs: 训练轮数（默认30）
    --batch_size: 批次大小（默认32）
    --lr: 学习率（默认0.001）
    --ensemble: 启用集成学习
    --mode: 训练模式（quick/fast/full）
    --optimizer: 优化器（adam/adamw/sgd）
    --scheduler: 学习率调度器（step/cosine/plateau）

输出文件：
    - models/: 训练好的模型文件
    - results/: 训练结果图表和报告

注意：
    - 所有命令需要在 src 目录下运行
    - 确保数据集位于 ../data/rock-data 目录
    - 集成学习训练时间较长（约60分钟）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
import time
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置控制台编码为UTF-8以支持emoji显示
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import DataManager
from models import get_model
from trainer import Trainer
from utils import evaluate_model, plot_confusion_matrix, plot_class_performance, print_model_info

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='岩石图像分类训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 快速测试 (5轮训练)
    python main.py --mode quick

    # 快速训练 (15轮训练)
    python main.py --mode fast

    # 完整训练 (30轮训练)
    python main.py --model resnet50 --epochs 30

    # 集成学习
    python main.py --ensemble --epochs 25

    # GPU训练 (如果可用)
    python main.py --model resnet50 --epochs 50 --batch_size 64
        """
    )

    # 模型参数
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                               'efficientnet_b0', 'efficientnet_b1', 'custom_cnn',
                               'efficientnet_b2', 'inception_v3', 'vgg11', 'vgg13', 'vgg16'],
                       help='选择模型架构 (默认: resnet50)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数 (默认: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率 (默认: 0.001)')

    # 训练策略
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='优化器选择 (默认: adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['step', 'cosine', 'plateau'],
                       help='学习率调度器 (默认: cosine)')

    # 模型设置
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='是否使用预训练模型 (默认: True)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率 (默认: 0.5)')

    # 集成学习
    parser.add_argument('--ensemble', action='store_true',
                       help='是否使用集成学习')
    parser.add_argument('--ensemble_models', nargs='+',
                       default=['resnet50', 'inception_v3', 'efficientnet_b0'],
                       help='集成学习使用的模型列表')

    # 训练模式
    parser.add_argument('--mode', type=str, default='full',
                       choices=['quick', 'fast', 'full'],
                       help='训练模式: quick(5轮测试), fast(15轮快速), full(完整训练)')

    # 系统设置
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='是否使用混合精度训练 (默认: True)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数 (默认: 4)')

    # 输出设置
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='是否保存图表 (默认: True)')
    parser.add_argument('--no_plots', action='store_true',
                       help='不保存图表 (覆盖 --save_plots)')

    return parser.parse_args()

def update_config(args):
    """根据命令行参数更新配置"""
    # 基本参数
    Config.MODEL_TYPE = args.model
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr

    # 训练策略
    Config.OPTIMIZER = args.optimizer
    Config.LR_SCHEDULER = args.scheduler

    # 模型设置
    Config.PRETRAINED = args.pretrained
    Config.DROPOUT_RATE = args.dropout

    # 集成学习
    Config.ENSEMBLE = args.ensemble
    # Config.ENSEMBLE_MODELS = args.ensemble_models

    # 系统设置
    Config.MIXED_PRECISION = args.mixed_precision and torch.cuda.is_available()
    Config.NUM_WORKERS = args.num_workers
    Config.SAVE_PLOTS = args.save_plots and not args.no_plots

    # 根据训练模式调整参数
    if args.mode == 'quick':
        Config.EPOCHS = 5
        Config.BATCH_SIZE = 16
        Config.EARLY_STOPPING = False
        Config.MODEL_TYPE = 'resnet18'
        print("🚀 快速测试模式: 5轮训练, ResNet18, 小批次")
    elif args.mode == 'fast':
        Config.EPOCHS = 15
        Config.BATCH_SIZE = 64
        Config.MODEL_TYPE = 'resnet18'
        print("⚡ 快速训练模式: 15轮训练, ResNet18, 大批次")
    else:
        print("🔬 完整训练模式: 自定义参数")

def create_efficient_model(model_type='resnet18', num_classes=9, pretrained=True, dropout_rate=0.5):
    """创建高效的模型（针对快速训练优化）"""
    if model_type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        # 智能冻结策略：只训练最后一层和layer4
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True

        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, num_classes)
        )
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        # 解冻更多层以获得更好性能
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.layer3.parameters():
            param.requires_grad = True

        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, num_classes)
        )
    else:
        # 使用通用模型创建函数
        model = get_model(model_type, num_classes, pretrained, dropout_rate)

    return model

def train_single_model(args):
    """训练单个模型"""
    print("=" * 70)
    print("🚀 岩石图像分类 - 单模型训练")
    print("=" * 70)

    # 检查数据
    if not os.path.exists('../data/rock-data'):
        print("❌ 数据目录不存在，请确保 ../data/rock-data 文件夹存在")
        return None

    # 设备信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    try:
        # 加载数据
        print("\n📊 加载数据...")
        dataManager = DataManager()
        train_loader, valid_loader, test_loader, class_names = dataManager.get_data_loaders()

        print(f"   训练样本: {len(train_loader.dataset):,}")
        print(f"   验证样本: {len(valid_loader.dataset):,}")
        print(f"   测试样本: {len(test_loader.dataset):,}")
        print(f"   类别数量: {len(class_names)}")

        # 创建模型
        print(f"\n🧠 创建模型: {Config.MODEL_TYPE}")
        if args.mode in ['quick', 'fast']:
            model = create_efficient_model(
                model_type=Config.MODEL_TYPE,
                num_classes=len(class_names),
                pretrained=Config.PRETRAINED,
                dropout_rate=Config.DROPOUT_RATE
            )
        else:
            model = get_model(
                model_type=Config.MODEL_TYPE,
                num_classes=len(class_names),
                pretrained=Config.PRETRAINED,
                dropout_rate=Config.DROPOUT_RATE
            )

        # 打印模型信息
        print_model_info(model)

        # 创建训练器
        print(f"\n🏃 开始训练 ({Config.EPOCHS} 轮)...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            config=Config
        )

        # 训练模型
        start_time = time.time()
        history, test_acc, predictions, targets = trainer.train()
        total_time = time.time() - start_time

        print(f"\n⏱️  总训练时间: {total_time/60:.1f} 分钟")

        # 绘制训练历史
        if Config.SAVE_PLOTS:
            trainer.plot_training_history()

        # 评估模型
        print("\n📈 模型评估...")
        results = evaluate_model(
            predictions=predictions,
            targets=targets,
            class_names=class_names,
            save_dir=Config.RESULTS_DIR if Config.SAVE_PLOTS else None
        )

        # 绘制结果
        if Config.SAVE_PLOTS:
            plot_confusion_matrix(
                cm=results['confusion_matrix'],
                class_names=class_names,
                save_path=os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png')
            )

            plot_class_performance(
                precision=results['precision'],
                recall=results['recall'],
                f1=results['f1_score'],
                class_names=class_names,
                save_path=os.path.join(Config.RESULTS_DIR, 'class_performance.png')
            )

        return test_acc, results, trainer

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_ensemble_models(args):
    """训练集成模型"""
    print("=" * 70)
    print("🤝 岩石图像分类 - 集成学习训练")
    print("=" * 70)

    # 检查数据
    if not os.path.exists('../data/rock-data'):
        print("❌ 数据目录不存在，请确保 ../data/rock-data 文件夹存在")
        return None

    # 设备信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

    try:
        # 加载数据
        print("\n📊 加载数据...")
        dataManager = DataManager()
        train_loader, valid_loader, test_loader, class_names = dataManager.get_data_loaders()

        print(f"   训练样本: {len(train_loader.dataset):,}")
        print(f"   验证样本: {len(valid_loader.dataset):,}")
        print(f"   测试样本: {len(test_loader.dataset):,}")
        print(f"   集成模型: {Config.ENSEMBLE_MODELS}")

        # 训练多个模型
        models = {}
        val_accuracies = []

        for i, model_type in enumerate(Config.ENSEMBLE_MODELS):
            print(f"\n🧠 训练模型 {i+1}/{len(Config.ENSEMBLE_MODELS)}: {model_type}")
            print("-" * 50)

            # 创建模型
            if args.mode in ['quick', 'fast']:
                model = create_efficient_model(
                    model_type=model_type,
                    num_classes=len(class_names),
                    pretrained=Config.PRETRAINED,
                    dropout_rate=Config.DROPOUT_RATE
                )
            else:
                model = get_model(
                    model_type=model_type,
                    num_classes=len(class_names),
                    pretrained=Config.PRETRAINED,
                    dropout_rate=Config.DROPOUT_RATE
                )

            # 创建训练器
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                config=Config
            )

            # 训练模型
            history, test_acc, predictions, targets = trainer.train()

            # 保存模型和结果
            models[model_type] = model
            val_accuracies.append(max(history['val_acc']))
            trainer.save_model(f'best_{model_type}.pth')

            print(f"{model_type} 最佳验证准确率: {max(history['val_acc']):.4f}")

        # 集成预测
        print(f"\n🎯 集成预测...")
        ensemble_predictions = []
        all_targets = []

        # 设置所有模型为评估模式
        for model in models.values():
            model.eval()
            model.to(device)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # 收集所有模型的预测概率
                batch_probs = []
                for model in models.values():
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    batch_probs.append(probs.cpu().numpy())

                # 平均概率
                avg_probs = np.mean(batch_probs, axis=0)
                predictions = np.argmax(avg_probs, axis=1)

                ensemble_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())

        # 计算集成准确率
        ensemble_acc = sum(p == t for p, t in zip(ensemble_predictions, all_targets)) / len(all_targets)

        # 评估集成结果
        print("\n📈 集成模型评估...")
        results = evaluate_model(
            predictions=ensemble_predictions,
            targets=all_targets,
            class_names=class_names,
            save_dir=Config.RESULTS_DIR if Config.SAVE_PLOTS else None
        )

        # 显示结果对比
        print("\n📊 模型性能对比:")
        print("-" * 50)
        for model_type, val_acc in zip(Config.ENSEMBLE_MODELS, val_accuracies):
            print(f"{model_type:15s}: {val_acc:.4f}")
        print(f"{'集成模型':15s}: {ensemble_acc:.4f}")
        print(f"{'最佳单模型':15s}: {max(val_accuracies):.4f}")
        print(f"{'集成提升':15s}: {ensemble_acc - max(val_accuracies):+.4f}")

        # 绘制结果
        if Config.SAVE_PLOTS:
            plot_confusion_matrix(
                cm=results['confusion_matrix'],
                class_names=class_names,
                save_path=os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png')
            )

            plot_class_performance(
                precision=results['precision'],
                recall=results['recall'],
                f1=results['f1_score'],
                class_names=class_names,
                save_path=os.path.join(Config.RESULTS_DIR, 'class_performance.png')
            )

        return ensemble_acc, results, models

    except Exception as e:
        print(f"❌ 集成训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("🌟 岩石图像分类系统 v2.0")
    print("=" * 70)

    # 解析命令行参数
    args = parse_arguments()

    # 更新配置
    update_config(args)

    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 显示配置信息
    print(f"\n⚙️  配置信息:")
    print(f"   模型: {Config.MODEL_TYPE}")
    print(f"   训练轮数: {Config.EPOCHS}")
    print(f"   批次大小: {Config.BATCH_SIZE}")
    print(f"   学习率: {Config.LEARNING_RATE}")
    print(f"   优化器: {Config.OPTIMIZER}")
    print(f"   调度器: {Config.LR_SCHEDULER}")
    print(f"   预训练: {Config.PRETRAINED}")
    print(f"   混合精度: {Config.MIXED_PRECISION}")
    print(f"   集成学习: {Config.ENSEMBLE}")

    try:
        if Config.ENSEMBLE:
            # 集成学习训练
            result = train_ensemble_models(args)
            if result is not None:
                ensemble_acc, results, models = result

                print("\n🎉 集成学习训练完成!")
                print(f"🎯 集成模型准确率: {ensemble_acc:.4f}")

                if ensemble_acc >= 0.8:
                    print("🏆 恭喜！达到了80%以上的目标准确率！")
                elif ensemble_acc >= 0.75:
                    print("✅ 达到了75%以上的准确率，非常接近目标！")
                else:
                    print("📈 准确率有待提升，建议尝试更多优化策略")
        else:
            # 单模型训练
            result = train_single_model(args)
            if result is not None:
                test_acc, results, trainer = result

                print("\n🎉 单模型训练完成!")
                print(f"🎯 测试准确率: {test_acc:.4f}")

                if test_acc >= 0.8:
                    print("🏆 恭喜！达到了80%以上的目标准确率！")
                elif test_acc >= 0.7:
                    print("✅ 达到了70%以上的准确率，表现良好！")
                    print("💡 建议尝试集成学习以进一步提升: --ensemble")
                else:
                    print("📈 准确率有待提升，建议:")
                    print("   - 尝试集成学习: --ensemble")
                    print("   - 增加训练轮数: --epochs 50")
                    print("   - 使用更大模型: --model resnet50")

        # GPU建议
        if not torch.cuda.is_available():
            print("\n💡 性能提升建议:")
            print("   当前使用CPU训练。如果有NVIDIA GPU，可以安装CUDA版本:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("   这将大大提升训练速度和模型性能！")

        # 使用建议
        print("\n📚 更多使用方式:")
        print("   快速测试: python main.py --mode quick")
        print("   快速训练: python main.py --mode fast")
        print("   完整训练: python main.py --model resnet50 --epochs 50")
        print("   集成学习: python main.py --ensemble --epochs 30")

    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()