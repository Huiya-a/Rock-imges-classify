"""
直接从 test_models 目录下加载训练好的模型文件进行测试的代码文件

注意参数中的文件名称为 test_models 目录下的名称，使用前请注意检查该目录下是否存在对应的文件

测试集成模型时，按实验中测试出的测试准确最高的模型组合进行测试，不支持自定义模型组合

注意：输入的模型架构必须与模型文件相匹配！否则程序在测试过程中会报错！

使用示例:
    # 单模型测试
    python test.py --model resnet50 --file resnet50.pth

    # 集成模型测试
    python test.py --ensemble   # --ensemble 参数会覆盖其他参数
"""

import torch
import argparse
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from config import Config
from data_loader import DataManager
from models import get_model
from trainer import Trainer
from utils import evaluate_model, plot_confusion_matrix, plot_class_performance, print_model_info

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='测试预训练模型脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 单模型测试
    python test.py --model resnet50 --file resnet50.pth
    # 集成模型测试
    python test.py --ensemble
"""
    )

    # 模型架构
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                               'efficientnet_b0', 'efficientnet_b1', 'custom_cnn',
                               'efficientnet_b2', 'inception_v3', 'vgg11', 'vgg13', 'vgg16'],
                        help='选择模型架构')
    # 模型文件名称
    parser.add_argument('--file', type=str, default='resnet50.pth',
                        help='test_models 目录下的模型文件名称')
    # 集成模型测试
    parser.add_argument('--ensemble', action='store_true',
                        help='是否评估集成模型，启用该参数后会覆盖其他参数')
    
    return parser.parse_args()

def main():
    """主函数"""
    print("🌟 岩石图像分类系统 v2.0")
    print("=" * 70)

    # 解析命令行参数
    args = parse_arguments()
    model_type = args.model
    file = args.file
    ensemble = args.ensemble

    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 显示配置信息
    print(f"\n⚙️  配置信息:")
    print(f"   模型: {model_type}")
    print(f"   模型文件名称：{file}")

    # 检查文件路径
    if not os.path.exists(os.path.join('test_models', file)):
        print("❌ 模型文件不存在，请确保输入了正确的名称")
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

        print(f"   测试样本: {len(test_loader.dataset):,}")
        print(f"   类别数量: {len(class_names)}")

        if ensemble:
            models = {}
            for i, model_type in enumerate(Config.ENSEMBLE_MODELS):
                # 创建模型
                model = get_model(
                    model_type=model_type,
                    num_classes=len(class_names),
                    pretrained=Config.PRETRAINED,
                    dropout_rate=Config.DROPOUT_RATE
                )

                filename = model_type + '.pth'
                filepath = os.path.join('test_models', filename)
                if os.path.exists(filepath):
                    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f'模型已从 {filepath} 加载')
                else:
                    print(f'模型文件 {filepath} 不存在')
                    return None

                models[model_type] = model
                
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
            print(f'测试准确率: {ensemble_acc:.4f}')

        else:
            # 创建模型
            model = get_model(
                model_type=Config.MODEL_TYPE,
                num_classes=len(class_names),
                pretrained=Config.PRETRAINED,
                dropout_rate=Config.DROPOUT_RATE
            )

            filepath = os.path.join('test_models', file)
            if os.path.exists(filepath):
                checkpoint = torch.load(filepath, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f'模型已从 {filepath} 加载')
            else:
                print(f'模型文件 {filepath} 不存在')
                return None

            # 打印模型信息
            print_model_info(model)

            # 创建训练器
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                config=Config
            )

            # 评估模型
            print("\n📈 模型评估...")
            model.eval()
            test_loss, test_acc, predictions, targets = trainer.test()
            results = evaluate_model(
                predictions=predictions,
                targets=targets,
                class_names=class_names,
                save_dir=Config.RESULTS_DIR if Config.SAVE_PLOTS else None
            )
            print(f'测试准确率: {test_acc:.4f}')

        # 绘制结果
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

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
if __name__ == '__main__':
    main()