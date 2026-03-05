# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import os
from config import Config

def evaluate_model(predictions, targets, class_names, save_dir=None):
    """
    评估模型性能
    
    Args:
        predictions: 预测结果
        targets: 真实标签
        class_names: 类别名称
        save_dir: 保存目录
    
    Returns:
        dict: 评估结果
    """
    # 计算准确率
    accuracy = accuracy_score(targets, predictions)
    
    # 计算精确率、召回率、F1分数
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, labels=range(len(class_names))
    )
    
    # 计算宏平均和微平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average='macro'
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average='micro'
    )
    
    # 生成分类报告
    report = classification_report(
        targets, predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # 生成混淆矩阵
    cm = confusion_matrix(targets, predictions)
    
    # 整理结果
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1
        },
        'micro_avg': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1_score': micro_f1
        },
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    # 打印结果
    print("=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"总体准确率: {accuracy:.4f}")
    print(f"宏平均 - 精确率: {macro_precision:.4f}, 召回率: {macro_recall:.4f}, F1分数: {macro_f1:.4f}")
    print(f"微平均 - 精确率: {micro_precision:.4f}, 召回率: {micro_recall:.4f}, F1分数: {micro_f1:.4f}")
    print("\n各类别详细结果:")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:20s} - 精确率: {precision[i]:.4f}, 召回率: {recall[i]:.4f}, "
              f"F1分数: {f1[i]:.4f}, 支持数: {support[i]}")
    
    print("\n分类报告:")
    print(classification_report(targets, predictions, target_names=class_names))
    
    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存分类报告
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write("模型评估结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"总体准确率: {accuracy:.4f}\n")
            f.write(f"宏平均 - 精确率: {macro_precision:.4f}, 召回率: {macro_recall:.4f}, F1分数: {macro_f1:.4f}\n")
            f.write(f"微平均 - 精确率: {micro_precision:.4f}, 召回率: {micro_recall:.4f}, F1分数: {micro_f1:.4f}\n")
            f.write("\n各类别详细结果:\n")
            f.write("-" * 60 + "\n")
            
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name:20s} - 精确率: {precision[i]:.4f}, 召回率: {recall[i]:.4f}, "
                       f"F1分数: {f1[i]:.4f}, 支持数: {support[i]}\n")
            
            f.write("\n分类报告:\n")
            f.write(classification_report(targets, predictions, target_names=class_names))
    
    return results

def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(10, 8)):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建热力图
    sns.heatmap(cm_percent, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title('混淆矩阵 (百分比)', fontsize=16, pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_class_performance(precision, recall, f1, class_names, save_path=None):
    """
    绘制各类别性能对比图
    
    Args:
        precision: 精确率数组
        recall: 召回率数组
        f1: F1分数数组
        class_names: 类别名称
        save_path: 保存路径
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    bars1 = ax.bar(x - width, precision, width, label='精确率', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='召回率', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1分数', alpha=0.8)
    
    ax.set_xlabel('类别')
    ax.set_ylabel('分数')
    ax.set_title('各类别性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def save_predictions(predictions, targets, class_names, save_path):
    """
    保存预测结果
    
    Args:
        predictions: 预测结果
        targets: 真实标签
        class_names: 类别名称
        save_path: 保存路径
    """
    results = []
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        results.append({
            'sample_id': i,
            'true_label': class_names[target],
            'predicted_label': class_names[pred],
            'correct': pred == target
        })
    
    # 保存为CSV格式
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {save_path}")

def calculate_model_size(model):
    """
    计算模型大小
    
    Args:
        model: PyTorch模型
    
    Returns:
        dict: 模型大小信息
    """
    param_size = 0
    param_sum = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    
    buffer_size = 0
    buffer_sum = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    
    all_size = (param_size + buffer_size) / 1024 / 1024  # MB
    
    return {
        'param_count': param_sum,
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_count': buffer_sum,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': all_size
    }

def print_model_info(model, input_size=(3, 224, 224)):
    """
    打印模型信息
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸
    """
    model_info = calculate_model_size(model)
    
    print("=" * 50)
    print("模型信息")
    print("=" * 50)
    print(f"参数数量: {model_info['param_count']:,}")
    print(f"参数大小: {model_info['param_size_mb']:.2f} MB")
    print(f"缓冲区数量: {model_info['buffer_count']:,}")
    print(f"缓冲区大小: {model_info['buffer_size_mb']:.2f} MB")
    print(f"总模型大小: {model_info['total_size_mb']:.2f} MB")
    print("=" * 50)
