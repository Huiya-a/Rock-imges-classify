# -*- coding: utf-8 -*-
"""
训练器模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from config import Config

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        self.best_weights = model.state_dict().copy()

class FocalLoss(nn.Module):
    """Focal Loss实现"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Trainer:
    """训练器类"""

    def __init__(self, model, train_loader, valid_loader, test_loader, config=Config):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE

        # 将模型移到设备
        self.model.to(self.device)

        # 设置损失函数
        self.criterion = self._get_criterion()

        # 设置优化器
        self.optimizer = self._get_optimizer()

        # 设置学习率调度器
        self.scheduler = self._get_scheduler()

        # 早停机制
        if config.EARLY_STOPPING:
            self.early_stopping = EarlyStopping(
                patience=config.PATIENCE,
                min_delta=config.MIN_DELTA
            )
        else:
            self.early_stopping = None

        # 混合精度训练
        if config.MIXED_PRECISION and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # 训练历史
        self.history = defaultdict(list)

        # 创建保存目录
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

    def _get_criterion(self):
        """获取损失函数"""
        if self.config.LOSS_FUNCTION == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.config.LOSS_FUNCTION == 'focal_loss':
            return FocalLoss()
        elif self.config.LOSS_FUNCTION == 'label_smoothing':
            return LabelSmoothingCrossEntropy(smoothing=self.config.LABEL_SMOOTHING)
        else:
            raise ValueError(f"Unsupported loss function: {self.config.LOSS_FUNCTION}")

    def _get_optimizer(self):
        """获取优化器"""
        if self.config.OPTIMIZER == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                momentum=self.config.MOMENTUM,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER}")

    def _get_scheduler(self):
        """获取学习率调度器"""
        if self.config.LR_SCHEDULER == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.config.LR_STEP_SIZE,
                gamma=self.config.LR_GAMMA
            )
        elif self.config.LR_SCHEDULER == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.EPOCHS
            )
        elif self.config.LR_SCHEDULER == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.LR_GAMMA,
                patience=10
            )
        else:
            return None

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                # 混合精度训练
                with torch.amp.autocast('cuda'):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 常规训练
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        epoch_loss = running_loss / len(self.valid_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def test(self):
        """测试模型"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy())

        test_loss /= len(self.test_loader)
        test_acc = correct / total

        return test_loss, test_acc, predictions, targets

    def train(self):
        """完整的训练过程"""
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"训练轮数: {self.config.EPOCHS}")
        print("-" * 50)

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(self.config.EPOCHS):
            epoch_start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate_epoch()

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - epoch_start_time

            # 打印进度
            print(f'Epoch {epoch+1:3d}/{self.config.EPOCHS} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.6f} | '
                  f'Time: {epoch_time:.2f}s')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f'新的最佳验证准确率: {best_val_acc:.4f}')

            # 早停检查
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, self.model):
                    print(f'早停触发，在第 {epoch+1} 轮停止训练')
                    break

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')

        total_time = time.time() - start_time
        print(f'\n训练完成! 总时间: {total_time:.2f}s')
        print(f'最佳验证准确率: {best_val_acc:.4f}')

        # 测试最佳模型
        self.load_model('best_model.pth')
        test_loss, test_acc, predictions, targets = self.test()
        print(f'测试准确率: {test_acc:.4f}')

        return self.history, test_acc, predictions, targets

    def save_model(self, filename):
        """保存模型"""
        filepath = os.path.join(self.config.MODEL_SAVE_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)

    def load_model(self, filename):
        """加载模型"""
        filepath = os.path.join(self.config.MODEL_SAVE_DIR, filename)
        if os.path.exists(filepath):
            # 修复PyTorch 2.6的weights_only问题
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'模型已从 {filepath} 加载')
        else:
            print(f'模型文件 {filepath} 不存在')

    def plot_training_history(self):
        """绘制训练历史"""
        if not self.history:
            print("没有训练历史可绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 准确率曲线
        axes[0, 1].plot(self.history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 学习率曲线
        axes[1, 0].plot(self.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)

        # 验证准确率详细视图
        axes[1, 1].plot(self.history['val_acc'])
        axes[1, 1].set_title('Validation Accuracy Detail')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if self.config.SAVE_PLOTS:
            plt.savefig(os.path.join(self.config.RESULTS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')

        plt.show()
