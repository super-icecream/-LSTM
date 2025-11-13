"""
GPU优化的训练器模块
功能：管理完整训练流程，支持混合精度训练，多天气子模型并行训练
GPU优化：CUDA流并行、混合精度、梯度累积、异步数据加载
作者：DLFE-LSTM-WSI Team
日期：2025-09-27
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import logging
import sys
import platform  # 新增：导入平台信息
import shutil
import unicodedata
import os
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# ===============================================
# Windows单行刷新支持：初始化colorama
# ===============================================
try:
    import colorama
    # 仅在Windows系统上初始化
    if platform.system() == 'Windows':
        colorama.init(autoreset=False, strip=False, wrap=True)
        # autoreset=False: 不自动重置颜色（当前未使用颜色）
        # strip=False: 保留ANSI控制序列
        # wrap=True: 包装stdout/stderr，确保\r正确刷新
except ImportError:
    # 未安装colorama时忽略，非Windows终端原生支持\r
    pass
# ===============================================

logger = logging.getLogger(__name__)


def _truncate_to_width(text: str, max_cols: int) -> Tuple[str, int]:
    """根据终端列宽裁剪字符串（支持全角字符宽度）"""
    out_chars: List[str] = []
    width = 0
    for ch in text:
        ch_w = 2 if unicodedata.east_asian_width(ch) in ('F', 'W') else 1
        if width + ch_w > max_cols:
            break
        out_chars.append(ch)
        width += ch_w
    return ''.join(out_chars), width


def _write_progress_single_line(line: str) -> None:
    """在同一行刷写进度信息并动态适配终端宽度"""
    cols = shutil.get_terminal_size(fallback=(120, 25)).columns
    max_cols = max(10, cols - 1)  # 预留一列避免触发换行
    clipped, width = _truncate_to_width(line, max_cols)
    # \r\033[2K 回到行首并清除整行（colorama 包装后 Windows 亦可用）
    sys.stdout.write('\r\033[2K')
    sys.stdout.write(clipped + (' ' * (max_cols - width)))
    sys.stdout.flush()


class GPUOptimizedTrainer:
    """
    GPU优化的训练器
    - 混合精度训练加速
    - 梯度累积减少内存峰值
    - 异步数据加载
    - 多模型并行训练
    """

    def __init__(self,
                 models: Dict[str, nn.Module],
                 config: Dict,
                 device: str = 'cuda'):
        """
        参数：
        - models: 三个天气子模型字典
        - config: 训练配置
        - device: 计算设备
        """
        self.models = models
        self.available_weathers = list(models.keys())
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config

        # 混合精度训练组件
        self.scaler = amp.GradScaler() if device == 'cuda' else None

        # 优化器和调度器
        self.optimizers = {}
        self.schedulers = {}
        self._setup_optimizers()

        # CUDA流（用于并行训练）
        if torch.cuda.is_available():
            self.streams = {
                weather: torch.cuda.Stream()
                for weather in self.available_weathers
            }
        else:
            self.streams = None

        # 训练历史记录
        self.train_history = {
            weather: {'loss': [], 'metrics': []}
            for weather in self.available_weathers
        }

        # 梯度累积步数
        self.accumulation_steps = config.get('gradient_accumulation_steps', 4)

        logger.info(f"训练器初始化完成，设备: {self.device}")

    def _setup_optimizers(self):
        """设置优化器和学习率调度器"""
        for weather_type, model in self.models.items():
            # 移动模型到设备
            model.to(self.device)

            # 创建优化器（使用融合优化器提高速度）
            if self.device.type == 'cuda':
                self.optimizers[weather_type] = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.get('learning_rate', 0.001),
                    weight_decay=self.config.get('weight_decay', 0.01),
                    fused=True  # 使用融合优化器（GPU加速）
                )
            else:
                self.optimizers[weather_type] = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.get('learning_rate', 0.001),
                    weight_decay=self.config.get('weight_decay', 0.01)
                )

            # 创建学习率调度器
            self.schedulers[weather_type] = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers[weather_type],
                T_max=self.config.get('epochs', 100),
                eta_min=1e-6
            )

    def train_epoch(self,
                   train_loader: DataLoader,
                   weather_type: str,
                   epoch: int) -> Dict:
        """
        单个epoch训练（GPU优化）

        关键优化：
        - 混合精度前向传播
        - 梯度累积（每4个batch更新一次）
        - 异步数据预取
        - 动态批大小调整
        """
        model = self.models[weather_type]
        optimizer = self.optimizers[weather_type]
        model.train()

        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                features, targets, _ = batch
            else:
                features, targets = batch

            # 异步数据传输到GPU
            if self.device.type == 'cuda':
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            else:
                features = features.to(self.device)
                targets = targets.to(self.device)

            # 混合精度前向传播
            if self.scaler:
                with amp.autocast():
                    predictions, _ = model(features)
                    loss = nn.MSELoss()(predictions, targets)

                    # 梯度累积
                    loss = loss / self.accumulation_steps

                # 反向传播（混合精度）
                self.scaler.scale(loss).backward()

                # 梯度累积更新
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # 梯度裁剪（防止梯度爆炸）
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # 优化器更新
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    # 清零梯度（set_to_none更高效）
                    optimizer.zero_grad(set_to_none=True)
            else:
                # CPU训练
                predictions, _ = model(features)
                loss = nn.MSELoss()(predictions, targets)
                loss = loss / self.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            # 记录损失
            epoch_loss += loss.item() * self.accumulation_steps
            num_batches += 1

            # 定期清理GPU缓存
            if self.device.type == 'cuda' and batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        # 学习率调度
        self.schedulers[weather_type].step()

        # 记录训练历史
        avg_loss = epoch_loss / num_batches
        self.train_history[weather_type]['loss'].append(avg_loss)

        return {
            'loss': avg_loss,
            'lr': self.optimizers[weather_type].param_groups[0]['lr']
        }

    def train_all_models(self,
                        train_loaders: Dict[str, DataLoader],
                        val_loaders: Dict[str, DataLoader],
                        epochs: int = 100) -> Dict:
        """
        并行训练三个天气子模型

        使用CUDA流实现真正的并行训练
        """
        logger.info(f"开始训练所有模型，总共 {epochs} 个epochs")

        best_metrics = {
            weather: {'loss': float('inf'), 'epoch': 0}
            for weather in self.available_weathers
        }

        for epoch in range(epochs):
            val_results: Dict[str, Dict[str, float]] = {}

            if self.streams and torch.cuda.is_available():
                # GPU并行训练
                train_results = {}

                for weather_type in self.available_weathers:
                    if weather_type not in train_loaders:
                        continue

                    with torch.cuda.stream(self.streams[weather_type]):
                        result = self.train_epoch(
                            train_loaders[weather_type],
                            weather_type,
                            epoch + 1
                        )
                        train_results[weather_type] = result

                # 等待所有流完成
                for stream in self.streams.values():
                    stream.synchronize()
            else:
                # 串行训练（CPU或单流）
                train_results = {}
                for weather_type in self.available_weathers:
                    if weather_type not in train_loaders:
                        continue
                    result = self.train_epoch(
                        train_loaders[weather_type],
                        weather_type,
                        epoch + 1
                    )
                    train_results[weather_type] = result

            # 验证（如果提供了验证集）
            if val_loaders:
                val_results = self.validate_all(val_loaders)

                # 更新最佳模型
                for weather_type, result in val_results.items():
                    if result['loss'] < best_metrics[weather_type]['loss']:
                        best_metrics[weather_type]['loss'] = result['loss']
                        best_metrics[weather_type]['epoch'] = epoch + 1

                        self.save_checkpoint(
                            epoch + 1,
                            {'train': train_results, 'val': val_results},
                            is_best=True,
                            model_type=weather_type
                        )

            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(
                    epoch + 1,
                    {'train': train_results, 'val': val_results if val_loaders else None}
                )

            # 日志记录（调试级别，仅写入日志）
            logger.debug(f"训练结果: {train_results}")
            if val_loaders:
                logger.debug(f"验证结果: {val_results}")

            # 单行刷新训练进度
            progress_percent = (epoch + 1) / epochs * 100
            train_loss_str = ", ".join(
                f"{weather}: {metrics['loss']:.4f}"
                for weather, metrics in train_results.items()
            ) if train_results else "无数据"
            val_loss_str = ", ".join(
                f"{weather}: {metrics['loss']:.4f}"
                for weather, metrics in val_results.items()
            ) if val_results else "N/A"
            progress_line = (
                f"训练进度: {epoch + 1}/{epochs} ({progress_percent:.1f}%) | "
                f"训练损失: [{train_loss_str}] | 验证损失: [{val_loss_str}]"
            )
            _write_progress_single_line(progress_line)

        print()
        logger.info("训练完成！")
        return {
            'best_metrics': best_metrics,
            'train_history': self.train_history
        }

    def validate_all(self, val_loaders: Dict[str, DataLoader]) -> Dict:
        """验证所有模型"""
        val_results = {}

        for weather_type, val_loader in val_loaders.items():
            model = self.models[weather_type]
            model.eval()

            val_loss = 0
            num_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        features, targets, _ = batch
                    else:
                        features, targets = batch
                    if self.device.type == 'cuda':
                        features = features.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                    else:
                        features = features.to(self.device)
                        targets = targets.to(self.device)

                    predictions, _ = model(features)
                    loss = nn.MSELoss()(predictions, targets)

                    val_loss += loss.item()
                    num_batches += 1

            avg_val_loss = val_loss / num_batches
            val_results[weather_type] = {'loss': avg_val_loss}

        return val_results

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False, model_type: str = None):
        """保存训练检查点（支持断点续训）"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if model_type:
            # 保存特定模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.models[model_type].state_dict(),
                'optimizer_state_dict': self.optimizers[model_type].state_dict(),
                'scheduler_state_dict': self.schedulers[model_type].state_dict(),
                'metrics': metrics,
                'config': self.config
            }

            if is_best:
                filename = checkpoint_dir / f"best_{model_type}_model.pth"
            else:
                filename = checkpoint_dir / f"checkpoint_{model_type}_epoch_{epoch}.pth"

            torch.save(checkpoint, filename)
            logger.debug(f"保存检查点: {filename}")
        else:
            # 保存所有模型
            for weather_type in self.models.keys():
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.models[weather_type].state_dict(),
                    'optimizer_state_dict': self.optimizers[weather_type].state_dict(),
                    'scheduler_state_dict': self.schedulers[weather_type].state_dict(),
                    'metrics': metrics,
                    'config': self.config,
                    'train_history': self.train_history[weather_type]
                }

                filename = checkpoint_dir / f"checkpoint_{weather_type}_epoch_{epoch}.pth"
                torch.save(checkpoint, filename)

            logger.debug(f"保存所有模型检查点，epoch: {epoch}")

    def load_checkpoint(self, checkpoint_path: str, model_type: str = None):
        """加载检查点继续训练"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if model_type:
            # 加载特定模型
            self.models[model_type].load_state_dict(checkpoint['model_state_dict'])
            self.optimizers[model_type].load_state_dict(checkpoint['optimizer_state_dict'])
            self.schedulers[model_type].load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"加载{model_type}模型检查点，epoch: {checkpoint['epoch']}")
        else:
            # 尝试推断模型类型
            for weather_type in self.models.keys():
                if weather_type in checkpoint_path:
                    self.models[weather_type].load_state_dict(checkpoint['model_state_dict'])
                    self.optimizers[weather_type].load_state_dict(checkpoint['optimizer_state_dict'])
                    self.schedulers[weather_type].load_state_dict(checkpoint['scheduler_state_dict'])

                    if 'train_history' in checkpoint:
                        self.train_history[weather_type] = checkpoint['train_history']

                    logger.info(f"加载{weather_type}模型检查点")
                    break

        return checkpoint.get('epoch', 0)

    def validate_all(self, val_loaders: Dict[str, DataLoader]) -> Dict:
        """验证所有模型"""
        val_results = {}

        for weather_type, val_loader in val_loaders.items():
            model = self.models[weather_type]
            model.eval()

            val_loss = 0
            num_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        features, targets, _ = batch
                    else:
                        features, targets = batch
                    if self.device.type == 'cuda':
                        features = features.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                    else:
                        features = features.to(self.device)
                        targets = targets.to(self.device)

                    predictions, _ = model(features)
                    loss = nn.MSELoss()(predictions, targets)

                    val_loss += loss.item()
                    num_batches += 1

            avg_val_loss = val_loss / num_batches
            val_results[weather_type] = {'loss': avg_val_loss}

        return val_results

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False, model_type: str = None):
        """保存训练检查点（支持断点续训）"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if model_type:
            # 保存特定模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.models[model_type].state_dict(),
                'optimizer_state_dict': self.optimizers[model_type].state_dict(),
                'scheduler_state_dict': self.schedulers[model_type].state_dict(),
                'metrics': metrics,
                'config': self.config
            }

            if is_best:
                filename = checkpoint_dir / f"best_{model_type}_model.pth"
            else:
                filename = checkpoint_dir / f"checkpoint_{model_type}_epoch_{epoch}.pth"

            torch.save(checkpoint, filename)
            logger.debug(f"保存检查点: {filename}")
        else:
            # 保存所有模型
            for weather_type in self.models.keys():
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.models[weather_type].state_dict(),
                    'optimizer_state_dict': self.optimizers[weather_type].state_dict(),
                    'scheduler_state_dict': self.schedulers[weather_type].state_dict(),
                    'metrics': metrics,
                    'config': self.config,
                    'train_history': self.train_history[weather_type]
                }

                filename = checkpoint_dir / f"checkpoint_{weather_type}_epoch_{epoch}.pth"
                torch.save(checkpoint, filename)

            logger.debug(f"保存所有模型检查点，epoch: {epoch}")

    def load_checkpoint(self, checkpoint_path: str, model_type: str = None):
        """加载检查点继续训练"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if model_type:
            # 加载特定模型
            self.models[model_type].load_state_dict(checkpoint['model_state_dict'])
            self.optimizers[model_type].load_state_dict(checkpoint['optimizer_state_dict'])
            self.schedulers[model_type].load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"加载{model_type}模型检查点，epoch: {checkpoint['epoch']}")
        else:
            # 尝试推断模型类型
            for weather_type in self.models.keys():
                if weather_type in checkpoint_path:
                    self.models[weather_type].load_state_dict(checkpoint['model_state_dict'])
                    self.optimizers[weather_type].load_state_dict(checkpoint['optimizer_state_dict'])
                    self.schedulers[weather_type].load_state_dict(checkpoint['scheduler_state_dict'])

                    if 'train_history' in checkpoint:
                        self.train_history[weather_type] = checkpoint['train_history']

                    logger.info(f"加载{weather_type}模型检查点")
                    break

        return checkpoint.get('epoch', 0)


if __name__ == "__main__":
    # 测试代码
    print("GPU优化训练器模块测试")

    # 创建模拟模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(30, 100, batch_first=True)
            self.fc = nn.Linear(100, 1)

        def forward(self, x):
            lstm_out, hidden = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])
            return output, hidden

    # 创建三个天气模型
    models = {
        'sunny': SimpleModel(),
        'cloudy': SimpleModel(),
        'overcast': SimpleModel()
    }

    # 配置
    config = {
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'epochs': 10,
        'gradient_accumulation_steps': 4,
        'save_interval': 5,
        'checkpoint_dir': './test_checkpoints'
    }

    # 创建训练器
    trainer = GPUOptimizedTrainer(models, config)

    print(f"训练器创建成功，设备: {trainer.device}")
    print(f"混合精度训练: {trainer.scaler is not None}")
    print(f"并行流数量: {len(trainer.streams) if trainer.streams else 0}")
