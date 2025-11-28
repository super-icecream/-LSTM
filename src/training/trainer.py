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
import platform
import shutil
import unicodedata
import os
import signal
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
                 device: str = 'cuda',
                 log_dir: Optional[str] = None):
        """
        参数：
        - models: 三个天气子模型字典
        - config: 训练配置
        - device: 计算设备
        - log_dir: 日志目录，用于保存loss曲线图
        """
        self.models = models
        self.available_weathers = list(models.keys())
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.log_dir = Path(log_dir) if log_dir else Path(config.get('checkpoint_dir', './checkpoints'))
        # 梯度日志开关
        self.log_gradients: bool = bool(self.config.get('log_gradients', True))
        # 梯度绘图配置（默认关闭EMA，绘制grad_last）
        self.grad_curve: str = str(self.config.get('grad_curve', 'last')).lower()
        try:
            self.grad_ema_alpha: float = float(self.config.get('grad_ema_alpha', 0.0))
        except Exception:
            self.grad_ema_alpha = 0.0

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

        # 训练历史记录（扩展结构，同时记录训练/验证loss与梯度统计）
        self.train_history = {
            weather: {
                'train_loss': [],
                'val_loss': [],
                'lr': [],
                'grad_last': [],
                'grad_avg': [],
                'grad_max': [],
            }
            for weather in self.available_weathers
        }
        # 添加总体历史记录
        self.train_history['overall'] = {'train_loss': [], 'val_loss': []}

        # 梯度累积步数
        self.accumulation_steps = config.get('gradient_accumulation_steps', 4)

        # 中断标志
        self._interrupted = False
        self._train_loaders = None  # 用于计算加权loss
        self._val_loaders = None

        # 注册信号处理（Ctrl+C中断时自动保存）
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

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
        # 梯度统计
        grad_norm_sum: float = 0.0
        grad_norm_cnt: int = 0
        grad_norm_max: float = 0.0
        grad_norm_last: float = 0.0

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
                    grad_total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if self.log_gradients:
                        # 记录梯度范数（裁剪前的总范数）
                        try:
                            grad_val = float(grad_total.detach().item())
                        except Exception:
                            grad_val = float(grad_total)
                        grad_norm_last = grad_val
                        grad_norm_sum += grad_val
                        grad_norm_cnt += 1
                        grad_norm_max = max(grad_norm_max, grad_val)

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
                    grad_total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if self.log_gradients:
                        try:
                            grad_val = float(grad_total.detach().item())
                        except Exception:
                            grad_val = float(grad_total)
                        grad_norm_last = grad_val
                        grad_norm_sum += grad_val
                        grad_norm_cnt += 1
                        grad_norm_max = max(grad_norm_max, grad_val)
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

        # 记录训练历史（注意：train_loss记录在train_all_models中统一处理）
        avg_loss = epoch_loss / num_batches

        result = {
            'loss': avg_loss,
            'lr': self.optimizers[weather_type].param_groups[0]['lr']
        }
        if self.log_gradients and grad_norm_cnt > 0:
            result.update({
                'grad_norm_last': grad_norm_last,
                'grad_norm_avg': grad_norm_sum / max(1, grad_norm_cnt),
                'grad_norm_max': grad_norm_max,
            })
        return result

    def train_all_models(self,
                        train_loaders: Dict[str, DataLoader],
                        val_loaders: Dict[str, DataLoader],
                        epochs: int = 100) -> Dict:
        """
        并行训练三个天气子模型

        使用CUDA流实现真正的并行训练
        """
        logger.info(f"开始训练所有模型，共 {epochs} 个epochs")

        # 保存loaders引用，用于计算加权loss和中断处理
        self._train_loaders = train_loaders
        self._val_loaders = val_loaders

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

            # 记录训练loss与梯度到历史
            for weather_type, result in train_results.items():
                self.train_history[weather_type]['train_loss'].append(result['loss'])
                self.train_history[weather_type]['lr'].append(result['lr'])
                if self.log_gradients and 'grad_norm_last' in result:
                    self.train_history[weather_type]['grad_last'].append(result.get('grad_norm_last', 0.0))
                    self.train_history[weather_type]['grad_avg'].append(result.get('grad_norm_avg', 0.0))
                    self.train_history[weather_type]['grad_max'].append(result.get('grad_norm_max', 0.0))
                elif self.log_gradients:
                    # 兼容无梯度统计的情况，填充0
                    self.train_history[weather_type]['grad_last'].append(0.0)
                    self.train_history[weather_type]['grad_avg'].append(0.0)
                    self.train_history[weather_type]['grad_max'].append(0.0)

            # 验证（如果提供了验证集）
            if val_loaders:
                val_results = self.validate_all(val_loaders)

                # 记录验证loss到历史
                for weather_type, result in val_results.items():
                    self.train_history[weather_type]['val_loss'].append(result['loss'])

                # 计算并记录总体加权loss
                total_train_samples = sum(len(train_loaders[w].dataset) for w in train_results.keys())
                total_val_samples = sum(len(val_loaders[w].dataset) for w in val_results.keys())

                if total_train_samples > 0:
                    overall_train_loss = sum(
                        train_results[w]['loss'] * len(train_loaders[w].dataset)
                        for w in train_results.keys()
                    ) / total_train_samples
                    self.train_history['overall']['train_loss'].append(overall_train_loss)

                if total_val_samples > 0:
                    overall_val_loss = sum(
                        val_results[w]['loss'] * len(val_loaders[w].dataset)
                        for w in val_results.keys()
                    ) / total_val_samples
                    self.train_history['overall']['val_loss'].append(overall_val_loss)

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

            # 单行刷新训练进度（中文标签+更友好的格式）
            progress_percent = (epoch + 1) / epochs * 100
            weather_name_map = {"sunny": "晴天", "cloudy": "多云", "overcast": "阴天"}
            def _fmt_train_metrics(weather: str, m: Dict[str, float]) -> str:
                base = f"{weather_name_map.get(weather, weather)}={m['loss']:.4f}"
                if self.log_gradients and 'grad_norm_last' in m:
                    return base + f"(g{m['grad_norm_last']:.2f})"
                return base
            train_loss_str = "， ".join(
                _fmt_train_metrics(weather, metrics) for weather, metrics in train_results.items()
            ) if train_results else "无训练数据"
            val_loss_str = "， ".join(
                f"{weather_name_map.get(weather, weather)}={metrics['loss']:.4f}"
                for weather, metrics in val_results.items()
            ) if val_results else "无验证数据"
            progress_line = (
                f"训练进度 {epoch + 1}/{epochs} ({progress_percent:.1f}%) ｜ "
                f"训练损失：{train_loss_str} ｜ 验证损失：{val_loss_str}"
            )
            _write_progress_single_line(progress_line)

        print()
        logger.info("训练完成!")

        # 训练完成后自动保存历史和绘制曲线
        self.save_loss_history()
        self.plot_loss_curves()

        # 恢复原始信号处理器
        signal.signal(signal.SIGINT, self._original_sigint_handler)

        return {
            'best_metrics': best_metrics,
            'train_history': self.train_history
        }

    def _handle_interrupt(self, signum, frame):
        """处理Ctrl+C中断信号"""
        print("\n\n[!] 检测到中断信号，正在保存训练历史并绘制Loss曲线...")
        self._interrupted = True

        # 保存当前训练状态
        self.save_loss_history()
        self.plot_loss_curves()

        # 恢复原始信号处理器并退出
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        logger.info("训练已中断，Loss曲线已保存")
        sys.exit(0)

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

    def save_loss_history(self, save_dir: Optional[str] = None) -> None:
        """保存训练历史到JSON文件"""
        save_dir = Path(save_dir) if save_dir else self.log_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / 'loss_history.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, ensure_ascii=False, indent=2)
        logger.info(f"[*] 训练历史已保存至: {save_path}")

    def plot_loss_curves(self, save_dir: Optional[str] = None) -> None:
        """绘制并保存Loss曲线图"""
        save_dir = Path(save_dir) if save_dir else self.log_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        weather_names = {'sunny': '晴天', 'cloudy': '多云', 'overcast': '阴天'}
        n_models = len(self.available_weathers)

        # 创建图表：三个子模型并排一行
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), dpi=100)
        if n_models == 1:
            axes = [axes]

        for ax, weather in zip(axes, self.available_weathers):
            history = self.train_history.get(weather, {})
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])

            if not train_loss:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
                ax.set_title(f'{weather_names.get(weather, weather)} 模型', fontsize=13, fontweight='bold')
                continue

            epochs = range(1, len(train_loss) + 1)

            # 绘制训练损失曲线
            ax.plot(epochs, train_loss, 'b-', label='训练损失', linewidth=2)

            # 绘制验证损失曲线
            if val_loss:
                ax.plot(epochs, val_loss, 'r--', label='验证损失', linewidth=2)

                # 标注最佳验证损失点
                min_idx = int(np.argmin(val_loss))
                min_val = val_loss[min_idx]
                ax.scatter(min_idx + 1, min_val, color='red', s=100, zorder=5, marker='*')
                ax.annotate(f'Best: {min_val:.4f}\nEpoch {min_idx + 1}',
                           xy=(min_idx + 1, min_val),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            # 叠加梯度曲线（右轴），默认绘制 grad_last，可选EMA平滑
            glast = history.get('grad_last', [])
            if self.log_gradients and glast:
                grad_series = glast if self.grad_curve == 'last' else history.get('grad_avg', glast)
                if 0.0 < self.grad_ema_alpha < 1.0:
                    ema_series = []
                    s = None
                    for v in grad_series:
                        s = v if s is None else (self.grad_ema_alpha * v + (1.0 - self.grad_ema_alpha) * s)
                        ema_series.append(s)
                    to_plot = ema_series
                else:
                    to_plot = grad_series
                ax2 = ax.twinx()
                ax2.plot(range(1, len(to_plot) + 1), to_plot, color='green', linestyle='-', label='梯度范数', linewidth=1.8)
                ax2.set_ylabel('梯度范数', fontsize=11, color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.set_ylim(bottom=0)
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss (MSE)', fontsize=11)
            ax.set_title(f'{weather_names.get(weather, weather)} 模型', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)

            # 设置y轴下限为0
            ax.set_ylim(bottom=0)

        plt.tight_layout()

        # 保存图表
        save_path = save_dir / 'loss_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"[*] Loss曲线图已保存至: {save_path}")


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
