# -*- coding: utf-8 -*-
"""
LSTM 训练器
支持 early stopping 和验证集监控
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time

from src.utils.seed import get_dataloader_generator, worker_init_fn


@dataclass
class TrainConfig:
    """训练配置"""
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10  # early stopping patience
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42  # 随机种子
    num_workers: int = 0  # DataLoader workers


@dataclass
class EpochResult:
    """单个 epoch 的结果"""
    epoch: int
    train_loss: float
    val_loss: float
    val_rmse_4h_pu: float  # 4h RMSE in p.u.
    elapsed_sec: float


@dataclass
class TrainResult:
    """训练结果"""
    best_epoch: int
    best_val_loss: float
    best_val_rmse_4h_pu: float
    epoch_results: List[EpochResult]
    stopped_early: bool


class LSTMTrainer:
    """LSTM 训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        nominal_capacity: float = 50.0,
        horizons: List[int] = None,
        horizon_names: List[str] = None
    ):
        """
        Args:
            model: LSTM 模型
            config: 训练配置
            nominal_capacity: 装机容量 (MW)
            horizons: 评估的 horizon 列表 (步数)
            horizon_names: horizon 名称列表
        """
        self.model = model.to(config.device)
        self.config = config
        self.nominal_capacity = nominal_capacity
        self.horizons = horizons or [4, 8, 16]
        # horizon_names 可能是字典或列表
        if isinstance(horizon_names, dict):
            self.horizon_names = [horizon_names.get(h, f"{h}step") for h in self.horizons]
        else:
            self.horizon_names = horizon_names or ["1h", "2h", "4h"]
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.criterion = nn.MSELoss()
    
    def _prepare_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True
    ) -> DataLoader:
        """准备 DataLoader (支持可复现性)"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 可复现性配置
        generator = get_dataloader_generator(self.config.seed) if shuffle else None
        init_fn = worker_init_fn if self.config.num_workers > 0 else None
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.config.num_workers,
            generator=generator,
            worker_init_fn=init_fn
        )
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """验证并返回 loss 和各 horizon 的 RMSE (p.u.)"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_preds = []
        all_targets = []
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device)
            
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
        
        avg_loss = total_loss / n_batches
        
        # 计算各 horizon 的 RMSE
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        rmse_by_horizon = {}
        for h, name in zip(self.horizons, self.horizon_names):
            # horizon 是步数，索引从 0 开始，所以取 :h
            errors = preds[:, :h] - targets[:, :h]
            rmse_pu = np.sqrt(np.mean(errors ** 2))
            rmse_mw = rmse_pu * self.nominal_capacity
            nrmse = rmse_pu  # nRMSE = RMSE_MW / capacity = RMSE_pu
            rmse_by_horizon[name] = {
                'rmse_pu': rmse_pu,
                'rmse_mw': rmse_mw,
                'nrmse': nrmse
            }
        
        return avg_loss, rmse_by_horizon
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> TrainResult:
        """
        训练模型
        
        Args:
            X_train: 训练集输入 (n_samples, Lx, n_features)
            y_train: 训练集目标 (n_samples, Hmax) - Power_pu
            X_val: 验证集输入
            y_val: 验证集目标
            verbose: 是否打印训练过程
        
        Returns:
            TrainResult
        """
        train_loader = self._prepare_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._prepare_dataloader(X_val, y_val, shuffle=False)
        
        best_val_loss = float('inf')
        best_val_rmse_4h = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state_dict = None
        
        epoch_results = []
        
        if verbose:
            print("\n" + "=" * 70)
            print("LSTM Training")
            print("=" * 70)
            print(f"Device: {self.config.device}")
            print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
            print(f"Batch size: {self.config.batch_size}, LR: {self.config.lr}")
            print(f"Max epochs: {self.config.max_epochs}, Patience: {self.config.patience}")
            print("-" * 70)
            print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
                  f"{'4h RMSE(MW)':>12} | {'4h nRMSE':>10} | {'Time':>6}")
            print("-" * 70)
        
        for epoch in range(1, self.config.max_epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss = self._train_epoch(train_loader)
            
            # 验证
            val_loss, rmse_by_horizon = self._validate(val_loader)
            
            elapsed = time.time() - start_time
            
            # 获取 4h RMSE
            rmse_4h_pu = rmse_by_horizon['4h']['rmse_pu']
            rmse_4h_mw = rmse_by_horizon['4h']['rmse_mw']
            nrmse_4h = rmse_by_horizon['4h']['nrmse']
            
            epoch_result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_rmse_4h_pu=rmse_4h_pu,
                elapsed_sec=elapsed
            )
            epoch_results.append(epoch_result)
            
            if verbose:
                print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                      f"{rmse_4h_mw:>12.4f} | {nrmse_4h:>10.4f} | {elapsed:>5.1f}s")
            
            # Early stopping 监控 4h RMSE
            if rmse_4h_pu < best_val_rmse_4h:
                best_val_rmse_4h = rmse_4h_pu
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                if verbose:
                    print("-" * 70)
                    print(f"Early stopping at epoch {epoch} (patience={self.config.patience})")
                break
        
        # 恢复最佳模型
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            model_source = "best"
        else:
            model_source = "last"
        
        stopped_early = patience_counter >= self.config.patience
        
        if verbose:
            print("-" * 70)
            print(f"Best epoch: {best_epoch}, Best 4h RMSE(p.u.): {best_val_rmse_4h:.6f}")
            print(f"[Checkpoint] 评估使用: {model_source} epoch 权重 (epoch={best_epoch})")
            print("=" * 70)
        
        return TrainResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_rmse_4h_pu=best_val_rmse_4h,
            epoch_results=epoch_results,
            stopped_early=stopped_early
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: (n_samples, Lx, n_features)
        
        Returns:
            (n_samples, Hmax) - Power_pu 预测
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.config.device)
        
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        
        return y_pred.cpu().numpy()
