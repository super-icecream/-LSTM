"""
动态相空间重构模块 (DPSR)
功能：实现基于NCA的动态权重学习和相空间重构
作者：DLFE-LSTM-WSI Team
日期：2025-09-26
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
import pickle
import sys
import time
from scipy.optimize import minimize
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda"
    else:
        DEFAULT_DEVICE = "cpu"
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


class DPSR:
    """
    动态相空间重构 (Dynamic Phase Space Reconstruction)

    基于NCA（邻域成分分析）的动态权重学习，实现时变特征的相空间重构。
    将9维输入（5个IMF分量 + 4个气象特征）重构为30维动态特征向量。

    Attributes:
        embedding_dim (int): 嵌入维度（默认30）
        neighborhood_size (int): 局部邻域大小
        regularization (float): 正则化参数λ
        time_delay (int): 时间延迟τ
        max_iter (int): 最大迭代次数
    """

    def __init__(self,
                 embedding_dim: Union[int, str] = 30,
                 embedding_dim_range: Tuple[int, int] = (15, 30),
                 neighborhood_size: int = 32,
                 regularization: float = 0.01,
                 time_delay: Union[int, str] = 1,
                 max_iter: int = 100,
                 learning_rate: float = 0.01,
                 sigma_scale: float = 0.5,
                 max_time_delay: int = 60,
                 device: str = "auto"):
        """
        初始化DPSR

        Args:
            embedding_dim: 目标嵌入维度M=30
            neighborhood_size: 局部邻域大小L
            regularization: L2正则化参数λ
            time_delay: 时间延迟τ（默认1）
            max_iter: NCA优化最大迭代次数
            learning_rate: 初始学习率
        """
        self.embedding_dim = embedding_dim
        self.embedding_dim_range = embedding_dim_range
        self.neighborhood_size = int(np.clip(neighborhood_size, 24, 48))
        self.regularization = float(np.clip(regularization, 0.001, 0.1))
        self.time_delay = time_delay
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.sigma_scale = sigma_scale
        self.max_time_delay = max_time_delay

        self.actual_embedding_dim: Optional[int] = None
        self.actual_time_delay: Optional[int] = None

        # 动态权重存储
        self.weights = {}  # 每个时间点的权重
        self.global_weights = None  # 全局权重（训练后平均）
        self.is_fitted = False

        # 优化历史
        self.optimization_history = {
            'loss': [],
            'accuracy': []
        }

        # Device initialization
        self.use_gpu = False
        self.device = "cpu"
        self._torch = torch if TORCH_AVAILABLE else None
        self._torch_device = None

        if TORCH_AVAILABLE:
            selected_device = device
            if selected_device not in ("auto", "cpu", "cuda"):
                logger.warning("Unknown device type %s, fallback to auto mode", selected_device)
                selected_device = "auto"

            if selected_device == "auto":
                selected_device = DEFAULT_DEVICE

            if selected_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, DPSR falls back to CPU mode")
                selected_device = "cpu"

            self.device = selected_device
            self.use_gpu = self.device == "cuda"
            self._torch_device = torch.device(self.device)

            if self.use_gpu:
                try:
                    device_name = torch.cuda.get_device_name(self._torch_device)
                except Exception:  # pragma: no cover - driver differences
                    device_name = "CUDA"
                logger.info("DPSR enabling GPU execution on %s", device_name)
            else:
                logger.info("DPSR running on CPU mode (PyTorch detected, CUDA disabled)")
        else:
            if device == "cuda":
                logger.warning("PyTorch not installed; CUDA mode unavailable, using CPU")
            logger.info("DPSR running on CPU mode (PyTorch not available)")

        logger.info(
            "DPSR初始化: 目标嵌入维度=%s, 邻域大小=%d, 正则化=%.4f",
            str(embedding_dim), self.neighborhood_size, self.regularization
        )

    def _default_target_dim(self) -> int:
        if isinstance(self.embedding_dim, int):
            return self.embedding_dim
        return self.embedding_dim_range[1]

    @staticmethod
    def _prepare_day_mask(day_mask: Optional[Union[np.ndarray, List[bool]]],
                          n_samples: int,
                          module: str = "DPSR") -> Optional[np.ndarray]:
        if day_mask is None:
            return None
        mask_array = np.asarray(day_mask, dtype=bool).reshape(-1)
        if mask_array.shape[0] != n_samples:
            raise ValueError(
                f"{module} day_mask 长度({mask_array.shape[0]}) "
                f"与样本数({n_samples}) 不一致"
            )
        return mask_array

    def _compute_weighted_distances(self,
                                    data: Union[np.ndarray, "torch.Tensor"],
                                    w_squared: Union[np.ndarray, "torch.Tensor"]):
        """计算加权欧氏距离矩阵（自动选择CPU/GPU）"""
        if self.use_gpu and self._torch is not None:
            torch_module = self._torch
            data_tensor = data if isinstance(data, torch_module.Tensor) else torch_module.as_tensor(
                data, dtype=torch_module.float32, device=self._torch_device
            )
            weight_tensor = w_squared if isinstance(w_squared, torch_module.Tensor) else torch_module.as_tensor(
                w_squared, dtype=torch_module.float32, device=self._torch_device
            )

            diff = data_tensor.unsqueeze(1) - data_tensor.unsqueeze(0)
            weighted = torch_module.sum(diff * diff * weight_tensor.view(1, 1, -1), dim=2)
            distances = torch_module.sqrt(torch_module.clamp(weighted, min=0.0))
            distances.fill_diagonal_(0.0)
            return distances

        diff = data[:, None, :] - data[None, :, :]
        weighted = np.sum(w_squared.reshape(1, 1, -1) * diff ** 2, axis=2)
        distances = np.sqrt(np.maximum(weighted, 0.0))
        np.fill_diagonal(distances, 0.0)
        return distances

    def _objective_gpu(self,
                       X_tensor: "torch.Tensor",
                       y_tensor: "torch.Tensor",
                       weights: np.ndarray) -> float:
        """使用GPU计算目标函数值。"""
        torch_module = self._torch
        w_tensor = torch_module.as_tensor(weights, dtype=torch_module.float32, device=self._torch_device)
        w_squared = w_tensor.pow(2)

        distances = self._compute_weighted_distances(X_tensor, w_squared)
        sigma_val = float(self._compute_adaptive_sigma(distances.detach().cpu().numpy()))
        sigma = torch_module.tensor(sigma_val, device=self._torch_device, dtype=torch_module.float32)

        kernel = torch_module.exp(-distances / sigma)
        kernel.fill_diagonal_(0.0)
        row_sums = kernel.sum(dim=1, keepdim=True) + 1e-8
        probabilities = kernel / row_sums

        y_pred = probabilities @ y_tensor
        mae = torch_module.mean(torch_module.abs(y_tensor - y_pred))
        regularization_term = self.regularization * torch_module.sum(w_tensor.pow(2))

        loss = mae + regularization_term
        return float(loss.item())

    def _gradient_gpu(self,
                      X_tensor: "torch.Tensor",
                      y_tensor: "torch.Tensor",
                      weights: np.ndarray) -> np.ndarray:
        """使用GPU计算梯度，消除三重嵌套循环。"""
        torch_module = self._torch
        n_samples = X_tensor.shape[0]

        w_tensor = torch_module.as_tensor(weights, dtype=torch_module.float32, device=self._torch_device)
        w_squared = w_tensor.pow(2)

        diff = X_tensor.unsqueeze(1) - X_tensor.unsqueeze(0)
        diff_sq = diff.pow(2)

        weighted = torch_module.sum(diff_sq * w_squared.view(1, 1, -1), dim=2)
        distances = torch_module.sqrt(torch_module.clamp(weighted, min=0.0))
        distances.fill_diagonal_(0.0)

        sigma_val = float(self._compute_adaptive_sigma(distances.detach().cpu().numpy()))
        sigma = torch_module.tensor(sigma_val, device=self._torch_device, dtype=torch_module.float32)

        kernel = torch_module.exp(-distances / sigma)
        kernel.fill_diagonal_(0.0)

        row_sums = kernel.sum(dim=1, keepdim=True) + 1e-8
        probabilities = kernel / row_sums
        y_pred = probabilities @ y_tensor
        diff_y = y_tensor - y_pred
        mae_grad = -torch_module.sign(diff_y)

        dist_safe = torch_module.where(distances <= 1e-10,
                                       torch_module.ones_like(distances),
                                       distances)

        weight_view = w_tensor.view(1, 1, -1)
        kernel_grads = kernel.unsqueeze(-1) * (-weight_view * diff_sq / dist_safe.unsqueeze(-1))

        eye_mask = torch_module.eye(n_samples, device=self._torch_device, dtype=torch_module.bool)
        kernel_grads = kernel_grads.masked_fill(eye_mask.unsqueeze(-1), 0.0)

        sum_kernel_grads = kernel_grads.sum(dim=1)

        row_sums_inv = 1.0 / row_sums
        prob_grad = kernel_grads * row_sums_inv
        prob_grad -= probabilities.unsqueeze(-1) * sum_kernel_grads.unsqueeze(1) * row_sums_inv
        prob_grad = prob_grad.masked_fill(eye_mask.unsqueeze(-1), 0.0)

        mae_matrix = mae_grad.view(-1, 1, 1)
        y_matrix = y_tensor.view(1, -1, 1)

        grad_tensor = (mae_matrix * y_matrix * prob_grad).sum(dim=(0, 1)) / n_samples
        grad_tensor = grad_tensor + 2 * self.regularization * w_tensor
        grad_tensor = torch_module.clamp(grad_tensor, min=-1e3, max=1e3)

        grad_np = grad_tensor.detach().cpu().numpy()
        if not np.all(np.isfinite(grad_np)):
            logger.warning("Gradient contains NaN or Inf in GPU path; zero out gradient")
            grad_np = np.zeros_like(weights)
        return grad_np

    def _compute_adaptive_sigma(self, distances: np.ndarray) -> float:
        """Return constant kernel width sigma=1.0 (Gu et al., IEEE TSTE 2025)."""
        # Reference: Z. Gu et al., Photovoltaic Power Prediction Considering Multifactorial Dynamic Effects,
        # IEEE Transactions on Sustainable Energy, vol. 16, no. 3, pp. 2197-2209, 2025.
        return 1.0

    @staticmethod
    def _average_mutual_information(signal: np.ndarray, lag: int) -> float:
        """估计平均互信息，采用简单直方图估计"""
        if lag <= 0 or lag >= len(signal):
            return np.inf

        x = signal[:-lag]
        y = signal[lag:]
        bins = int(np.sqrt(len(x)))
        histogram_xy, _, _ = np.histogram2d(x, y, bins=bins)
        histogram_x, _ = np.histogram(x, bins=bins)
        histogram_y, _ = np.histogram(y, bins=bins)

        pxy = histogram_xy / np.sum(histogram_xy)
        px = histogram_x / np.sum(histogram_x)
        py = histogram_y / np.sum(histogram_y)

        px_py = px[:, None] * py[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(pxy > 0, pxy / (px_py + 1e-12), 1)
            ami = np.nansum(pxy * np.log(ratio + 1e-12))
        return ami

    def _determine_time_delay(self, signal: np.ndarray) -> int:
        """基于平均互信息确定时间延迟τ"""
        if isinstance(self.time_delay, int):
            return max(1, min(self.time_delay, self.max_time_delay))

        ami_values = []
        for lag in range(1, self.max_time_delay + 1):
            ami = self._average_mutual_information(signal, lag)
            ami_values.append((lag, ami))

        for idx in range(1, len(ami_values)):
            if ami_values[idx][1] > ami_values[idx - 1][1]:
                return ami_values[idx - 1][0]

        return ami_values[-1][0]

    def _determine_embedding_dim(self, signal: np.ndarray) -> int:
        """使用假近邻法估算嵌入维度"""
        if isinstance(self.embedding_dim, int):
            return max(self.embedding_dim_range[0], min(self.embedding_dim, self.embedding_dim_range[1]))

        window = min(200, len(signal) - 1)
        if window <= 0:
            return self.embedding_dim_range[0]

        normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
        max_dim = self.embedding_dim_range[1]
        threshold = 0.01

        for m in range(self.embedding_dim_range[0], max_dim + 1):
            embedded_m = self.phase_space_embedding(normalized, m, self.actual_time_delay or 1)
            embedded_m1 = self.phase_space_embedding(normalized, m + 1, self.actual_time_delay or 1)

            truncated = min(len(embedded_m), len(embedded_m1))
            if truncated == 0:
                continue

            ratios = []
            for i in range(truncated):
                diff_m = embedded_m[i] - embedded_m[(i + 1) % truncated]
                diff_m1 = embedded_m1[i] - embedded_m1[(i + 1) % truncated]
                denom = np.linalg.norm(diff_m) + 1e-12
                ratios.append(np.linalg.norm(diff_m1) / denom)

            mean_ratio = np.mean(ratios)
            if mean_ratio < threshold:
                return m

        return max_dim

    def construct_neighborhood(self,
                             data: np.ndarray,
                             time_idx: int,
                             window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建局部时序邻域

        对于时刻i，构建局部邻域Si = {zi-L+1, ..., zi}，
        保持时序连续性，不跨数据集边界。

        Args:
            data: 输入数据 (n_samples x n_features)
            time_idx: 当前时间索引
            window_size: 邻域窗口大小，如果None则使用默认值

        Returns:
            Tuple[np.ndarray, np.ndarray]: (邻域数据, 对应索引)
        """
        if window_size is None:
            window_size = self.neighborhood_size

        n_samples = data.shape[0]

        # 确定邻域边界
        start_idx = max(0, time_idx - window_size + 1)
        end_idx = min(n_samples, time_idx + 1)

        # 提取邻域数据
        neighborhood_data = data[start_idx:end_idx]
        neighborhood_indices = np.arange(start_idx, end_idx)

        # 如果邻域太小，进行填充
        if len(neighborhood_data) < 3:
            # 至少需要3个样本
            if time_idx == 0:
                # 使用后续数据填充
                end_idx = min(3, n_samples)
                neighborhood_data = data[0:end_idx]
                neighborhood_indices = np.arange(0, end_idx)
            else:
                # 使用前面的数据填充
                start_idx = max(0, time_idx - 2)
                neighborhood_data = data[start_idx:time_idx + 1]
                neighborhood_indices = np.arange(start_idx, time_idx + 1)

        return neighborhood_data, neighborhood_indices

    def phase_space_embedding(self,
                             signal: np.ndarray,
                             embedding_dim: int,
                             time_delay: int = 1) -> np.ndarray:
        """
        经典相空间重构

        将一维时序重构为高维相空间：
        zi = [xi, xi-τ, xi-2τ, ..., xi-(m-1)τ]

        Args:
            signal: 一维时序信号
            embedding_dim: 嵌入维度m
            time_delay: 时间延迟τ

        Returns:
            重构的相空间矩阵
        """
        n = len(signal)
        m = embedding_dim

        # 计算重构后的样本数
        n_embedded = n - (m - 1) * time_delay

        if n_embedded <= 0:
            raise ValueError(f"信号太短，无法进行{m}维嵌入")

        # 构建相空间矩阵
        embedded = np.zeros((n_embedded, m))

        for i in range(n_embedded):
            embedded[i] = signal[i:i + m * time_delay:time_delay]

        return embedded

    def nca_optimization(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        init_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        NCA（邻域成分分析）权重优化

        优化目标函数：
        f(w) = (1/n)Σl(yi, ŷi) + λΣwr²

        其中预测值：
        ŷi = Σpij*yj, pij = exp(-dw(si,sj)^2 / (2σ^2)) / Σexp(-dw(si,sk)^2 / (2σ^2))

        加权距离：
        dw(si,sj) = sqrt(Σwr² * (sir - sjr)²)

        Args:
            X: 特征矩阵 (n_samples x n_features)
            y: 目标值 (n_samples,)
            init_weights: 初始权重

        Returns:
            优化后的权重向量
        """
        n_samples, n_features = X.shape
        
        # 诊断信息

        # 初始化权重
        if init_weights is None:
            weights = np.full(n_features, 1.0 / n_features)
        else:
            weights = init_weights.copy()

        # 标准化目标值（回归问题）
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-8
        y_normalized = (y - y_mean) / y_std

        # 进度跟踪
        optimization_start_time = time.time()
        iteration_count = 0
        last_loss = None

        torch_module = self._torch if (self.use_gpu and self._torch is not None) else None
        if torch_module is not None:
            X_tensor = torch_module.as_tensor(X, dtype=torch_module.float32, device=self._torch_device)
            y_tensor = torch_module.as_tensor(y_normalized, dtype=torch_module.float32, device=self._torch_device)
        else:
            X_tensor = None
            y_tensor = None

        # 定义目标函数
        def objective(w):
            """Objective function with MAE loss (Gu et al., Eq. 11)."""
            w = w.reshape(-1)
            if torch_module is not None:
                return self._objective_gpu(X_tensor, y_tensor, w)

            w_squared = w ** 2
            distances = self._compute_weighted_distances(X, w_squared)
            sigma = self._compute_adaptive_sigma(distances)
            with np.errstate(divide='ignore', invalid='ignore'):
                K = np.exp(-distances / sigma)
                K = np.maximum(K, 1e-10)
            np.fill_diagonal(K, 0.0)
            row_sums = np.sum(K, axis=1, keepdims=True) + 1e-8
            probabilities = K / row_sums
            y_pred = probabilities @ y_normalized
            prediction_error = np.mean(np.abs(y_normalized - y_pred))
            regularization_term = self.regularization * np.sum(w ** 2)
            return prediction_error + regularization_term

        def gradient(w):
            """Gradient of the MAE-based objective (Gu et al., Eq. 12)."""
            w = w.reshape(-1)
            if torch_module is not None:
                return self._gradient_gpu(X_tensor, y_tensor, w)

            w_squared = w ** 2
            distances = self._compute_weighted_distances(X, w_squared)
            sigma = self._compute_adaptive_sigma(distances)

            with np.errstate(divide='ignore', invalid='ignore'):
                K = np.exp(-distances / sigma)
                K = np.maximum(K, 1e-10)
            np.fill_diagonal(K, 0.0)

            row_sums = np.sum(K, axis=1, keepdims=True) + 1e-8
            probabilities = K / row_sums
            y_pred = probabilities @ y_normalized
            diff_y = y_normalized - y_pred
            mae_grad = -np.sign(diff_y)

            grad = np.zeros(n_features)

            for k in range(n_features):
                grad_k = 0.0

                for i in range(n_samples):
                    kernel_grads_i = np.zeros(n_samples)

                    for j in range(n_samples):
                        if i == j:
                            continue

                        diff_ijk = X[i, k] - X[j, k]
                        if distances[i, j] > 1e-10:
                            dist_ij = max(distances[i, j], 1e-10)
                            kernel_grads_i[j] = K[i, j] * (
                                -w[k] * (diff_ijk ** 2) / dist_ij
                            )

                    sum_kernel_grads = np.sum(kernel_grads_i)

                    for j in range(n_samples):
                        if i == j:
                            continue

                        prob_grad_ij = (
                            kernel_grads_i[j] / row_sums[i, 0]
                            - probabilities[i, j] * sum_kernel_grads / row_sums[i, 0]
                        )

                        grad_k += mae_grad[i] * y_normalized[j] * prob_grad_ij

                grad[k] = grad_k / n_samples + 2 * self.regularization * w[k]

            grad = np.clip(grad, -1e3, 1e3)

            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                logger.warning("Gradient contains NaN or Inf; zero out gradient")
                grad = np.zeros_like(w)

            return grad

        # 优化器迭代回调函数（静默模式，由上层显示进度）
        def callback(xk):
            nonlocal iteration_count, last_loss
            iteration_count += 1
            
            # 只记录，不输出（由上层显示进度）
            current_loss = objective(xk)
            last_loss = current_loss

        # Optimization settings
        options = {
            'maxiter': self.max_iter,
            'gtol': 1e-3,
            'ftol': 1e-5,
            'maxls': 30,
            'disp': False
        }

        # Run optimization (静默模式，由上层显示进度)
        result = minimize(
            objective,
            weights,
            method='L-BFGS-B',
            jac=gradient,
            bounds=[(0, 1)] * n_features,
            options=options,
            callback=callback  # 添加回调
        )

        if result.success:
            optimized_weights = result.x
            optimized_weights = optimized_weights / (np.sum(optimized_weights) + 1e-10)
        else:
            if np.all(result.x >= 0) and np.sum(result.x) > 0:
                optimized_weights = result.x / (np.sum(result.x) + 1e-10)
            else:
                optimized_weights = np.ones(n_features) / n_features

        self.optimization_history['loss'].append(result.fun if result.success else np.inf)

        # 返回权重和优化结果（供上层显示进度）
        return optimized_weights, {
            'loss': result.fun,
            'iter': result.nit,
            'success': result.success,
            'time': time.time() - optimization_start_time
        }

    def dynamic_reconstruction(self,
                             data: np.ndarray,
                             weights: np.ndarray) -> np.ndarray:
        """
        动态相空间重构

        使用NCA学习的权重进行动态重构：
        Vi = [wi1*xi1, wi2*xi-1,1, ..., wiM*xi-M+1,1,
              ..., wid*xid, ..., wiMd*xi-M+1,d]

        Args:
            data: 输入数据 (n_samples x n_features)
            weights: 特征权重

        Returns:
            重构的特征向量 (n_samples x embedding_dim)
        """
        n_samples, n_features = data.shape

        # 确定每个特征的嵌入维度
        target_dim = self.actual_embedding_dim if self.actual_embedding_dim is not None else (
            self.embedding_dim if isinstance(self.embedding_dim, int) else self.embedding_dim_range[1]
        )
        dim_per_feature = target_dim // n_features
        extra_dims = target_dim % n_features

        # 构建重构矩阵
        reconstructed = []

        for feat_idx in range(n_features):
            # 当前特征的嵌入维度
            current_dim = dim_per_feature
            if feat_idx < extra_dims:
                current_dim += 1

            # 提取当前特征的时序
            feature_series = data[:, feat_idx]

            # 相空间嵌入
            if len(feature_series) >= current_dim:
                time_delay = self.actual_time_delay if self.actual_time_delay is not None else (
                    self.time_delay if isinstance(self.time_delay, int) else 1
                )
                embedded = self.phase_space_embedding(
                    feature_series,
                    embedding_dim=current_dim,
                    time_delay=time_delay
                )
            else:
                # 信号太短，使用重复填充
                embedded = np.tile(feature_series.reshape(-1, 1),
                                  (1, current_dim))[:n_samples]

            # 应用动态权重
            weighted_embedded = embedded * weights[feat_idx]

            reconstructed.append(weighted_embedded)

        # 合并所有特征的嵌入
        if reconstructed:
            # 确保所有嵌入具有相同的样本数
            min_samples = min(r.shape[0] for r in reconstructed)
            reconstructed_aligned = [r[:min_samples] for r in reconstructed]

            # 水平拼接
            final_reconstruction = np.hstack(reconstructed_aligned)

            # 调整到目标维度
            if final_reconstruction.shape[1] > target_dim:
                final_reconstruction = final_reconstruction[:, :target_dim]
            elif final_reconstruction.shape[1] < target_dim:
                padding = np.zeros((final_reconstruction.shape[0],
                                   target_dim - final_reconstruction.shape[1]))
                final_reconstruction = np.hstack([final_reconstruction, padding])
        else:
            final_reconstruction = np.zeros((n_samples, target_dim))

        return final_reconstruction

    def fit_transform(self,
                     data: Union[np.ndarray, pd.DataFrame],
                     labels: Optional[np.ndarray] = None,
                     day_mask: Optional[Union[np.ndarray, List[bool]]] = None) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        主处理流程：学习权重并进行动态重构

        1. 对每个时间窗口构建邻域
        2. NCA优化学习权重
        3. 应用权重进行动态重构
        4. 返回重构后的特征矩阵

        Args:
            data: 输入数据 (n_samples x n_features)
            labels: 目标标签（可选，用于监督学习）

        Returns:
            Tuple[重构特征矩阵, 权重字典]
        """
        # 统一转换为 numpy 数组
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.asarray(data)

        original_n_samples, n_features = data_array.shape
        # Avoid non-ASCII text to prevent mojibake on some consoles
        logger.info("DPSR input: samples=%d, features=%d", original_n_samples, n_features)

        # 若未提供标签，使用下一时刻的数值作为伪标签
        if labels is None:
            # 滚动一步生成伪标签
            labels = np.roll(data_array[:, 0], -1)
            labels[-1] = labels[-2]  # 用倒数第二个补齐最后一个标签
        labels = np.asarray(labels)

        mask_array = self._prepare_day_mask(day_mask, original_n_samples)
        if mask_array is not None:
            valid_indices = np.where(mask_array)[0]
            if not valid_indices.size:
                target_dim = self._default_target_dim()
                logger.warning("DPSR day_mask has no true positions after filtering")
                zeros = np.zeros((original_n_samples, target_dim), dtype=np.float32)
                return zeros, {}
            data_array = data_array[valid_indices]
            labels = labels[valid_indices]
        else:
            valid_indices = np.arange(original_n_samples)

        n_samples = data_array.shape[0]
        index_map = valid_indices

        base_signal = data_array[:, 0]

        self.actual_time_delay = self._determine_time_delay(base_signal)
        self.actual_embedding_dim = self._determine_embedding_dim(base_signal)

        # 初始化输出
        target_dim = None
        reconstructed_features = None
        time_weights = {}

        # 分批处理，避免内存溢出
        batch_size = min(100, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # 进度跟踪
        process_start_time = time.time()
        last_loss = 0.0
        last_iter = 0

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_indices = np.arange(start_idx, end_idx)

            # 对批次中的每个时间点
            for local_idx, global_idx in enumerate(batch_indices):
                # 构建邻域
                neighborhood_data, neighbor_indices = self.construct_neighborhood(
                    data_array, global_idx
                )

                # 获取邻域标签
                neighborhood_labels = labels[neighbor_indices]

                # NCA优化学习权重
                if len(neighborhood_data) >= 3:  # 至少需要3个样本
                    try:
                        weights, opt_info = self.nca_optimization(
                            neighborhood_data,
                            neighborhood_labels
                        )
                        last_loss = opt_info['loss']
                        last_iter = opt_info['iter']
                    except Exception as e:
                        logger.warning(f"时刻{global_idx}的NCA优化失败: {e}")
                        weights = np.ones(n_features) / n_features
                else:
                    # 邻域太小，使用均匀权重
                    weights = np.ones(n_features) / n_features

                # 存储权重
                orig_idx = int(index_map[global_idx])
                time_weights[orig_idx] = weights

                current_data = data_array[global_idx:global_idx + 1]
                reconstructed = self.dynamic_reconstruction(current_data, weights)

        if reconstructed_features is None:
            reconstructed_features = np.zeros((n_samples, self.actual_embedding_dim or self.embedding_dim_range[1]))

        full_features = np.zeros(
            (original_n_samples, reconstructed_features.shape[1]),
            dtype=reconstructed_features.dtype,
        )
        full_features[index_map] = reconstructed_features[:len(index_map)]
        # 统计全局权重并标记为已拟合
        if time_weights:
            W = np.stack(list(time_weights.values()), axis=0)
            self.global_weights = W.mean(axis=0)
        else:
            # 若极端情况下没有任何时刻成功学习，使用均匀权重占位
            self.global_weights = np.ones(n_features) / max(n_features, 1)
        self.is_fitted = True

        logger.info("DPSR training done: output_shape=%s, weight_times=%d", tuple(full_features.shape), len(time_weights))
        return full_features, time_weights



    def transform(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 day_mask: Optional[Union[np.ndarray, List[bool]]] = None) -> np.ndarray:
        """
        使用已学习的权重进行转换

        Args:
            data: 输入数据

        Returns:
            重构的特征矩阵
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用fit_transform学习权重")

        # 转换为numpy数组
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.asarray(data)

        original_n_samples = data_array.shape[0]
        mask_array = self._prepare_day_mask(day_mask, original_n_samples)
        if mask_array is not None:
            valid_indices = np.where(mask_array)[0]
            if not valid_indices.size:
                target_dim = self.actual_embedding_dim or self._default_target_dim()
                return np.zeros((original_n_samples, target_dim), dtype=np.float32)
            data_array = data_array[valid_indices]
        else:
            valid_indices = np.arange(original_n_samples)

        n_samples = data_array.shape[0]

        # 按全局权重做动态重构
        reconstructed_features = self.dynamic_reconstruction(data_array, self.global_weights)

        # 长度对齐：不足填充，超出截断
        if reconstructed_features.shape[0] < n_samples:
            padding = np.zeros((n_samples - reconstructed_features.shape[0],
                               reconstructed_features.shape[1]))
            reconstructed_features = np.vstack([reconstructed_features, padding])
        elif reconstructed_features.shape[0] > n_samples:
            reconstructed_features = reconstructed_features[:n_samples]

        full_features = np.zeros((original_n_samples, reconstructed_features.shape[1]),
                                 dtype=reconstructed_features.dtype)
        full_features[valid_indices[:len(reconstructed_features)]] = reconstructed_features

        return full_features


    def save_weights(self, filepath: Union[str, Path]) -> None:
        """
        保存学习的权重

        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，无权重可保存")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        splits = getattr(self, 'last_split_sizes', {})
        weights_data = {
            'global_weights': self.global_weights.tolist(),
            'time_weights': {int(k): v.tolist() for k, v in self.weights.items()},
            'embedding_dim': self.embedding_dim,
            'neighborhood_size': self.neighborhood_size,
            'regularization': self.regularization,
            'split_sizes': splits
        }

        if filepath.suffix == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(weights_data, f, indent=2)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(weights_data, f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        logger.info(f"DPSR权重已保存: {filepath}")

    def load_weights(self, filepath: Union[str, Path]) -> None:
        """
        加载权重

        Args:
            filepath: 权重文件路径
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"权重文件不存在: {filepath}")

        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                weights_data = json.load(f)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                weights_data = pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        self.global_weights = np.array(weights_data['global_weights'])
        self.weights = {int(k): np.array(v) for k, v in weights_data['time_weights'].items()}
        self.embedding_dim = weights_data['embedding_dim']
        self.neighborhood_size = weights_data['neighborhood_size']
        self.regularization = weights_data['regularization']
        self.is_fitted = True
        self.last_split_sizes = weights_data.get('split_sizes', {})

        logger.info(f"DPSR权重已加载: {filepath}")


def test_dpsr():
    """测试DPSR模块"""
    # 创建模拟数据（9维：5个IMF + 4个气象特征）
    n_samples = 200
    n_features = 9

    np.random.seed(42)

    # 生成具有时序相关性的数据
    data = np.zeros((n_samples, n_features))
    for i in range(n_features):
        # 生成不同频率的正弦波 + 噪声
        freq = 0.1 * (i + 1)
        data[:, i] = np.sin(2 * np.pi * freq * np.arange(n_samples) / n_samples)
        data[:, i] += 0.1 * np.random.randn(n_samples)

    # 创建简单的标签（基于第一个特征）
    labels = np.roll(data[:, 0], -1) + 0.01 * np.random.randn(n_samples)

    # 初始化DPSR
    dpsr = DPSR(embedding_dim=30, neighborhood_size=20)

    # 执行动态重构
    print("开始DPSR处理...")
    reconstructed, weights = dpsr.fit_transform(data, labels)

    print(f"输入形状: {data.shape}")
    print(f"输出形状: {reconstructed.shape}")
    print(f"学习的权重数: {len(weights)}")
    print(f"平均权重: {dpsr.global_weights}")

    # 测试transform方法
    test_data = data[:50]  # 使用前50个样本测试
    transformed = dpsr.transform(test_data)
    print(f"Transform测试: 输入{test_data.shape} -> 输出{transformed.shape}")

    return dpsr, reconstructed


if __name__ == "__main__":
    # 运行测试
    test_dpsr()
