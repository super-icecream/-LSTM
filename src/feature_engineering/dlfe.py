"""
动态局部特征嵌入模块 (DLFE)
功能：基于ADMM算法的流形学习，实现动态特征降维
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
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def diagnose_matrix_sparsity(matrix: np.ndarray, name: str = "Matrix") -> Dict:
    """
    Diagnose matrix sparsity and estimate memory impact.

    Args:
        matrix: Matrix to inspect.
        name: Matrix name for logging.

    Returns:
        Dictionary containing diagnostic metrics.
    """
    n, m = matrix.shape
    total_elements = n * m

    nonzero_mask = np.abs(matrix) > 1e-10
    nonzero_elements = np.count_nonzero(nonzero_mask)

    sparsity = 1.0 - (nonzero_elements / total_elements)

    dense_memory_gb = (total_elements * 8) / (1024 ** 3)
    sparse_memory_mb = (nonzero_elements * 12) / (1024 ** 2)

    savings_ratio = (dense_memory_gb * 1024) / sparse_memory_mb if sparse_memory_mb > 0 else float("inf")

    nonzero_per_row = np.sum(nonzero_mask, axis=1)
    avg_nonzero_per_row = float(np.mean(nonzero_per_row))
    max_nonzero_per_row = int(np.max(nonzero_per_row))
    min_nonzero_per_row = int(np.min(nonzero_per_row))

    result = {
        "shape": (n, m),
        "total_elements": int(total_elements),
        "nonzero_elements": int(nonzero_elements),
        "sparsity": float(sparsity),
        "avg_nonzero_per_row": avg_nonzero_per_row,
        "max_nonzero_per_row": max_nonzero_per_row,
        "min_nonzero_per_row": min_nonzero_per_row,
        "dense_memory_gb": float(dense_memory_gb),
        "sparse_memory_mb": float(sparse_memory_mb),
        "savings_ratio": float(savings_ratio),
    }

    diag_logger = logging.getLogger(__name__)
    diag_logger.info("=" * 70)
    diag_logger.info(f"📊 {name} 稀疏性诊断报告")
    diag_logger.info("=" * 70)
    diag_logger.info(f"矩阵形状: {n} × {m}")
    diag_logger.info(f"总元素数: {total_elements:,}")
    diag_logger.info(f"非零元素: {nonzero_elements:,}")
    diag_logger.info(f"稀疏度: {sparsity * 100:.2f}%")
    diag_logger.info(f"每行非零元素: 平均={avg_nonzero_per_row:.1f}, 最大={max_nonzero_per_row}, 最小={min_nonzero_per_row}")
    diag_logger.info("")
    diag_logger.info("💾 内存占用估算:")
    diag_logger.info(f"  密集存储 (float64): {dense_memory_gb:.2f} GB")
    diag_logger.info(f"  稀疏存储 (COO):     {sparse_memory_mb:.2f} MB")
    diag_logger.info(f"  节省比例:           {savings_ratio:.1f}x")
    diag_logger.info("")

    should_use_sparse = sparsity > 0.9 and dense_memory_gb > 1.0
    if should_use_sparse:
        diag_logger.info("✅ 推荐使用稀疏矩阵优化！")
        diag_logger.info(f"   理由: 稀疏度={sparsity * 100:.1f}%, 密集存储需要={dense_memory_gb:.2f}GB")
    else:
        diag_logger.info("⚠️  稀疏矩阵优化收益不明显")
        diag_logger.info(f"   理由: 稀疏度={sparsity * 100:.1f}%, 密集存储仅需={dense_memory_gb:.2f}GB")

    diag_logger.info("=" * 70)

    result["should_use_sparse"] = should_use_sparse
    return result


class DLFE:
    """
    Dynamic Local Feature Embedding (DLFE)

    Manifold-learning dimensionality reduction solved via ADMM while
    preserving local neighbourhood structure in DPSR outputs.

    Attributes:
        target_dim (int): target embedding dimension (default 30)
        sigma (float): RBF kernel parameter (fixed 1.0 per Gu et al. 2025)
        alpha (float): ADMM balance parameter (2^-10 approx 0.00098)
        beta (float): ADMM regularisation parameter (0.1, sparsity)
        max_iter (int): ADMM maximum iterations
        tol (float): convergence tolerance
    """

    def __init__(self,
                 target_dim: int = 30,
                 sigma: float = 1.0,
                 alpha: float = 0.0009765625,
                 beta: float = 0.1,
                 max_iter: int = 100,
                 tol: float = 1e-5,
                 device: str = "auto",
                 use_float32_eigh: bool = False,
                 use_sparse_matrix: bool = True):
        """
        初始化DLFE

        Args:
            target_dim: target embedding dimension (30).
            sigma: RBF kernel width (fixed 1.0 per Gu et al. 2025).
            alpha: ADMM balance parameter (2^-10 approx 0.00098).
            beta: ADMM sparsity regularisation parameter (0.1).
            max_iter: ADMM maximum iterations.
            tol: convergence tolerance.
            device: compute device ('auto', 'cuda', 'cpu').
            use_float32_eigh: 是否启用 GPU 特征分解（lobpcg），大型稀疏矩阵时可能触发 OOM。
            use_sparse_matrix: 是否启用稀疏矩阵优化（CPU eigsh），稀疏度>95%时推荐。
        """
        self.target_dim = target_dim
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.use_float32_eigh = use_float32_eigh
        self.use_sparse_matrix = use_sparse_matrix

        # Device management (GPU acceleration path)
        self.use_gpu = False
        self.device = "cpu"
        self._torch = torch if TORCH_AVAILABLE else None
        self._torch_device = None

        if TORCH_AVAILABLE:
            selected_device = device
            if selected_device not in ("auto", "cpu", "cuda"):
                logger.warning("未知设备类型%s，已回退到自动检测模式", selected_device)
                selected_device = "auto"

            if selected_device == "auto":
                selected_device = DEFAULT_DEVICE

            if selected_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA不可用，DLFE将使用CPU模式执行计算")
                selected_device = "cpu"

            self.device = selected_device
            self.use_gpu = selected_device == "cuda"
            self._torch_device = torch.device(selected_device)

            if self.use_gpu:
                try:
                    device_name = torch.cuda.get_device_name(self._torch_device)
                except Exception:  # pragma: no cover - 驱动实现差异
                    device_name = "CUDA"
                logger.info("DLFE启用GPU加速：%s", device_name)
            else:
                logger.info("DLFE运行在CPU模式 (PyTorch检测到，但未启用CUDA)")
        else:
            if device == "cuda":
                logger.warning("未安装PyTorch，无法启用CUDA；DLFE将使用CPU模式")
            logger.info("DLFE运行在CPU模式 (PyTorch不可用)")

        # 映射矩阵
        self.mapping_matrix = None  # A矩阵
        self.is_fitted = False

        # 优化历史
        self.optimization_history = {
            'objective': [],
            'constraint_violation': [],
            'relative_change': [],
            'iterations': 0
        }
        self._sparsity_diagnosis: Optional[Dict] = None

        logger.info(f"DLFE初始化: 目标维度={target_dim}, σ={sigma}, α={alpha}, β={beta}")

    def build_similarity_matrix(self,
                              X: np.ndarray,
                              weights: Optional[np.ndarray] = None,
                              k_neighbors: Optional[int] = None) -> np.ndarray:
        """
        构建相似度矩阵Q

        使用高斯核计算样本间的相似度：
        Qij = exp(-dw(si-sj)²/σ²) if sj ∈ Si(si)
              0                    otherwise

        Args:
            X: 输入数据 (n_samples x n_features)
            weights: DPSR提供的特征权重（可选）
            k_neighbors: k近邻数量（如果None则使用全连接）

        Returns:
            相似度矩阵Q (n_samples x n_samples)
        """
        if self.use_gpu and self._torch is not None:
            return self._build_similarity_matrix_gpu(X, weights, k_neighbors)

        n_samples, n_features = X.shape

        # 如果提供了权重，进行加权
        if weights is not None:
            if len(weights.shape) == 1:
                # 一维权重，广播到所有特征
                X_weighted = X * np.sqrt(weights.reshape(1, -1))
            else:
                # 多维权重（每个样本不同权重）
                X_weighted = X * np.sqrt(weights)
        else:
            X_weighted = X

        # 计算欧氏距离矩阵
        # 使用广播计算所有点对的距离
        # ||xi - xj||² = ||xi||² + ||xj||² - 2*xi·xj
        X_norm = np.sum(X_weighted ** 2, axis=1, keepdims=True)
        distances_squared = X_norm + X_norm.T - 2 * np.dot(X_weighted, X_weighted.T)

        # 数值稳定性：确保距离非负
        distances_squared = np.maximum(distances_squared, 0)

        # 计算高斯相似度
        # sigma fixed to 1.0 (Gu et al., 2025)
        Q = np.exp(-distances_squared / (2 * self.sigma ** 2))

        # 如果指定k近邻，只保留k个最近邻
        if k_neighbors is not None and k_neighbors < n_samples - 1:
            for i in range(n_samples):
                # 找到第k+1近的距离作为阈值
                sorted_indices = np.argsort(distances_squared[i])
                # 保留自己和k个最近邻
                threshold_idx = min(k_neighbors + 1, n_samples)
                keep_indices = sorted_indices[:threshold_idx]

                # 创建掩码
                mask = np.zeros(n_samples, dtype=bool)
                mask[keep_indices] = True

                # 应用掩码
                Q[i, ~mask] = 0
                Q[~mask, i] = 0

        # 对角线设为0（不包括自相似）
        np.fill_diagonal(Q, 0)

        # 对称化（确保数值对称性）
        Q = (Q + Q.T) / 2

        logger.debug(f"相似度矩阵构建完成: 形状={Q.shape}, 非零元素比例={np.count_nonzero(Q) / Q.size:.2%}")

        return Q

    def _build_similarity_matrix_gpu(self,
                                      X: np.ndarray,
                                      weights: Optional[np.ndarray] = None,
                                      k_neighbors: Optional[int] = None) -> np.ndarray:
        """GPU版本相似度矩阵构建，数学上等价于CPU实现。"""
        if not self.use_gpu or self._torch is None:
            raise RuntimeError("GPU路径不可用，无法调用_build_similarity_matrix_gpu")

        torch_module = self._torch
        device = self._torch_device
        n_samples = X.shape[0]
        batch_size = min(5000, n_samples)
        gaussian_denominator = 2.0 * (self.sigma ** 2)

        Q_batches: List[np.ndarray] = []

        with torch_module.no_grad():
            X_gpu = torch_module.as_tensor(X, dtype=torch_module.double, device=device)

            if weights is not None:
                if weights.ndim == 1:
                    weights_gpu = torch_module.as_tensor(weights, dtype=torch_module.double, device=device)
                    X_weighted = X_gpu * torch_module.sqrt(weights_gpu.view(1, -1))
                else:
                    weights_gpu = torch_module.as_tensor(weights, dtype=torch_module.double, device=device)
                    X_weighted = X_gpu * torch_module.sqrt(weights_gpu)
            else:
                X_weighted = X_gpu

            norm_all = torch_module.sum(X_weighted ** 2, dim=1, keepdim=True).T

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_weighted[start_idx:end_idx]
                norm_batch = torch_module.sum(X_batch ** 2, dim=1, keepdim=True)

                distances_sq = norm_batch + norm_all - 2.0 * torch_module.mm(X_batch, X_weighted.T)
                distances_sq = torch_module.clamp(distances_sq, min=0.0)

                Q_batch = torch_module.exp(-distances_sq / gaussian_denominator)

                if k_neighbors is not None and k_neighbors < n_samples - 1:
                    threshold = min(k_neighbors + 1, n_samples)
                    for local_idx in range(end_idx - start_idx):
                        sorted_indices = torch_module.argsort(distances_sq[local_idx])
                        keep_indices = sorted_indices[:threshold]
                        mask = torch_module.zeros(n_samples, dtype=torch_module.bool, device=device)
                        mask[keep_indices] = True
                        Q_batch[local_idx, ~mask] = 0.0

                Q_batches.append(Q_batch.cpu().numpy())

                if self.use_gpu:
                    del X_batch, norm_batch, distances_sq, Q_batch
                    torch_module.cuda.empty_cache()

        Q = np.vstack(Q_batches)
        np.fill_diagonal(Q, 0.0)
        Q = (Q + Q.T) / 2.0

        if self.use_gpu:
            del X_gpu, X_weighted, norm_all
            if 'weights_gpu' in locals():
                del weights_gpu
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

        return Q

    def construct_laplacian(self, Q: np.ndarray) -> np.ndarray:
        """
        构建图拉普拉斯矩阵

        L = D - Q
        其中D为度矩阵，Dii = ΣQij

        Args:
            Q: 相似度矩阵

        Returns:
            拉普拉斯矩阵L
        """
        if self.use_gpu and self._torch is not None:
            return self._construct_laplacian_gpu(Q)

        # 计算度矩阵
        degrees = np.sum(Q, axis=1)
        D = np.diag(degrees)

        # 计算拉普拉斯矩阵
        L = D - Q

        # 归一化拉普拉斯（可选，提高数值稳定性）
        # L_norm = D^(-1/2) * L * D^(-1/2)
        # 避免除零
        degrees_sqrt_inv = np.zeros_like(degrees)
        non_zero_mask = degrees > 1e-10
        degrees_sqrt_inv[non_zero_mask] = 1.0 / np.sqrt(degrees[non_zero_mask])

        D_sqrt_inv = np.diag(degrees_sqrt_inv)
        L_normalized = D_sqrt_inv @ L @ D_sqrt_inv

        # 确保对称性
        L_normalized = (L_normalized + L_normalized.T) / 2

        return L_normalized

    def _construct_laplacian_gpu(self, Q: np.ndarray) -> np.ndarray:
        """GPU版本拉普拉斯构建，使用广播代替显式对角矩阵。"""
        if not self.use_gpu or self._torch is None:
            raise RuntimeError("GPU路径不可用，无法调用_construct_laplacian_gpu")

        torch_module = self._torch
        device = self._torch_device
        n_samples = Q.shape[0]
        chunk_size = min(5000, n_samples)

        with torch_module.no_grad():
            degrees = torch_module.from_numpy(Q.sum(axis=1)).to(device=device, dtype=torch_module.double)
            degrees_sqrt_inv = torch_module.zeros_like(degrees)
            non_zero_mask = degrees > 1e-10
            degrees_sqrt_inv[non_zero_mask] = torch_module.pow(degrees[non_zero_mask], -0.5)
            col_scale = degrees_sqrt_inv.unsqueeze(0)

            L_normalized_chunks: List[np.ndarray] = []

            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)
                Q_chunk = torch_module.from_numpy(Q[start_idx:end_idx]).to(device=device, dtype=torch_module.double)

                L_chunk = -Q_chunk
                row_indices = torch_module.arange(end_idx - start_idx, device=device)
                col_indices = torch_module.arange(start_idx, end_idx, device=device)
                L_chunk[row_indices, col_indices] += degrees[start_idx:end_idx]

                row_scale = degrees_sqrt_inv[start_idx:end_idx].unsqueeze(1)
                L_normalized_chunk = row_scale * L_chunk * col_scale

                L_normalized_chunks.append(L_normalized_chunk.cpu().numpy())

                del Q_chunk, L_chunk, L_normalized_chunk, row_scale, row_indices, col_indices
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()

            L_normalized = np.vstack(L_normalized_chunks)
            L_normalized = (L_normalized + L_normalized.T) / 2.0

            del degrees, degrees_sqrt_inv, col_scale, L_normalized_chunks
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

        return L_normalized

    def _admm_optimization_gpu(self,
                               X: np.ndarray,
                               L: np.ndarray) -> np.ndarray:
        """
        GPU加速版 ADMM 计算，采用分块与稀疏线性算子避免显存暴涨。
        """
        if not self.use_gpu or self._torch is None:
            raise RuntimeError("GPU路径不可用，无法调用_admm_optimization_gpu")

        torch_module = self._torch
        device = self._torch_device

        n_samples, n_features = X.shape
        d = self.target_dim

        alpha = self.alpha
        beta = self.beta
        tol = self.tol
        max_iter = self.max_iter

        sample_chunk = min(5000, n_samples)

        # ========== 新增：早停检测函数 ==========
        def check_early_stopping(iter_idx: int,
                                 relative_change: float,
                                 obj_history: List[float],
                                 rel_history: List[float]) -> Tuple[bool, str]:
            """智能早停检测"""
            if relative_change < tol:
                return True, "F收敛"

            if iter_idx >= 5 and len(obj_history) >= 5:
                recent = obj_history[-5:]
                improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)
                if improvement < 1e-4:
                    return True, "目标函数停滞"

            if iter_idx >= 3 and len(rel_history) >= 3:
                if all(history_value < tol * 10 for history_value in rel_history[-3:]):
                    return True, "F振荡收敛"

            if iter_idx >= 20 and relative_change < tol * 50:
                return True, "快速收敛"

            return False, ""

        # ========== 新增：进度条更新函数 ==========
        def update_progress(iter_idx: int,
                            max_iterations: int,
                            objective: float,
                            rel_change: float,
                            phase: str = "计算中") -> None:
            """单行进度条更新"""
            if max_iterations <= 0:
                progress = 100.0
                filled = 30
            else:
                progress = (iter_idx + 1) / max_iterations * 100
                filled = min(30, int(30 * (iter_idx + 1) / max_iterations))

            bar = "█" * filled + "░" * (30 - filled)
            sys.stdout.write(
                f'\r  ADMM优化 [{bar}] {progress:.1f}% | '
                f'迭代:{iter_idx + 1}/{max_iterations} | '
                f'目标值:{objective:.4f} | '
                f'相对变化:{rel_change:.2e} | '
                f'{phase}'
            )
            sys.stdout.flush()

        logger.info("开始ADMM迭代（早停+进度条）...")
        start_time = time.time()

        X_gpu = torch_module.as_tensor(X, dtype=torch_module.double, device=device)

        A = torch_module.zeros((n_features, d), dtype=torch_module.double, device=device)
        F = torch_module.randn((n_samples, d), dtype=torch_module.double, device=device)
        q, _ = torch_module.linalg.qr(F, mode='reduced')
        F = q[:, :d]

        XTX = torch_module.zeros((n_features, n_features), dtype=torch_module.double, device=device)
        for start_idx in range(0, n_samples, sample_chunk):
            end_idx = min(start_idx + sample_chunk, n_samples)
            X_chunk = X_gpu[start_idx:end_idx]
            XTX = XTX + X_chunk.T @ X_chunk
        XTX_diag = torch_module.diag(XTX) + 1e-10

        optimization_history = {
            'objective': [],
            'constraint_violation': [],
            'relative_change': [],
            'iterations': 0
        }

        for iter_idx in range(max_iter):
            F_old = F.clone()

            if self.use_sparse_matrix:
                phase_label = "特征分解(CPU-稀疏)"
            elif self.use_float32_eigh and torch_module.cuda.is_available():
                phase_label = "特征分解(GPU-lobpcg)"
            else:
                phase_label = "特征分解(CPU)"

            update_progress(
                iter_idx,
                max_iter,
                optimization_history['objective'][-1] if optimization_history['objective'] else 0.0,
                optimization_history['relative_change'][-1] if optimization_history['relative_change'] else 0.0,
                phase_label
            )

            if self.use_sparse_matrix:
                try:
                    if iter_idx == 0:
                        logger.info(f"使用稀疏特征分解（CPU eigsh）计算 {d} 个最小特征值...")

                    from scipy.sparse import csr_matrix
                    import scipy.sparse

                    L_sparse = csr_matrix(L)
                    M_sparse = L_sparse + alpha * scipy.sparse.eye(n_samples, format="csr")

                    eigenvalues, eigenvectors = eigsh(
                        M_sparse,
                        k=d,
                        which="SM",
                        maxiter=1000,
                        tol=1e-6,
                    )

                    F = torch_module.from_numpy(eigenvectors).to(device=device, dtype=torch_module.float64)
                    if iter_idx == 0:
                        print("✓ 稀疏特征分解（CPU eigsh）完成，结果已传输到 GPU", flush=True)

                    del L_sparse, M_sparse, eigenvalues, eigenvectors

                except Exception as e:
                    if iter_idx == 0:
                        logger.warning(f"稀疏特征分解失败: {type(e).__name__}: {e}")
                        logger.info("回退到 CPU 密集特征分解...")

                    M_cpu = L + alpha * np.eye(n_samples, dtype=np.float64)
                    eigenvalues, eigenvectors = eigsh(M_cpu, k=d, which='SM')
                    F = torch_module.from_numpy(eigenvectors).to(device=device, dtype=torch_module.double)
                    del M_cpu, eigenvalues, eigenvectors
                    if iter_idx == 0:
                        logger.info("CPU eigsh 执行成功")

            elif self.use_float32_eigh and torch_module.cuda.is_available():
                try:
                    logger.info(f"尝试使用 GPU lobpcg 计算 {d} 个最小特征值...")

                    M_gpu = torch_module.from_numpy(L).to(device=device, dtype=torch_module.float64)
                    M_gpu = M_gpu + alpha * torch_module.eye(n_samples, dtype=torch_module.float64, device=device)

                    if torch_module.isnan(M_gpu).any() or torch_module.isinf(M_gpu).any():
                        raise ValueError("M_gpu 包含 NaN 或 Inf，无法进行特征分解")

                    eigenvalues, eigenvectors = torch_module.lobpcg(
                        M_gpu,
                        k=d,
                        largest=False,
                        niter=300,
                        tol=1e-6,
                        method='ortho'
                    )

                    if eigenvectors.shape[1] < d:
                        raise RuntimeError(f"lobpcg 只收敛了 {eigenvectors.shape[1]}/{d} 个特征向量")

                    orth_check = eigenvectors.T @ eigenvectors
                    orth_error = torch_module.norm(
                        orth_check - torch_module.eye(d, dtype=torch_module.float64, device=device)
                    ).item()

                    if orth_error > 1e-3:
                        logger.warning(f"GPU lobpcg 正交性误差: {orth_error:.2e}")
                    else:
                        logger.debug(f"GPU lobpcg 正交性验证通过: {orth_error:.2e}")

                    F = eigenvectors
                    logger.info(f"GPU lobpcg 成功，正交性误差: {orth_error:.2e}")

                    del M_gpu, eigenvalues
                    if torch_module.cuda.is_available():
                        torch_module.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"GPU lobpcg 失败: {type(e).__name__}: {e}")
                    logger.info("回退到 CPU eigsh 方法...")

                    if 'M_gpu' in locals():
                        del M_gpu
                    if torch_module.cuda.is_available():
                        torch_module.cuda.empty_cache()

                    M_cpu = L + alpha * np.eye(n_samples, dtype=np.float64)
                    eigenvalues, eigenvectors = eigsh(M_cpu, k=d, which='SM')
                    F = torch_module.from_numpy(eigenvectors).to(device=device, dtype=torch_module.double)
                    del M_cpu, eigenvalues, eigenvectors
                    logger.info("CPU eigsh 执行成功")

            else:
                M_cpu = L + alpha * np.eye(n_samples, dtype=np.float64)
                eigenvalues, eigenvectors = eigsh(M_cpu, k=d, which='SM')
                F = torch_module.from_numpy(eigenvectors).to(device=device, dtype=torch_module.double)
                del M_cpu, eigenvalues, eigenvectors

            if torch_module.count_nonzero(A).item() > 0:
                XA = torch_module.zeros((n_samples, d), dtype=torch_module.double, device=device)
                for start_idx in range(0, n_samples, sample_chunk):
                    end_idx = min(start_idx + sample_chunk, n_samples)
                    XA[start_idx:end_idx] = X_gpu[start_idx:end_idx] @ A
                correction = (self.alpha * XA) / (1.0 + self.alpha)
                F = F + correction
                q, _ = torch_module.linalg.qr(F, mode='reduced')
                F = q[:, :d]
                del XA, correction, q
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()

            update_progress(
                iter_idx,
                max_iter,
                optimization_history['objective'][-1] if optimization_history['objective'] else 0.0,
                optimization_history['relative_change'][-1] if optimization_history['relative_change'] else 0.0,
                "更新A矩阵"
            )

            XTF = torch_module.zeros((n_features, d), dtype=torch_module.double, device=device)
            for start_idx in range(0, n_samples, sample_chunk):
                end_idx = min(start_idx + sample_chunk, n_samples)
                X_chunk = X_gpu[start_idx:end_idx]
                F_chunk = F[start_idx:end_idx]
                XTF = XTF + X_chunk.T @ F_chunk

            for j in range(d):
                residual = XTF[:, j] if j == 0 else XTF[:, j] - XTX @ A[:, j]

                norm_residual = torch_module.linalg.norm(residual)
                threshold = beta / (2 * alpha)
                if norm_residual > threshold:
                    shrink = 1 - threshold / (norm_residual + 1e-12)
                    A[:, j] = shrink * residual / XTX_diag
                else:
                    A[:, j].zero_()

            update_progress(
                iter_idx,
                max_iter,
                optimization_history['objective'][-1] if optimization_history['objective'] else 0.0,
                optimization_history['relative_change'][-1] if optimization_history['relative_change'] else 0.0,
                "收敛判断"
            )

            F_change = torch_module.linalg.norm(F - F_old, ord='fro').item()
            F_reference = torch_module.linalg.norm(F_old, ord='fro').item() + 1e-10
            relative_change = F_change / F_reference

            trace_term = 0.0
            laplacian_chunk = min(2048, n_samples)
            for row_start in range(0, n_samples, laplacian_chunk):
                row_end = min(row_start + laplacian_chunk, n_samples)
                L_chunk = torch_module.from_numpy(L[row_start:row_end]).to(device=device, dtype=torch_module.double)
                LF_chunk = L_chunk @ F
                trace_term += torch_module.trace(F[row_start:row_end].T @ LF_chunk).item()
                del L_chunk, LF_chunk
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()

            XA_full = torch_module.zeros((n_samples, d), dtype=torch_module.double, device=device)
            for start_idx in range(0, n_samples, sample_chunk):
                end_idx = min(start_idx + sample_chunk, n_samples)
                XA_full[start_idx:end_idx] = X_gpu[start_idx:end_idx] @ A

            reconstruction_term = alpha * (torch_module.linalg.norm(XA_full - F, ord='fro') ** 2).item()
            regularizer = beta * torch_module.sum(torch_module.linalg.norm(A, dim=0)).item()
            objective = trace_term + reconstruction_term + regularizer

            optimization_history['objective'].append(objective)
            optimization_history['constraint_violation'].append(F_change)
            optimization_history['relative_change'].append(relative_change)
            optimization_history['iterations'] = iter_idx + 1

            update_progress(iter_idx, max_iter, objective, relative_change, "检查收敛")

            should_stop, reason = check_early_stopping(
                iter_idx,
                relative_change,
                optimization_history['objective'],
                optimization_history['relative_change']
            )

            if should_stop:
                elapsed = time.time() - start_time
                sys.stdout.write('\n')
                logger.info(f"  ✅ ADMM提前收敛！原因: {reason}")
                logger.info(f"     实际迭代: {iter_idx + 1}/{max_iter} | 耗时: {elapsed / 60:.1f}分钟")
                del XA_full, XTF
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                break

            del XA_full, XTF

            if torch_module.cuda.is_available() and (iter_idx + 1) % 10 == 0:
                torch_module.cuda.empty_cache()
        else:
            elapsed = time.time() - start_time
            sys.stdout.write('\n')
            logger.info(f"  ⚠️ ADMM达到最大迭代次数{max_iter}")
            logger.info(f"     总耗时: {elapsed / 60:.1f}分钟")

        A_cpu = A.detach().cpu().numpy()
        del X_gpu, A, F, XTX
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

        self.optimization_history = optimization_history
        return A_cpu

    def admm_optimization(self,
                         X: np.ndarray,
                         L: np.ndarray) -> np.ndarray:
        """
        ADMM算法求解映射矩阵

        优化目标：
        min tr(F^T*L*F) + α||ZA-F||²_F + β||A||_2,1
        s.t. F^T*F = I

        迭代步骤：
        1. F子问题：固定A，更新F
        2. A子问题：固定F，更新A
        3. 收敛判断：||F(k+1)-F(k)||_F < ε

        Args:
            X: 输入数据矩阵 Z (n_samples x n_features)
            L: 拉普拉斯矩阵 (n_samples x n_samples)

        Returns:
            映射矩阵A (n_features x target_dim)
        """
        n_samples, n_features = X.shape
        d = self.target_dim

        self.optimization_history = {
            'objective': [],
            'constraint_violation': [],
            'relative_change': [],
            'iterations': 0
        }

        # 初始化
        A = np.zeros((n_features, d))  # 映射矩阵
        F = np.random.randn(n_samples, d)  # 低维表示
        F = F / np.linalg.norm(F, axis=0, keepdims=True)  # 列归一化

        # ADMM迭代
        for iter_idx in range(self.max_iter):
            F_old = F.copy()

            # ========== Step 1: 更新F（固定A） ==========
            # 优化目标：min tr(F^T*L*F) + α||X*A - F||²_F
            # 等价于：min tr(F^T*(L + αI)*F) - 2α*tr(F^T*X*A)

            # 构造矩阵M = L + αI
            M = L + self.alpha * np.eye(n_samples)

            # 目标：F^T*M*F - 2α*F^T*X*A
            # 使用特征分解求解
            if n_features < 500:  # 小规模问题，直接求解
                # 计算M的特征分解
                eigenvalues, eigenvectors = eigh(M)

                # 选择最小的d个特征值对应的特征向量
                idx = np.argsort(eigenvalues)[:d]
                F = eigenvectors[:, idx]
            else:  # 大规模问题，使用稀疏求解
                # 使用Lanczos方法求解最小特征值
                eigenvalues, eigenvectors = eigsh(M, k=d, which='SM')
                F = eigenvectors

            # 如果A不为零，需要考虑X*A项
            if np.any(A != 0):
                XA = X @ A
                # 修正F使其更接近XA
                F = F + self.alpha * XA / (1 + self.alpha)
                # 重新正交化
                U, S, Vt = np.linalg.svd(F, full_matrices=False)
                F = U @ Vt

            # ========== Step 2: 更新A（固定F） ==========
            # 优化目标：min α||X*A - F||²_F + β||A||_2,1
            # 这是一个L2,1正则化的最小二乘问题

            # 计算X^T*X
            XTX = X.T @ X
            XTF = X.T @ F

            # 使用软阈值算子求解
            for j in range(d):
                # 对每一列单独求解
                # 梯度：2α*X^T*(X*aj - fj)
                # L2,1正则化的近端算子

                # 计算残差
                if j == 0:
                    residual = XTF[:, j]
                else:
                    residual = XTF[:, j] - XTX @ A[:, j]

                # 软阈值
                norm_residual = np.linalg.norm(residual)
                if norm_residual > self.beta / (2 * self.alpha):
                    A[:, j] = (1 - self.beta / (2 * self.alpha * norm_residual)) * \
                             residual / (np.diagonal(XTX) + 1e-10)
                else:
                    A[:, j] = 0

            # ========== Step 3: 收敛判断 ==========
            F_change = np.linalg.norm(F - F_old, 'fro')
            relative_change = F_change / (np.linalg.norm(F_old, 'fro') + 1e-10)

            # 计算目标函数值
            obj_value = np.trace(F.T @ L @ F) + \
                       self.alpha * np.linalg.norm(X @ A - F, 'fro') ** 2 + \
                       self.beta * np.sum(np.linalg.norm(A, axis=0))

            self.optimization_history['objective'].append(obj_value)
            self.optimization_history['constraint_violation'].append(F_change)
            self.optimization_history['relative_change'].append(relative_change)
            self.optimization_history['iterations'] = iter_idx + 1

            # 日志输出（每10次迭代）
            if (iter_idx + 1) % 10 == 0:
                logger.debug(f"ADMM迭代{iter_idx + 1}: 目标值={obj_value:.4f}, "
                           f"相对变化={relative_change:.6f}")

            # 收敛判断
            if relative_change < self.tol:
                logger.info(f"ADMM收敛于第{iter_idx + 1}次迭代")
                break

        else:
            logger.warning(f"ADMM达到最大迭代次数{self.max_iter}，可能未完全收敛")
            self.optimization_history['iterations'] = self.max_iter

        return A

    def fit(self,
            X_train: Union[np.ndarray, pd.DataFrame],
            dpsr_weights: Optional[Union[np.ndarray, Dict]] = None,
            day_mask: Optional[Union[np.ndarray, List[bool]]] = None) -> 'DLFE':
        """
        ??????????? A?

        Args:
            X_train: ???? (n_samples x n_features)
            dpsr_weights: ?? DPSR ?????
            day_mask: ?????True ??????

        Returns:
            self
        """
        if isinstance(X_train, pd.DataFrame):
            X = X_train.values
        else:
            X = np.asarray(X_train)

        n_samples, n_features = X.shape
        mask_array = None
        if day_mask is not None:
            mask_array = np.asarray(day_mask, dtype=bool).reshape(-1)
            if mask_array.shape[0] != n_samples:
                raise ValueError(
                    f"DLFE day_mask ??({mask_array.shape[0]}) ????({n_samples}) ???"
                )
            valid_idx = np.where(mask_array)[0]
            if not valid_idx.size:
                raise ValueError("DLFE fit: day_mask ??????????")
            X = X[valid_idx]
            n_samples = X.shape[0]
        else:
            valid_idx = None

        logger.info(f"??DLFE??: {n_samples}??, {n_features}?? -> {self.target_dim}?")

        if dpsr_weights is not None:
            if isinstance(dpsr_weights, dict):
                if valid_idx is not None:
                    weights_list = [dpsr_weights[idx] for idx in valid_idx if idx in dpsr_weights]
                else:
                    weights_list = list(dpsr_weights.values())
                avg_weights = np.mean(weights_list, axis=0) if weights_list else None
            else:
                avg_weights = dpsr_weights
        else:
            avg_weights = None

        logger.info("???????...")
        Q = self.build_similarity_matrix(X, weights=avg_weights, k_neighbors=min(50, n_samples - 1))

        logger.info("????????...")
        L = self.construct_laplacian(Q)

        logger.info("\n?????????...")
        Q_diagnosis = diagnose_matrix_sparsity(Q, "????? Q")
        L_diagnosis = diagnose_matrix_sparsity(L, "?????? L")
        L_diagnosis = diagnose_matrix_sparsity(L, "?????? L")
        self._sparsity_diagnosis = {
            "Q": Q_diagnosis,
            "L": L_diagnosis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if L_diagnosis["should_use_sparse"]:
            logger.warning("??  ????????? L ?????")
            logger.warning("    ???????????? %.2f GB GPU ???", L_diagnosis["dense_memory_gb"])
            logger.warning("    ????????????? %.2f MB?", L_diagnosis["sparse_memory_mb"])
            logger.warning("    ??? %.1fx ???", L_diagnosis["savings_ratio"])
            logger.warning("    ??? %.1fx ???", L_diagnosis["savings_ratio"])
        if self.use_gpu and self._torch is not None:
            import gc

            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
                allocated = self._torch.cuda.memory_allocated() / 1024 ** 3
                logger.info("? ???????GPU - ???: %.2f GB", allocated)

        logger.info("??ADMM??...")
        if self.use_gpu and self._torch is not None:
            logger.info("DLFE ??GPU????ADMM??")
            self.mapping_matrix = self._admm_optimization_gpu(X, L)
        else:
            logger.info("DLFE ??CPU????ADMM??")
            self.mapping_matrix = self.admm_optimization(X, L)

        self.is_fitted = True

        logger.info(f"DLFE???????????: {self.mapping_matrix.shape}")

        return self
    def transform(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 day_mask: Optional[Union[np.ndarray, List[bool]]] = None) -> np.ndarray:
        """
        ?????????????????
        """
        if not self.is_fitted:
            raise RuntimeError("????fit????????")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        original_n_samples = X_array.shape[0]
        mask_array = None
        if day_mask is not None:
            mask_array = np.asarray(day_mask, dtype=bool).reshape(-1)
            if mask_array.shape[0] != original_n_samples:
                raise ValueError(
                    f"DLFE day_mask ??({mask_array.shape[0]}) ????({original_n_samples}) ???"
                )
            valid_idx = np.where(mask_array)[0]
            if not valid_idx.size:
                return np.zeros((original_n_samples, self.target_dim), dtype=np.float32)
            X_array = X_array[valid_idx]
        else:
            valid_idx = np.arange(original_n_samples)

        if X_array.shape[1] != self.mapping_matrix.shape[0]:
            raise ValueError(f"???????: ??{X_array.shape[1]}?, ??{self.mapping_matrix.shape[0]}?")

        F = X_array @ self.mapping_matrix
        full_F = np.zeros((original_n_samples, F.shape[1]), dtype=F.dtype)
        full_F[valid_idx[:F.shape[0]]] = F[:len(valid_idx)]

        return full_F
    def fit_transform(self,
                     X_train: Union[np.ndarray, pd.DataFrame],
                     dpsr_weights: Optional[Union[np.ndarray, Dict]] = None,
                     day_mask: Optional[Union[np.ndarray, List[bool]]] = None) -> np.ndarray:
        """
        ?? fit + transform????????
        """
        self.fit(X_train, dpsr_weights, day_mask=day_mask)
        return self.transform(X_train, day_mask=day_mask)
    def save_mapping(self, filepath: Union[str, Path]) -> None:
        """
        保存映射矩阵

        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，无映射矩阵可保存")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        mapping_data = {
            'mapping_matrix': self.mapping_matrix.tolist(),
            'target_dim': self.target_dim,
            'sigma': self.sigma,
            'alpha': self.alpha,
            'beta': self.beta,
            'optimization_history': self.optimization_history
        }

        if filepath.suffix == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2)
        elif filepath.suffix == '.pkl':
            # pickle可以直接保存numpy数组
            mapping_data['mapping_matrix'] = self.mapping_matrix
            with open(filepath, 'wb') as f:
                pickle.dump(mapping_data, f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        logger.info(f"DLFE映射矩阵已保存: {filepath}")

    def load_mapping(self, filepath: Union[str, Path]) -> None:
        """
        加载映射矩阵

        Args:
            filepath: 映射文件路径
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"映射文件不存在: {filepath}")

        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            self.mapping_matrix = np.array(mapping_data['mapping_matrix'])
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                mapping_data = pickle.load(f)
            self.mapping_matrix = mapping_data['mapping_matrix']
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        self.target_dim = mapping_data['target_dim']
        self.sigma = mapping_data['sigma']
        self.alpha = mapping_data['alpha']
        self.beta = mapping_data['beta']
        self.optimization_history = mapping_data.get('optimization_history', {})
        self.is_fitted = True

        logger.info(f"DLFE映射矩阵已加载: {filepath}")

    def reconstruction_error(self,
                            X_original: np.ndarray,
                            X_reduced: np.ndarray) -> float:
        """
        计算重构误差

        Args:
            X_original: 原始高维数据
            X_reduced: 降维后的数据

        Returns:
            重构误差
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        # 尝试重构（使用伪逆）
        A_pinv = np.linalg.pinv(self.mapping_matrix)
        X_reconstructed = X_reduced @ A_pinv.T

        # 计算误差
        error = np.mean((X_original - X_reconstructed) ** 2)

        return error


def test_dlfe():
    """测试DLFE模块"""
    # 创建模拟数据
    n_samples = 150
    n_features = 50  # 高维输入

    np.random.seed(42)

    # 生成具有流形结构的数据
    # 使用瑞士卷数据集的思想
    t = np.linspace(0, 4 * np.pi, n_samples)
    # 3D流形
    X_3d = np.column_stack([
        t * np.cos(t),
        t * np.sin(t),
        t
    ])

    # 扩展到高维（添加噪声维度）
    X_high = np.hstack([
        X_3d,
        0.1 * np.random.randn(n_samples, n_features - 3)
    ])

    # 创建模拟的DPSR权重
    dpsr_weights = np.random.rand(n_features)
    dpsr_weights = dpsr_weights / np.sum(dpsr_weights)

    # 初始化DLFE
    dlfe = DLFE(target_dim=30)

    # 训练
    print("开始DLFE训练...")
    dlfe.fit(X_high, dpsr_weights)

    # 转换
    X_embedded = dlfe.transform(X_high)

    print(f"输入形状: {X_high.shape}")
    print(f"输出形状: {X_embedded.shape}")
    print(f"映射矩阵形状: {dlfe.mapping_matrix.shape}")
    print(f"优化迭代次数: {dlfe.optimization_history['iterations']}")

    # 计算重构误差
    error = dlfe.reconstruction_error(X_high, X_embedded)
    print(f"重构误差: {error:.6f}")

    return dlfe, X_embedded


if __name__ == "__main__":
    # 运行测试
    test_dlfe()
