# -*- coding: utf-8 -*-
"""
B3 阶段主入口脚本
状态向量 g_t -> scaler_g -> PCA -> FCM 聚类分析

与 B2 的区别：
- 新增状态向量 g_t 构建（基于输入窗口的统计特征）
- 新增 scaler_g (Z-score) 标准化
- 新增 PCA 降维
- 新增 FCM 聚类 (K=3)
- 预测器仍使用 B2 的 Global LSTM（不做路由）
- 按簇分析预测误差

目标：验证聚类可解释性，为后续 B4（路由/专家模型）提供决策依据
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import pickle
import numpy as np
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.data.loader import load_solar_data, print_load_summary
from src.data.splitter import split_by_day, print_split_summary
from src.data.window import generate_windows, WindowSample
from src.data.daylight import add_daylight_flag, filter_daylight_rows, print_daylight_summary
from src.models.lstm import GlobalLSTM
from src.training.trainer import TrainConfig
from src.evaluation.rolling_eval import RollingEvaluator, print_eval_table
from src.utils.io import save_results
from src.features.scaler import FeatureScaler, FeatureScalerG
from src.utils.seed import set_seed, print_seed_info


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# 状态向量 g_t 构建
# =============================================================================

def compute_state_vector(sample: WindowSample, feature_names: List[str], capacity: float) -> Tuple[np.ndarray, List[str]]:
    """
    计算单个样本的状态向量 g_t
    
    使用输入窗口 [t-Lx, t) 内的数据做统计，不包含 t，不使用未来点
    
    Args:
        sample: WindowSample，其中 X 是 (Lx, n_features) 的数组
        feature_names: 特征列名列表
        capacity: 装机容量 (MW)
    
    Returns:
        g_t: 状态向量 (D,)
        g_names: 特征名列表
    """
    X = sample.X  # (Lx, n_features)
    Lx = X.shape[0]
    
    g_values = []
    g_names = []
    
    # 找到各特征的索引
    def get_col_idx(partial_name):
        for i, name in enumerate(feature_names):
            if partial_name.lower() in name.lower():
                return i
        return None
    
    tsi_idx = get_col_idx('Total solar')
    dni_idx = get_col_idx('Direct normal')
    ghi_idx = get_col_idx('Global horizontal')
    temp_idx = get_col_idx('temperature')
    atm_idx = get_col_idx('Atmosphere')
    power_idx = get_col_idx('Power')
    
    # -------------------------------------------------------------------------
    # 1. 辐照统计 (TSI/DNI/GHI): mean/std/min/max
    # -------------------------------------------------------------------------
    irr_indices = [('TSI', tsi_idx), ('DNI', dni_idx), ('GHI', ghi_idx)]
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            g_values.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
            g_names.extend([f'{name}_mean', f'{name}_std', f'{name}_min', f'{name}_max'])
    
    # -------------------------------------------------------------------------
    # 2. 辐照变化强度: mean(|delta|), std(delta), max(delta)
    # -------------------------------------------------------------------------
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            delta = np.diff(vals)  # (Lx-1,)
            if len(delta) > 0:
                g_values.extend([np.mean(np.abs(delta)), np.std(delta), np.max(np.abs(delta))])
                g_names.extend([f'{name}_delta_mean_abs', f'{name}_delta_std', f'{name}_delta_max_abs'])
    
    # -------------------------------------------------------------------------
    # 3. 历史功率 (Power_pu): mean/std/mean(|delta|)
    # -------------------------------------------------------------------------
    if power_idx is not None:
        power_pu = X[:, power_idx] / capacity
        g_values.extend([power_pu.mean(), power_pu.std()])
        g_names.extend(['Power_pu_mean', 'Power_pu_std'])
        
        delta_power = np.diff(power_pu)
        if len(delta_power) > 0:
            g_values.append(np.mean(np.abs(delta_power)))
            g_names.append('Power_pu_delta_mean_abs')
    
    # -------------------------------------------------------------------------
    # 4. 气温、气压: mean/std
    # -------------------------------------------------------------------------
    if temp_idx is not None:
        temp = X[:, temp_idx]
        g_values.extend([temp.mean(), temp.std()])
        g_names.extend(['Temp_mean', 'Temp_std'])
    
    if atm_idx is not None:
        atm = X[:, atm_idx]
        g_values.extend([atm.mean(), atm.std()])
        g_names.extend(['Atm_mean', 'Atm_std'])
    
    return np.array(g_values, dtype=np.float32), g_names


def build_state_vectors(samples: List[WindowSample], feature_names: List[str], capacity: float) -> Tuple[np.ndarray, List[str]]:
    """
    为所有样本构建状态向量
    
    Returns:
        G: (N, D) 状态向量矩阵
        g_names: 特征名列表
    """
    G_list = []
    g_names = None
    
    for sample in samples:
        g, names = compute_state_vector(sample, feature_names, capacity)
        G_list.append(g)
        if g_names is None:
            g_names = names
    
    G = np.array(G_list, dtype=np.float32)
    return G, g_names


# =============================================================================
# scaler_g (Z-score) + 异常处理
# =============================================================================

@dataclass
class ScalerGResult:
    scaler: 'FeatureScalerG'
    G_train: np.ndarray
    G_val: np.ndarray
    G_test: np.ndarray
    dropped_features: List[str]
    dropped_samples: Dict[str, int]
    valid_feature_names: List[str]


def remove_nan_samples(G: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    移除包含 NaN/inf 的样本
    
    Returns:
        G_clean, valid_indices, n_dropped
    """
    valid_mask = np.all(np.isfinite(G), axis=1)
    n_dropped = np.sum(~valid_mask)
    valid_indices = np.where(valid_mask)[0]
    return G[valid_mask], valid_indices, n_dropped


# =============================================================================
# FCM 聚类
# =============================================================================

class FCM:
    """Fuzzy C-Means 聚类"""
    
    def __init__(self, n_clusters: int = 3, m: float = 2.0, max_iter: int = 300, 
                 tol: float = 1e-5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers_ = None
        self.n_iter_ = 0
        self.jm_ = None
    
    def _init_membership(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        """随机初始化隶属度矩阵"""
        U = rng.rand(n_samples, self.n_clusters)
        U = U / U.sum(axis=1, keepdims=True)
        return U
    
    def _compute_centers(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """计算聚类中心"""
        Um = U ** self.m
        centers = (Um.T @ X) / Um.sum(axis=0, keepdims=True).T
        return centers
    
    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """计算样本到各中心的距离"""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            diff = X - centers[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)
        return distances
    
    def _compute_membership(self, distances: np.ndarray) -> np.ndarray:
        """更新隶属度矩阵"""
        distances = np.maximum(distances, 1e-10)
        power = 2.0 / (self.m - 1)
        
        U = np.zeros_like(distances)
        for k in range(self.n_clusters):
            denom = 0
            for j in range(self.n_clusters):
                denom += (distances[:, k] / distances[:, j]) ** power
            U[:, k] = 1.0 / denom
        
        return U
    
    def _compute_jm(self, X: np.ndarray, U: np.ndarray, centers: np.ndarray) -> float:
        """计算目标函数 J_m"""
        distances = self._compute_distances(X, centers)
        Um = U ** self.m
        return np.sum(Um * distances)
    
    def fit(self, X: np.ndarray) -> 'FCM':
        """拟合 FCM"""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        
        U = self._init_membership(n_samples, rng)
        
        for i in range(self.max_iter):
            centers = self._compute_centers(X, U)
            distances = self._compute_distances(X, centers)
            U_new = self._compute_membership(distances)
            
            diff = np.abs(U_new - U).max()
            U = U_new
            
            if diff < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        self.centers_ = centers
        self.jm_ = self._compute_jm(X, U, centers)
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测隶属度和硬标签
        
        Returns:
            U: (N, K) 软隶属度
            labels: (N,) 硬标签
        """
        distances = self._compute_distances(X, self.centers_)
        U = self._compute_membership(distances)
        labels = np.argmax(U, axis=1)
        return U, labels


def run_fcm_multi_init(X: np.ndarray, n_clusters: int = 3, m: float = 2.0, 
                       max_iter: int = 300, tol: float = 1e-5, n_init: int = 10,
                       random_seed: int = 42) -> Tuple[FCM, List[dict]]:
    """
    多次初始化运行 FCM，选择 J_m 最小的
    
    Returns:
        best_fcm, run_history
    """
    run_history = []
    best_fcm = None
    best_jm = float('inf')
    
    for i in range(n_init):
        fcm = FCM(n_clusters=n_clusters, m=m, max_iter=max_iter, tol=tol, 
                  random_state=random_seed + i)
        fcm.fit(X)
        
        run_history.append({
            'run_id': i,
            'jm': float(fcm.jm_),
            'n_iter': fcm.n_iter_,
            'random_state': random_seed + i
        })
        
        if fcm.jm_ < best_jm:
            best_jm = fcm.jm_
            best_fcm = fcm
    
    return best_fcm, run_history


# =============================================================================
# 按簇误差统计
# =============================================================================

@dataclass
class ClusterMetrics:
    cluster_id: int
    n_samples: int
    mae_1h: float
    rmse_1h: float
    nrmse_1h: float
    mae_2h: float
    rmse_2h: float
    nrmse_2h: float
    mae_4h: float
    rmse_4h: float
    nrmse_4h: float
    mae_overall: float
    rmse_overall: float
    nrmse_overall: float
    avg_membership: float


def compute_cluster_metrics(predictions: np.ndarray, targets: np.ndarray, 
                           cluster_labels: np.ndarray, memberships: np.ndarray,
                           capacity: float, horizons: dict) -> List[ClusterMetrics]:
    """
    按簇计算误差统计
    
    Args:
        predictions: (N, Hmax) 预测值 (MW)
        targets: (N, Hmax) 真实值 (MW)
        cluster_labels: (N,) 硬标签
        memberships: (N, K) 软隶属度
        capacity: 装机容量
        horizons: {steps: name} 字典
    """
    n_clusters = memberships.shape[1]
    results = []
    
    for k in range(n_clusters):
        mask = cluster_labels == k
        n_samples = np.sum(mask)
        
        if n_samples == 0:
            continue
        
        pred_k = predictions[mask]
        tgt_k = targets[mask]
        
        # 计算各 horizon 的误差
        metrics_by_h = {}
        for h, name in horizons.items():
            errors = pred_k[:, :h] - tgt_k[:, :h]
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            nrmse = rmse / capacity
            metrics_by_h[name] = {'mae': mae, 'rmse': rmse, 'nrmse': nrmse}
        
        # Overall
        errors_all = pred_k - tgt_k
        mae_all = np.mean(np.abs(errors_all))
        rmse_all = np.sqrt(np.mean(errors_all ** 2))
        nrmse_all = rmse_all / capacity
        
        # 平均隶属度
        avg_membership = memberships[mask, k].mean()
        
        results.append(ClusterMetrics(
            cluster_id=k,
            n_samples=int(n_samples),
            mae_1h=metrics_by_h['1h']['mae'],
            rmse_1h=metrics_by_h['1h']['rmse'],
            nrmse_1h=metrics_by_h['1h']['nrmse'],
            mae_2h=metrics_by_h['2h']['mae'],
            rmse_2h=metrics_by_h['2h']['rmse'],
            nrmse_2h=metrics_by_h['2h']['nrmse'],
            mae_4h=metrics_by_h['4h']['mae'],
            rmse_4h=metrics_by_h['4h']['rmse'],
            nrmse_4h=metrics_by_h['4h']['nrmse'],
            mae_overall=mae_all,
            rmse_overall=rmse_all,
            nrmse_overall=nrmse_all,
            avg_membership=avg_membership
        ))
    
    return results


def print_cluster_metrics_table(cluster_metrics: List[ClusterMetrics], total_samples: int):
    """打印按簇误差表格"""
    print("\n" + "=" * 90)
    print("按簇误差统计 (Test)")
    print("=" * 90)
    print(f"{'Cluster':<10} {'N':<8} {'Ratio':<8} {'1h nRMSE':<12} {'2h nRMSE':<12} {'4h nRMSE':<12} {'Avg Memb':<10}")
    print("-" * 90)
    
    for cm in cluster_metrics:
        ratio = cm.n_samples / total_samples * 100
        print(f"Cluster {cm.cluster_id:<3} {cm.n_samples:<8} {ratio:>5.1f}%   "
              f"{cm.nrmse_1h:>8.4f}     {cm.nrmse_2h:>8.4f}     {cm.nrmse_4h:>8.4f}     {cm.avg_membership:>8.4f}")
    
    print("=" * 90)


# =============================================================================
# 主流程
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("B3 阶段 - 状态向量 g_t -> PCA -> FCM 聚类分析")
    print("=" * 70)
    
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    print(f"\n[1/10] 配置加载完成: {config_path}")
    
    data_config = config['data']
    split_config = config['split']
    window_config = config['window']
    eval_config = config['evaluation']
    output_config = config['output']
    daylight_config = config.get('daylight_filter', {'enabled': False})
    model_config = config.get('model', {})
    train_config_dict = config.get('training', {})
    
    daylight_enabled = daylight_config.get('enabled', False)
    daylight_mode = daylight_config.get('mode', 'mask') if daylight_enabled else None
    
    target_col = data_config['target_column']
    capacity = data_config['nominal_capacity_mw']
    
    # 随机种子设置 (可复现性)
    seed = train_config_dict.get('seed', 42)
    deterministic = train_config_dict.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)
    print_seed_info(seed, deterministic)
    
    # PCA 配置
    pca_config = config.get('pca', {})
    pca_n_components_fixed = pca_config.get('n_components', None)
    pca_var_ratio = pca_config.get('var_ratio', 0.95)
    pca_max_dim = pca_config.get('max_dim', 10)
    pca_fallback_dim = pca_config.get('fallback_dim', 8)
    pca_random_state = pca_config.get('random_state', 42)
    
    # FCM 配置
    fcm_config = {
        'n_clusters': 3,
        'm': 2.0,
        'max_iter': 300,
        'tol': 1e-5,
        'n_init': 10,
        'random_seed': 42
    }
    
    print(f"  - 白天筛选: {'启用 (mode=' + daylight_mode + ')' if daylight_enabled else '禁用'}")
    print(f"  - Lx: {window_config['input_length']}, Hmax: {window_config['max_horizon']}")
    if pca_n_components_fixed is not None:
        print(f"  - PCA: n_components={pca_n_components_fixed} (fixed)")
    else:
        print(f"  - PCA: var_ratio={pca_var_ratio}, max_dim={pca_max_dim}, fallback_dim={pca_fallback_dim} (auto)")
    print(f"  - FCM: K={fcm_config['n_clusters']}, m={fcm_config['m']}, n_init={fcm_config['n_init']}")
    
    # =========================================================================
    # 2. 数据读取
    # =========================================================================
    print("\n[2/10] 读取数据...")
    
    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )
    
    print_load_summary(load_result)
    
    # 获取特征列名
    feature_names = list(load_result.df.select_dtypes(include=[np.number]).columns)
    feature_names = [c for c in feature_names if c != 'is_daylight']
    
    # =========================================================================
    # 3. 白天筛选
    # =========================================================================
    print("\n[3/10] 白天筛选...")
    
    df_for_split = load_result.df
    
    if daylight_enabled:
        daylight_result = add_daylight_flag(
            df=load_result.df,
            time_column=data_config['time_column'],
            dni_col=daylight_config['dni_col'],
            threshold=daylight_config['threshold']
        )
        
        print_daylight_summary(daylight_result, mode=daylight_mode, dni_col=daylight_config['dni_col'])
        
        if daylight_mode == 'drop':
            filtered_df, rows_before, rows_after = filter_daylight_rows(daylight_result.df)
            df_for_split = filtered_df
        else:
            df_for_split = daylight_result.df
            print(f"\n[mask 模式] 保留所有行")
    
    # =========================================================================
    # 4. 按天切分
    # =========================================================================
    print("\n[4/10] 按天切分数据...")
    
    split_result = split_by_day(
        df=df_for_split,
        time_column=data_config['time_column'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )
    
    print_split_summary(split_result)
    
    # =========================================================================
    # 5. 生成滑窗样本
    # =========================================================================
    print("\n[5/10] 生成滑窗样本...")
    
    use_mask_mode = daylight_enabled and daylight_mode == 'mask'
    
    train_window_result = generate_windows(
        df=split_result.train_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        time_interval_minutes=data_config['time_interval_minutes'],
        daylight_mask_mode=use_mask_mode
    )
    
    val_window_result = generate_windows(
        df=split_result.val_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        time_interval_minutes=data_config['time_interval_minutes'],
        daylight_mask_mode=use_mask_mode
    )
    
    test_window_result = generate_windows(
        df=split_result.test_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        time_interval_minutes=data_config['time_interval_minutes'],
        daylight_mask_mode=use_mask_mode
    )
    
    print(f"Train 样本数: {train_window_result.stats.total_samples}")
    print(f"Val 样本数:   {val_window_result.stats.total_samples}")
    print(f"Test 样本数:  {test_window_result.stats.total_samples}")
    
    # =========================================================================
    # 6. 构建状态向量 g_t
    # =========================================================================
    print("\n[6/10] 构建状态向量 g_t...")
    
    G_train_raw, g_feature_names = build_state_vectors(
        train_window_result.samples, feature_names, capacity
    )
    G_val_raw, _ = build_state_vectors(
        val_window_result.samples, feature_names, capacity
    )
    G_test_raw, _ = build_state_vectors(
        test_window_result.samples, feature_names, capacity
    )
    
    print(f"g_t 原始维度: {G_train_raw.shape[1]}")
    print(f"g_t 特征列表: {g_feature_names}")
    
    # =========================================================================
    # 7. scaler_g + 异常处理
    # =========================================================================
    print("\n[7/10] scaler_g (Z-score) + 异常处理...")
    
    # 移除 NaN 样本
    G_train_clean, train_valid_idx, train_dropped = remove_nan_samples(G_train_raw, 'train')
    G_val_clean, val_valid_idx, val_dropped = remove_nan_samples(G_val_raw, 'val')
    G_test_clean, test_valid_idx, test_dropped = remove_nan_samples(G_test_raw, 'test')
    
    print(f"[异常处理] NaN/inf 样本丢弃: train={train_dropped}, val={val_dropped}, test={test_dropped}")
    
    # 拟合 scaler_g
    scaler_g = FeatureScalerG()
    G_train_scaled, valid_g_names, dropped_g_names = scaler_g.fit_transform(G_train_clean, g_feature_names)
    G_val_scaled = scaler_g.transform(G_val_clean)
    G_test_scaled = scaler_g.transform(G_test_clean)
    
    if dropped_g_names:
        print(f"[警告] 以下特征因 std=0 被丢弃: {dropped_g_names}")
    
    print(f"scaler_g 后维度: {G_train_scaled.shape[1]}")
    print(f"保留特征: {valid_g_names}")
    
    # =========================================================================
    # 8. PCA 降维
    # =========================================================================
    print("\n[8/10] PCA 降维...")
    
    # 根据配置选择 PCA 模式
    if pca_n_components_fixed is not None:
        # fixed 模式: 使用固定维数
        pca = PCA(n_components=pca_n_components_fixed, random_state=pca_random_state)
        pca.fit(G_train_scaled)
        n_pca_components = pca.n_components_
        pca_mode = 'fixed'
        print(f"PCA 模式: fixed (n_components={pca_n_components_fixed})")
    else:
        # auto 模式: 先尝试按解释方差比例
        pca = PCA(n_components=pca_var_ratio, random_state=pca_random_state)
        pca.fit(G_train_scaled)
        n_pca_components = pca.n_components_
        
        if n_pca_components > pca_max_dim:
            # fallback 模式: 超过最大维数，使用回退维数
            print(f"PCA {pca_var_ratio} 解释方差得到 {n_pca_components} 维 > {pca_max_dim}，fallback 为 {pca_fallback_dim} 维")
            pca = PCA(n_components=pca_fallback_dim, random_state=pca_random_state)
            pca.fit(G_train_scaled)
            n_pca_components = pca_fallback_dim
            pca_mode = 'fallback'
        else:
            pca_mode = 'auto'
            print(f"PCA 模式: auto (var_ratio={pca_var_ratio} -> {n_pca_components} 维)")
    
    G_train_pca = pca.transform(G_train_scaled)
    G_val_pca = pca.transform(G_val_scaled)
    G_test_pca = pca.transform(G_test_scaled)
    
    explained_var = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)
    total_explained = float(cumsum_var[-1])
    
    print(f"PCA 降维: {G_train_scaled.shape[1]} -> {n_pca_components}")
    print(f"解释方差: {explained_var.round(4).tolist()}")
    print(f"累计解释方差: {cumsum_var.round(4).tolist()} (总计: {total_explained:.4f})")
    
    # =========================================================================
    # 9. FCM 聚类
    # =========================================================================
    print("\n[9/10] FCM 聚类 (K=3)...")
    
    best_fcm, fcm_history = run_fcm_multi_init(
        G_train_pca,
        n_clusters=fcm_config['n_clusters'],
        m=fcm_config['m'],
        max_iter=fcm_config['max_iter'],
        tol=fcm_config['tol'],
        n_init=fcm_config['n_init'],
        random_seed=fcm_config['random_seed']
    )
    
    # 找到最佳运行
    best_run = min(fcm_history, key=lambda x: x['jm'])
    print(f"FCM 最佳运行: run_id={best_run['run_id']}, J_m={best_run['jm']:.4f}, n_iter={best_run['n_iter']}")
    
    # Train 集标签
    U_train, labels_train = best_fcm.predict(G_train_pca)
    U_test, labels_test = best_fcm.predict(G_test_pca)
    
    # 各簇占比
    for k in range(fcm_config['n_clusters']):
        ratio = np.sum(labels_train == k) / len(labels_train) * 100
        print(f"  Cluster {k}: {np.sum(labels_train == k)} 样本 ({ratio:.1f}%)")
    
    # 聚类质量指标
    silhouette = silhouette_score(G_train_pca, labels_train)
    dbi = davies_bouldin_score(G_train_pca, labels_train)
    
    # 中心间距离矩阵
    centers = best_fcm.centers_
    center_distances = np.zeros((fcm_config['n_clusters'], fcm_config['n_clusters']))
    for i in range(fcm_config['n_clusters']):
        for j in range(fcm_config['n_clusters']):
            center_distances[i, j] = np.linalg.norm(centers[i] - centers[j])
    
    print(f"\n聚类质量指标:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {dbi:.4f}")
    print(f"  中心间距离矩阵:\n{np.round(center_distances, 4)}")
    
    # =========================================================================
    # 9.5 簇语义映射 (仅用训练集)
    # =========================================================================
    print("\n[9.5/10] 簇语义映射 (仅用训练集)...")
    
    # 获取 g_t 原始特征中的关键特征索引
    def get_feature_idx(name_part, names_list):
        for i, n in enumerate(names_list):
            if name_part in n:
                return i
        return None
    
    power_pu_mean_idx = get_feature_idx('Power_pu_mean', g_feature_names)
    power_pu_delta_idx = get_feature_idx('Power_pu_delta_mean_abs', g_feature_names)
    ghi_delta_std_idx = get_feature_idx('GHI_delta_std', g_feature_names)
    
    # 按簇统计 (使用原始 G_train_raw 而非标准化后的)
    train_samples_valid = [train_window_result.samples[i] for i in train_valid_idx]
    cluster_stats_semantic = {}
    
    for k in range(fcm_config['n_clusters']):
        mask = labels_train == k
        G_k = G_train_raw[train_valid_idx][mask]
        
        stats = {
            'n_samples': int(np.sum(mask)),
            'Power_pu_mean': float(G_k[:, power_pu_mean_idx].mean()) if power_pu_mean_idx is not None else 0.0,
            'Power_pu_delta_mean_abs': float(G_k[:, power_pu_delta_idx].mean()) if power_pu_delta_idx is not None else 0.0,
            'GHI_delta_std': float(G_k[:, ghi_delta_std_idx].mean()) if ghi_delta_std_idx is not None else 0.0
        }
        cluster_stats_semantic[k] = stats
    
    # 打印统计摘要
    print(f"\n{'Cluster':<10} {'N':>8} {'Power_pu_mean':>15} {'Power_delta':>12} {'GHI_delta_std':>14}")
    print("-" * 65)
    for k in range(fcm_config['n_clusters']):
        s = cluster_stats_semantic[k]
        print(f"Cluster {k:<3} {s['n_samples']:>8} {s['Power_pu_mean']:>15.4f} {s['Power_pu_delta_mean_abs']:>12.4f} {s['GHI_delta_std']:>14.4f}")
    
    # 语义映射规则:
    # 1. Power_pu_mean 最高 -> clear (晴天稳定高功率)
    # 2. Power_pu_mean 最低 -> low_irradiance (低辐照/早晚)
    # 3. 中间 -> cloudy (多云波动)
    
    # 按 Power_pu_mean 排序
    sorted_clusters = sorted(cluster_stats_semantic.keys(), 
                            key=lambda k: cluster_stats_semantic[k]['Power_pu_mean'],
                            reverse=True)
    
    cluster_semantic_map = {}
    semantic_labels = ['clear', 'cloudy', 'low_irradiance']
    
    for rank, k in enumerate(sorted_clusters):
        cluster_semantic_map[k] = semantic_labels[rank]
    
    # 打印映射结果
    print(f"\n簇语义映射 (按 Power_pu_mean 排序):")
    for k in range(fcm_config['n_clusters']):
        s = cluster_stats_semantic[k]
        label = cluster_semantic_map[k]
        print(f"  Cluster {k} -> {label:15} (Power_pu_mean={s['Power_pu_mean']:.4f}, delta={s['Power_pu_delta_mean_abs']:.4f})")
    
    # =========================================================================
    # 10. 加载 B2 模型进行 Rolling Evaluation + 按簇误差分析
    # =========================================================================
    print("\n[10/10] 加载 B2 模型进行预测 + 按簇误差分析...")
    
    # 加载 scaler_X
    scaler_X_path = PROJECT_ROOT / output_config['results_dir'] / "scaler_X.pkl"
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    print(f"Loaded scaler_X from: {scaler_X_path}")
    
    # 加载模型
    model_path = PROJECT_ROOT / output_config['results_dir'] / "model_global.pt"
    
    input_size = scaler_X.n_features_
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.1)
    output_steps = window_config['max_horizon']
    
    model = GlobalLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_steps=output_steps,
        dropout=dropout
    )
    
    # 设备配置
    device_config = train_config_dict.get('device', 'auto')
    if device_config == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from: {model_path}")
    print(f"Device: {device}")
    
    # 获取有效的 test 样本
    test_samples_valid = [test_window_result.samples[i] for i in test_valid_idx]
    
    # Rolling evaluation
    predictions = []
    targets = []
    
    for sample in test_samples_valid:
        # 输入标准化
        x_scaled = scaler_X.transform(sample.X)
        x_tensor = torch.FloatTensor(x_scaled.reshape(1, -1, x_scaled.shape[-1])).to(device)
        
        with torch.no_grad():
            y_pred_pu = model(x_tensor)
        
        y_pred_mw = y_pred_pu.cpu().numpy().flatten() * capacity
        predictions.append(y_pred_mw)
        targets.append(sample.Y)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 按簇误差统计
    horizons = {4: '1h', 8: '2h', 16: '4h'}
    cluster_metrics = compute_cluster_metrics(
        predictions, targets, labels_test, U_test, capacity, horizons
    )
    
    print_cluster_metrics_table(cluster_metrics, len(test_samples_valid))
    
    # 整体误差
    print("\n[对比] 整体误差 (所有簇):")
    for h, name in horizons.items():
        errors = predictions[:, :h] - targets[:, :h]
        rmse = np.sqrt(np.mean(errors ** 2))
        nrmse = rmse / capacity
        print(f"  {name} nRMSE: {nrmse:.4f}")
    
    # =========================================================================
    # 保存产物
    # =========================================================================
    print("\n[保存产物]...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / output_config['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. g_feature_names.json
    with open(results_dir / "g_feature_names.json", 'w', encoding='utf-8') as f:
        json.dump({'original': g_feature_names, 'valid': valid_g_names, 'dropped': dropped_g_names}, f, indent=2)
    print(f"  g_feature_names.json saved")
    
    # 2. scaler_g.pkl
    with open(results_dir / "scaler_g.pkl", 'wb') as f:
        pickle.dump(scaler_g, f)
    print(f"  scaler_g.pkl saved")
    
    # 3. pca.pkl
    with open(results_dir / "pca.pkl", 'wb') as f:
        pickle.dump(pca, f)
    print(f"  pca.pkl saved")
    
    # 4. pca_explained_variance.json
    with open(results_dir / "pca_explained_variance.json", 'w', encoding='utf-8') as f:
        json.dump({
            'n_components': int(n_pca_components),
            'mode': pca_mode,
            'total_explained_variance': total_explained,
            'explained_variance_ratio': explained_var.tolist(),
            'cumulative_variance_ratio': cumsum_var.tolist()
        }, f, indent=2)
    print(f"  pca_explained_variance.json saved")
    
    # 5. fcm_centers.npy
    np.save(results_dir / "fcm_centers.npy", centers)
    print(f"  fcm_centers.npy saved")
    
    # 6. fcm_params.json
    fcm_params = {
        'n_clusters': fcm_config['n_clusters'],
        'm': fcm_config['m'],
        'max_iter': fcm_config['max_iter'],
        'tol': fcm_config['tol'],
        'n_init': fcm_config['n_init'],
        'random_seed': fcm_config['random_seed'],
        'best_run_id': best_run['run_id'],
        'best_jm': best_run['jm'],
        'best_n_iter': best_run['n_iter'],
        'run_history': fcm_history
    }
    with open(results_dir / "fcm_params.json", 'w', encoding='utf-8') as f:
        json.dump(fcm_params, f, indent=2)
    print(f"  fcm_params.json saved")
    
    # 6.5 router_config.json (包含簇语义映射)
    router_config = {
        'n_clusters': fcm_config['n_clusters'],
        'm': fcm_config['m'],
        'scaler_g_path': 'scaler_g.pkl',
        'pca_path': 'pca.pkl',
        'fcm_centers_path': 'fcm_centers.npy',
        'scaler_X_path': 'scaler_X.pkl',
        'cluster_semantic_map': {str(k): v for k, v in cluster_semantic_map.items()},
        'cluster_stats': {
            str(k): {
                'n_samples': s['n_samples'],
                'Power_pu_mean': s['Power_pu_mean'],
                'Power_pu_delta_mean_abs': s['Power_pu_delta_mean_abs'],
                'GHI_delta_std': s['GHI_delta_std'],
                'semantic_label': cluster_semantic_map[k]
            }
            for k, s in cluster_stats_semantic.items()
        }
    }
    with open(results_dir / "router_config.json", 'w', encoding='utf-8') as f:
        json.dump(router_config, f, indent=2, ensure_ascii=False)
    print(f"  router_config.json saved")
    
    # 7. metrics_B3.json
    metrics_b3 = {
        'experiment': 'B3: g_t -> PCA -> FCM analysis',
        'timestamp': timestamp,
        'g_dim_original': len(g_feature_names),
        'g_dim_after_scaler': len(valid_g_names),
        'pca': {
            'n_components': int(n_pca_components),
            'mode': pca_mode,
            'total_explained_variance': total_explained
        },
        'fcm': {
            'n_clusters': fcm_config['n_clusters'],
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(dbi),
            'center_distances': center_distances.tolist()
        },
        'cluster_ratios': {
            f'cluster_{k}': float(np.sum(labels_train == k) / len(labels_train))
            for k in range(fcm_config['n_clusters'])
        },
        'cluster_metrics_test': [
            {
                'cluster_id': cm.cluster_id,
                'n_samples': cm.n_samples,
                'nrmse_1h': float(cm.nrmse_1h),
                'nrmse_2h': float(cm.nrmse_2h),
                'nrmse_4h': float(cm.nrmse_4h),
                'nrmse_overall': float(cm.nrmse_overall),
                'avg_membership': float(cm.avg_membership)
            }
            for cm in cluster_metrics
        ],
        'dropped_samples': {
            'train': int(train_dropped),
            'val': int(val_dropped),
            'test': int(test_dropped)
        }
    }
    
    with open(results_dir / "metrics_B3.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_b3, f, indent=2, ensure_ascii=False)
    print(f"  metrics_B3.json saved")
    
    # 8. metrics_B3.csv
    import pandas as pd
    df_metrics = pd.DataFrame([
        {
            'cluster_id': cm.cluster_id,
            'n_samples': cm.n_samples,
            'ratio': cm.n_samples / len(test_samples_valid),
            'mae_1h': cm.mae_1h,
            'rmse_1h': cm.rmse_1h,
            'nrmse_1h': cm.nrmse_1h,
            'mae_2h': cm.mae_2h,
            'rmse_2h': cm.rmse_2h,
            'nrmse_2h': cm.nrmse_2h,
            'mae_4h': cm.mae_4h,
            'rmse_4h': cm.rmse_4h,
            'nrmse_4h': cm.nrmse_4h,
            'avg_membership': cm.avg_membership,
            'pca_dim': n_pca_components,
            'pca_mode': pca_mode,
            'pca_total_explained': total_explained
        }
        for cm in cluster_metrics
    ])
    df_metrics.to_csv(results_dir / "metrics_B3.csv", index=False, encoding='utf-8')
    print(f"  metrics_B3.csv saved")
    
    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 70)
    print("B3 阶段完成!")
    print(f"  - g_t 维度: {len(g_feature_names)} -> {len(valid_g_names)} (scaler) -> {n_pca_components} (PCA)")
    print(f"  - FCM: K={fcm_config['n_clusters']}, Silhouette={silhouette:.4f}, DBI={dbi:.4f}")
    print(f"  - 预测器: B2 Global LSTM (未修改)")
    for cm in cluster_metrics:
        print(f"  - Cluster {cm.cluster_id}: {cm.n_samples} 样本, 4h nRMSE={cm.nrmse_4h:.4f}")
    print("=" * 70)
    
    return metrics_b3


if __name__ == "__main__":
    main()
