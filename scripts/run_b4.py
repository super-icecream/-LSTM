# -*- coding: utf-8 -*-
"""
B4 阶段主入口脚本
硬路由多专家 LSTM

与 B3 的区别：
- B3: 仅做聚类分析，预测器仍使用 B2 Global LSTM
- B4: 使用 router (scaler_g + PCA + FCM) 进行硬路由，训练 3 个独立 LSTM 专家

流程：
1. 复用 B3 的 router: 加载 scaler_g.pkl, pca.pkl, fcm_centers.npy
2. 对 train/val/test 每个样本计算 g_t -> scaler_g -> PCA -> FCM -> 硬标签 k_t
3. 按 k_t 将样本分成 3 份，分别训练 3 个结构相同但参数独立的 LSTM 专家
4. 推理时先用 router 分簇，再送入对应专家预测
5. Rolling evaluation 输出整体和按簇误差
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time

from src.data.loader import load_solar_data, print_load_summary
from src.data.splitter import split_by_day, print_split_summary
from src.data.window import generate_windows, WindowSample
from src.data.daylight import add_daylight_flag, filter_daylight_rows, print_daylight_summary
from src.models.lstm import GlobalLSTM
from src.features.scaler import FeatureScaler, FeatureScalerG
from src.utils.seed import set_seed, get_dataloader_generator, worker_init_fn, print_seed_info


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# 状态向量 g_t 构建 (复用 B3)
# =============================================================================

def compute_state_vector(sample: WindowSample, feature_names: List[str], capacity: float) -> np.ndarray:
    """计算单个样本的状态向量 g_t"""
    X = sample.X
    
    g_values = []
    
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
    
    # 辐照统计
    irr_indices = [('TSI', tsi_idx), ('DNI', dni_idx), ('GHI', ghi_idx)]
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            g_values.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
    
    # 辐照变化强度
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            delta = np.diff(vals)
            if len(delta) > 0:
                g_values.extend([np.mean(np.abs(delta)), np.std(delta), np.max(np.abs(delta))])
    
    # 历史功率
    if power_idx is not None:
        power_pu = X[:, power_idx] / capacity
        g_values.extend([power_pu.mean(), power_pu.std()])
        delta_power = np.diff(power_pu)
        if len(delta_power) > 0:
            g_values.append(np.mean(np.abs(delta_power)))
    
    # 气温、气压
    if temp_idx is not None:
        temp = X[:, temp_idx]
        g_values.extend([temp.mean(), temp.std()])
    
    if atm_idx is not None:
        atm = X[:, atm_idx]
        g_values.extend([atm.mean(), atm.std()])
    
    return np.array(g_values, dtype=np.float32)


def build_state_vectors(samples: List[WindowSample], feature_names: List[str], capacity: float) -> np.ndarray:
    """为所有样本构建状态向量"""
    return np.array([compute_state_vector(s, feature_names, capacity) for s in samples], dtype=np.float32)


# =============================================================================
# Router 类 (封装 scaler_g + PCA + FCM)
# =============================================================================

class Router:
    """硬路由器: scaler_g -> PCA -> FCM -> argmax"""
    
    def __init__(self, scaler_g, pca, fcm_centers, m=2.0):
        self.scaler_g = scaler_g
        self.pca = pca
        self.fcm_centers = fcm_centers
        self.n_clusters = fcm_centers.shape[0]
        self.m = m
    
    def _compute_membership(self, X_pca: np.ndarray) -> np.ndarray:
        """计算 FCM 软隶属度"""
        n_samples = X_pca.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for k in range(self.n_clusters):
            diff = X_pca - self.fcm_centers[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)
        
        distances = np.maximum(distances, 1e-10)
        power = 2.0 / (self.m - 1)
        
        U = np.zeros_like(distances)
        for k in range(self.n_clusters):
            denom = 0
            for j in range(self.n_clusters):
                denom += (distances[:, k] / distances[:, j]) ** power
            U[:, k] = 1.0 / denom
        
        return U
    
    def route(self, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        路由样本到簇
        
        Args:
            G: (N, D) 原始状态向量
        
        Returns:
            labels: (N,) 硬标签
            memberships: (N, K) 软隶属度
        """
        G_scaled = self.scaler_g.transform(G)
        G_pca = self.pca.transform(G_scaled)
        memberships = self._compute_membership(G_pca)
        labels = np.argmax(memberships, axis=1)
        return labels, memberships
    
    def route_single(self, g: np.ndarray) -> Tuple[int, np.ndarray]:
        """路由单个样本"""
        labels, memberships = self.route(g.reshape(1, -1))
        return int(labels[0]), memberships[0]


# =============================================================================
# 专家训练器
# =============================================================================

@dataclass
class ExpertTrainResult:
    expert_id: int
    best_epoch: int
    best_val_loss: float
    best_val_rmse_4h_pu: float
    stopped_early: bool
    total_epochs: int
    train_samples: int
    val_samples: int


def train_expert(
    expert_id: int,
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    device: str,
    capacity: float,
    seed: int = 42
) -> Tuple[nn.Module, ExpertTrainResult]:
    """训练单个专家模型"""
    
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 256)
    max_epochs = config.get('max_epochs', 100)
    patience = config.get('patience', 10)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # DataLoader (支持可复现性)
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    generator = get_dataloader_generator(seed + expert_id)  # 每个专家使用不同但确定的种子
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        generator=generator
    )
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    best_val_loss = float('inf')
    best_val_rmse_4h_pu = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    
    print(f"\n  [Expert {expert_id}] Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"  {'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'4h RMSE(pu)':>12}")
    print(f"  {'-'*50}")
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t).item()
            
            # 4h RMSE (p.u.)
            errors_4h = y_val_pred[:, :16] - y_val_t[:, :16]
            rmse_4h_pu = torch.sqrt(torch.mean(errors_4h ** 2)).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_rmse_4h_pu = rmse_4h_pu
            best_epoch = epoch + 1
            # 深拷贝: 必须 clone tensor，否则后续训练会覆盖
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter == 0:
            marker = " *" if patience_counter == 0 else ""
            print(f"  {epoch+1:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {rmse_4h_pu:>12.6f}{marker}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    
    # 加载最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)
        model_source = "best"
    else:
        model_source = "last"
    
    print(f"  [Checkpoint] 评估使用: {model_source} epoch 权重 (epoch={best_epoch})")
    
    result = ExpertTrainResult(
        expert_id=expert_id,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_rmse_4h_pu=best_val_rmse_4h_pu,
        stopped_early=patience_counter >= patience,
        total_epochs=epoch + 1,
        train_samples=len(X_train),
        val_samples=len(X_val)
    )
    
    return model, result


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
    avg_membership: float


def compute_metrics_by_cluster(
    predictions: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    memberships: np.ndarray,
    capacity: float,
    n_clusters: int
) -> List[ClusterMetrics]:
    """按簇计算误差"""
    results = []
    
    for k in range(n_clusters):
        mask = labels == k
        n_samples = np.sum(mask)
        
        if n_samples == 0:
            continue
        
        pred_k = predictions[mask]
        tgt_k = targets[mask]
        
        # 1h (4步)
        e_1h = pred_k[:, :4] - tgt_k[:, :4]
        mae_1h = np.mean(np.abs(e_1h))
        rmse_1h = np.sqrt(np.mean(e_1h ** 2))
        
        # 2h (8步)
        e_2h = pred_k[:, :8] - tgt_k[:, :8]
        mae_2h = np.mean(np.abs(e_2h))
        rmse_2h = np.sqrt(np.mean(e_2h ** 2))
        
        # 4h (16步)
        e_4h = pred_k[:, :16] - tgt_k[:, :16]
        mae_4h = np.mean(np.abs(e_4h))
        rmse_4h = np.sqrt(np.mean(e_4h ** 2))
        
        avg_memb = memberships[mask, k].mean()
        
        results.append(ClusterMetrics(
            cluster_id=k,
            n_samples=int(n_samples),
            mae_1h=mae_1h, rmse_1h=rmse_1h, nrmse_1h=rmse_1h/capacity,
            mae_2h=mae_2h, rmse_2h=rmse_2h, nrmse_2h=rmse_2h/capacity,
            mae_4h=mae_4h, rmse_4h=rmse_4h, nrmse_4h=rmse_4h/capacity,
            avg_membership=avg_memb
        ))
    
    return results


# =============================================================================
# 主流程
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("B4 阶段 - 硬路由多专家 LSTM")
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
    output_config = config['output']
    daylight_config = config.get('daylight_filter', {'enabled': False})
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    
    daylight_enabled = daylight_config.get('enabled', False)
    daylight_mode = daylight_config.get('mode', 'mask') if daylight_enabled else None
    
    target_col = data_config['target_column']
    capacity = data_config['nominal_capacity_mw']
    n_clusters = 3
    
    # 随机种子设置 (可复现性)
    seed = train_config.get('seed', 42)
    deterministic = train_config.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)
    print_seed_info(seed, deterministic)
    
    # 设备配置
    device_config = train_config.get('device', 'auto')
    if device_config == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    # 注意: 如果 deterministic=True, cudnn.benchmark 会被设为 False
    # 如果需要 benchmark 加速，需设置 deterministic=False
    cudnn_benchmark = train_config.get('cudnn_benchmark', False)
    if cudnn_benchmark and device == "cuda" and not deterministic:
        torch.backends.cudnn.benchmark = True
    
    print(f"  - Device: {device}")
    if device == "cuda":
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - n_clusters: {n_clusters}")
    
    # =========================================================================
    # 2. 加载 Router 组件 (从 B3 产物)
    # =========================================================================
    print("\n[2/10] 加载 Router 组件...")
    
    results_dir = PROJECT_ROOT / output_config['results_dir']
    
    # 加载 router_config.json (包含簇语义映射)
    with open(results_dir / "router_config.json", 'r', encoding='utf-8') as f:
        router_config_data = json.load(f)
    
    cluster_semantic_map = {int(k): v for k, v in router_config_data['cluster_semantic_map'].items()}
    cluster_stats = router_config_data.get('cluster_stats', {})
    
    with open(results_dir / "scaler_g.pkl", 'rb') as f:
        scaler_g = pickle.load(f)
    print(f"  Loaded scaler_g.pkl")
    
    with open(results_dir / "pca.pkl", 'rb') as f:
        pca = pickle.load(f)
    print(f"  Loaded pca.pkl (n_components={pca.n_components_})")
    
    fcm_centers = np.load(results_dir / "fcm_centers.npy")
    print(f"  Loaded fcm_centers.npy (shape={fcm_centers.shape})")
    
    with open(results_dir / "scaler_X.pkl", 'rb') as f:
        scaler_X = pickle.load(f)
    print(f"  Loaded scaler_X.pkl")
    
    router = Router(scaler_g, pca, fcm_centers, m=2.0)
    
    # 打印簇语义映射
    print(f"\n  簇语义映射 (来自 B3 训练集):")
    print(f"  {'Cluster':<10} {'Semantic':>15} {'Power_pu_mean':>15} {'Power_delta':>12}")
    print(f"  {'-'*55}")
    for k in range(n_clusters):
        label = cluster_semantic_map.get(k, 'unknown')
        stats = cluster_stats.get(str(k), {})
        ppm = stats.get('Power_pu_mean', 0.0)
        ppd = stats.get('Power_pu_delta_mean_abs', 0.0)
        print(f"  Cluster {k:<3} {label:>15} {ppm:>15.4f} {ppd:>12.4f}")
    
    # =========================================================================
    # 3. 数据读取
    # =========================================================================
    print("\n[3/10] 读取数据...")
    
    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )
    
    feature_names = list(load_result.df.select_dtypes(include=[np.number]).columns)
    feature_names = [c for c in feature_names if c != 'is_daylight']
    
    # =========================================================================
    # 4. 白天筛选 + 按天切分
    # =========================================================================
    print("\n[4/10] 白天筛选 + 按天切分...")
    
    df_for_split = load_result.df
    
    if daylight_enabled:
        daylight_result = add_daylight_flag(
            df=load_result.df,
            time_column=data_config['time_column'],
            dni_col=daylight_config['dni_col'],
            threshold=daylight_config['threshold']
        )
        df_for_split = daylight_result.df
    
    split_result = split_by_day(
        df=df_for_split,
        time_column=data_config['time_column'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )
    
    print(f"  Train days: {split_result.train_days}, Val days: {split_result.val_days}, Test days: {split_result.test_days}")
    
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
    
    print(f"  Train: {len(train_window_result.samples)}, Val: {len(val_window_result.samples)}, Test: {len(test_window_result.samples)}")
    
    # =========================================================================
    # 6. 构建状态向量 g_t + 路由
    # =========================================================================
    print("\n[6/10] 构建状态向量 + 路由分簇...")
    
    # 构建 g_t
    G_train = build_state_vectors(train_window_result.samples, feature_names, capacity)
    G_val = build_state_vectors(val_window_result.samples, feature_names, capacity)
    G_test = build_state_vectors(test_window_result.samples, feature_names, capacity)
    
    # 路由
    labels_train, memb_train = router.route(G_train)
    labels_val, memb_val = router.route(G_val)
    labels_test, memb_test = router.route(G_test)
    
    # 统计
    cluster_stats_dist = {'train': {}, 'val': {}, 'test': {}}
    for k in range(n_clusters):
        cluster_stats_dist['train'][k] = int(np.sum(labels_train == k))
        cluster_stats_dist['val'][k] = int(np.sum(labels_val == k))
        cluster_stats_dist['test'][k] = int(np.sum(labels_test == k))
    
    print("\n  Cluster distribution (with semantic labels):")
    print(f"  {'Cluster':<10} {'Semantic':>15} {'Train':>10} {'Val':>10} {'Test':>10}")
    print(f"  {'-'*60}")
    for k in range(n_clusters):
        label = cluster_semantic_map.get(k, 'unknown')
        print(f"  Cluster {k:<3} {label:>15} {cluster_stats_dist['train'][k]:>10} {cluster_stats_dist['val'][k]:>10} {cluster_stats_dist['test'][k]:>10}")
    
    # =========================================================================
    # 7. 准备训练数据 (按簇分组)
    # =========================================================================
    print("\n[7/10] 准备训练数据...")
    
    def prepare_cluster_data(samples, labels, scaler_X, capacity):
        """按簇准备训练数据"""
        cluster_data = {k: {'X': [], 'y': [], 'indices': []} for k in range(n_clusters)}
        
        for i, (sample, k) in enumerate(zip(samples, labels)):
            X_scaled = scaler_X.transform(sample.X)
            y_pu = sample.Y / capacity
            cluster_data[k]['X'].append(X_scaled)
            cluster_data[k]['y'].append(y_pu)
            cluster_data[k]['indices'].append(i)
        
        for k in range(n_clusters):
            if cluster_data[k]['X']:
                cluster_data[k]['X'] = np.array(cluster_data[k]['X'], dtype=np.float32)
                cluster_data[k]['y'] = np.array(cluster_data[k]['y'], dtype=np.float32)
            else:
                cluster_data[k]['X'] = np.empty((0, window_config['input_length'], scaler_X.n_features_), dtype=np.float32)
                cluster_data[k]['y'] = np.empty((0, window_config['max_horizon']), dtype=np.float32)
        
        return cluster_data
    
    train_cluster_data = prepare_cluster_data(train_window_result.samples, labels_train, scaler_X, capacity)
    val_cluster_data = prepare_cluster_data(val_window_result.samples, labels_val, scaler_X, capacity)
    test_cluster_data = prepare_cluster_data(test_window_result.samples, labels_test, scaler_X, capacity)
    
    # =========================================================================
    # 8. 训练专家模型
    # =========================================================================
    print("\n[8/10] 训练专家模型...")
    
    input_size = scaler_X.n_features_
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.1)
    output_steps = window_config['max_horizon']
    
    experts = {}
    expert_results = {}
    
    for k in range(n_clusters):
        print(f"\n{'='*60}")
        print(f"Training Expert {k}")
        print(f"{'='*60}")
        
        X_train_k = train_cluster_data[k]['X']
        y_train_k = train_cluster_data[k]['y']
        X_val_k = val_cluster_data[k]['X']
        y_val_k = val_cluster_data[k]['y']
        
        if len(X_train_k) == 0:
            print(f"  [Warning] No training samples for cluster {k}, skipping...")
            continue
        
        if len(X_val_k) == 0:
            print(f"  [Warning] No validation samples for cluster {k}, using train for val...")
            X_val_k = X_train_k[:min(100, len(X_train_k))]
            y_val_k = y_train_k[:min(100, len(y_train_k))]
        
        model = GlobalLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_steps=output_steps,
            dropout=dropout
        )
        
        trained_model, result = train_expert(
            expert_id=k,
            model=model,
            X_train=X_train_k,
            y_train=y_train_k,
            X_val=X_val_k,
            y_val=y_val_k,
            config=train_config,
            device=device,
            capacity=capacity,
            seed=seed
        )
        
        experts[k] = trained_model
        expert_results[k] = result
        
        print(f"  Best epoch: {result.best_epoch}, Best val loss: {result.best_val_loss:.6f}")
    
    # =========================================================================
    # 9. Rolling Evaluation (Test)
    # =========================================================================
    print("\n[9/10] Rolling Evaluation (Test)...")
    
    predictions_all = []
    targets_all = []
    
    for k in range(n_clusters):
        if k not in experts:
            continue
        
        model = experts[k]
        model.eval()
        
        X_test_k = test_cluster_data[k]['X']
        y_test_k = test_cluster_data[k]['y']
        
        if len(X_test_k) == 0:
            continue
        
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test_k).to(device)
            y_pred_pu = model(X_t).cpu().numpy()
        
        # 转换回 MW
        y_pred_mw = y_pred_pu * capacity
        y_true_mw = y_test_k * capacity
        
        predictions_all.append(y_pred_mw)
        targets_all.append(y_true_mw)
    
    predictions_all = np.vstack(predictions_all)
    targets_all = np.vstack(targets_all)
    
    # 整体误差
    def calc_metrics(pred, tgt, h):
        e = pred[:, :h] - tgt[:, :h]
        mae = np.mean(np.abs(e))
        rmse = np.sqrt(np.mean(e ** 2))
        return mae, rmse, rmse / capacity
    
    mae_1h, rmse_1h, nrmse_1h = calc_metrics(predictions_all, targets_all, 4)
    mae_2h, rmse_2h, nrmse_2h = calc_metrics(predictions_all, targets_all, 8)
    mae_4h, rmse_4h, nrmse_4h = calc_metrics(predictions_all, targets_all, 16)
    
    print(f"\n整体误差 (Test):")
    print(f"  1h: MAE={mae_1h:.4f} MW, RMSE={rmse_1h:.4f} MW, nRMSE={nrmse_1h:.4f}")
    print(f"  2h: MAE={mae_2h:.4f} MW, RMSE={rmse_2h:.4f} MW, nRMSE={nrmse_2h:.4f}")
    print(f"  4h: MAE={mae_4h:.4f} MW, RMSE={rmse_4h:.4f} MW, nRMSE={nrmse_4h:.4f}")
    
    # 按簇误差
    # 重新收集按簇的预测和目标
    cluster_predictions = {}
    cluster_targets = {}
    cluster_memberships = {}
    
    for k in range(n_clusters):
        if k not in experts:
            continue
        
        X_test_k = test_cluster_data[k]['X']
        y_test_k = test_cluster_data[k]['y']
        indices_k = test_cluster_data[k]['indices']
        
        if len(X_test_k) == 0:
            continue
        
        model = experts[k]
        model.eval()
        
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test_k).to(device)
            y_pred_pu = model(X_t).cpu().numpy()
        
        cluster_predictions[k] = y_pred_pu * capacity
        cluster_targets[k] = y_test_k * capacity
        cluster_memberships[k] = memb_test[indices_k, k]
    
    cluster_metrics = []
    for k in range(n_clusters):
        if k not in cluster_predictions:
            continue
        
        pred = cluster_predictions[k]
        tgt = cluster_targets[k]
        memb = cluster_memberships[k]
        n_samples = len(pred)
        
        cm_mae_1h, cm_rmse_1h, cm_nrmse_1h = calc_metrics(pred, tgt, 4)
        cm_mae_2h, cm_rmse_2h, cm_nrmse_2h = calc_metrics(pred, tgt, 8)
        cm_mae_4h, cm_rmse_4h, cm_nrmse_4h = calc_metrics(pred, tgt, 16)
        
        cluster_metrics.append(ClusterMetrics(
            cluster_id=k,
            n_samples=n_samples,
            mae_1h=cm_mae_1h, rmse_1h=cm_rmse_1h, nrmse_1h=cm_nrmse_1h,
            mae_2h=cm_mae_2h, rmse_2h=cm_rmse_2h, nrmse_2h=cm_nrmse_2h,
            mae_4h=cm_mae_4h, rmse_4h=cm_rmse_4h, nrmse_4h=cm_nrmse_4h,
            avg_membership=float(memb.mean())
        ))
    
    # =========================================================================
    # 10. 汇总表 + 保存产物
    # =========================================================================
    print("\n[10/10] 汇总表 + 保存产物...")
    
    # 汇总表
    print("\n" + "=" * 100)
    print("B4 汇总表")
    print("=" * 100)
    
    # 样本分布
    print("\n[样本分布]")
    print(f"{'Cluster':<10} {'Train':>10} {'Train%':>10} {'Val':>10} {'Val%':>10} {'Test':>10} {'Test%':>10}")
    print("-" * 70)
    total_train = len(labels_train)
    total_val = len(labels_val)
    total_test = len(labels_test)
    for k in range(n_clusters):
        t = cluster_stats_dist['train'][k]
        v = cluster_stats_dist['val'][k]
        te = cluster_stats_dist['test'][k]
        print(f"Cluster {k:<3} {t:>10} {t/total_train*100:>9.1f}% {v:>10} {v/total_val*100:>9.1f}% {te:>10} {te/total_test*100:>9.1f}%")
    
    # 专家训练结果
    print("\n[专家训练结果]")
    print(f"{'Expert':<10} {'Train N':>10} {'Val N':>10} {'Best Epoch':>12} {'Best Val Loss':>14}")
    print("-" * 60)
    for k in range(n_clusters):
        if k in expert_results:
            r = expert_results[k]
            print(f"Expert {k:<4} {r.train_samples:>10} {r.val_samples:>10} {r.best_epoch:>12} {r.best_val_loss:>14.6f}")
    
    # Test 按簇误差 (带语义标签)
    print("\n[Test 按簇误差]")
    print(f"{'Cluster':<10} {'Semantic':>15} {'N':>8} {'1h nRMSE':>12} {'2h nRMSE':>12} {'4h nRMSE':>12} {'Avg Memb':>10}")
    print("-" * 85)
    for cm in cluster_metrics:
        label = cluster_semantic_map.get(cm.cluster_id, 'unknown')
        print(f"Cluster {cm.cluster_id:<3} {label:>15} {cm.n_samples:>8} {cm.nrmse_1h:>12.4f} {cm.nrmse_2h:>12.4f} {cm.nrmse_4h:>12.4f} {cm.avg_membership:>10.4f}")
    print("-" * 85)
    print(f"{'Overall':<10} {'':<15} {len(predictions_all):>8} {nrmse_1h:>12.4f} {nrmse_2h:>12.4f} {nrmse_4h:>12.4f}")
    print("=" * 100)
    
    # 保存产物
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存专家模型
    for k in range(n_clusters):
        if k in experts:
            model_path = results_dir / f"model_expert_{k}.pt"
            torch.save(experts[k].state_dict(), model_path)
            print(f"  Saved: {model_path.name}")
    
    # 注: router_config.json 已由 B3 生成，包含簇语义映射，B4 不再覆盖
    
    # metrics_B4.json
    metrics_b4 = {
        'experiment': 'B4: Hard-routing Multi-Expert LSTM',
        'timestamp': timestamp,
        'n_clusters': n_clusters,
        'overall_metrics': {
            '1h': {'mae': float(mae_1h), 'rmse': float(rmse_1h), 'nrmse': float(nrmse_1h)},
            '2h': {'mae': float(mae_2h), 'rmse': float(rmse_2h), 'nrmse': float(nrmse_2h)},
            '4h': {'mae': float(mae_4h), 'rmse': float(rmse_4h), 'nrmse': float(nrmse_4h)}
        },
        'cluster_distribution': cluster_stats_dist,
        'cluster_semantic_map': cluster_semantic_map,
        'expert_training': {
            k: {
                'best_epoch': r.best_epoch,
                'best_val_loss': float(r.best_val_loss),
                'train_samples': r.train_samples,
                'val_samples': r.val_samples
            }
            for k, r in expert_results.items()
        },
        'cluster_metrics_test': [
            {
                'cluster_id': cm.cluster_id,
                'n_samples': cm.n_samples,
                'nrmse_1h': float(cm.nrmse_1h),
                'nrmse_2h': float(cm.nrmse_2h),
                'nrmse_4h': float(cm.nrmse_4h),
                'avg_membership': float(cm.avg_membership)
            }
            for cm in cluster_metrics
        ]
    }
    with open(results_dir / "metrics_B4.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_b4, f, indent=2, ensure_ascii=False)
    print(f"  Saved: metrics_B4.json")
    
    # metrics_B4.csv
    import pandas as pd
    df_metrics = pd.DataFrame([
        {
            'cluster_id': cm.cluster_id,
            'n_samples': cm.n_samples,
            'ratio': cm.n_samples / len(predictions_all),
            'mae_1h': cm.mae_1h, 'rmse_1h': cm.rmse_1h, 'nrmse_1h': cm.nrmse_1h,
            'mae_2h': cm.mae_2h, 'rmse_2h': cm.rmse_2h, 'nrmse_2h': cm.nrmse_2h,
            'mae_4h': cm.mae_4h, 'rmse_4h': cm.rmse_4h, 'nrmse_4h': cm.nrmse_4h,
            'avg_membership': cm.avg_membership
        }
        for cm in cluster_metrics
    ])
    df_metrics.to_csv(results_dir / "metrics_B4.csv", index=False, encoding='utf-8')
    print(f"  Saved: metrics_B4.csv")
    
    # 训练日志
    logs_dir = PROJECT_ROOT / output_config['logs_dir']
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    train_log = {
        'experiment': 'B4',
        'timestamp': timestamp,
        'config': {
            'model': model_config,
            'training': train_config,
            'n_clusters': n_clusters
        },
        'expert_results': {
            k: {
                'best_epoch': r.best_epoch,
                'best_val_loss': float(r.best_val_loss),
                'best_val_rmse_4h_pu': float(r.best_val_rmse_4h_pu),
                'stopped_early': r.stopped_early,
                'total_epochs': r.total_epochs,
                'train_samples': r.train_samples,
                'val_samples': r.val_samples
            }
            for k, r in expert_results.items()
        },
        'test_metrics': {
            'overall': {
                '1h_nrmse': float(nrmse_1h),
                '2h_nrmse': float(nrmse_2h),
                '4h_nrmse': float(nrmse_4h)
            },
            'by_cluster': {
                cm.cluster_id: {
                    'n_samples': cm.n_samples,
                    '1h_nrmse': float(cm.nrmse_1h),
                    '2h_nrmse': float(cm.nrmse_2h),
                    '4h_nrmse': float(cm.nrmse_4h)
                }
                for cm in cluster_metrics
            }
        }
    }
    with open(logs_dir / f"train_log_b4_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(train_log, f, indent=2, ensure_ascii=False)
    print(f"  Saved: train_log_b4_{timestamp}.json")
    
    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 70)
    print("B4 阶段完成!")
    print(f"  - 专家数: {n_clusters}")
    print(f"  - Test 整体 4h nRMSE: {nrmse_4h:.4f} ({nrmse_4h*100:.2f}%)")
    for cm in cluster_metrics:
        print(f"  - Expert {cm.cluster_id}: {cm.n_samples} 样本, 4h nRMSE={cm.nrmse_4h:.4f}")
    print("=" * 70)
    
    return metrics_b4


if __name__ == "__main__":
    main()
