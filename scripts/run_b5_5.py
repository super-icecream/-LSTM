# -*- coding: utf-8 -*-
"""
B5.5 阶段主入口脚本
Warm-Start 专家模型训练 + Hard vs Soft 路由对比

与 B4 的区别：
- B4: 从头训练专家模型
- B5.5: 从 B2 全局模型 warm-start，微调专家模型

与 B5 的区别：
- B5: 直接加载 B4 训练好的专家模型
- B5.5: 重新训练 warm-start 专家模型

流程：
1. 复用 B4 的数据处理/路由分簇
2. Warm-start: 加载 model_global.pt 初始化每个专家
3. 微调专家模型 (按簇 train/val)
4. 复用 B5 的 Hard vs Soft rolling evaluation
5. 输出 metrics_B5_5.json/csv
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import pickle
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

from src.data.loader import load_solar_data
from src.data.splitter import split_by_day
from src.data.window import generate_windows, WindowSample
from src.data.daylight import add_daylight_flag, filter_daylight_rows
from src.models.lstm import GlobalLSTM
from src.utils.seed import set_seed, print_seed_info, get_dataloader_generator, worker_init_fn


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# 状态向量 g_t 构建 (与 B3/B4/B5 完全一致)
# =============================================================================

def compute_state_vector(sample: WindowSample, feature_names: List[str], capacity: float) -> Tuple[np.ndarray, List[str]]:
    """计算单个样本的状态向量 g_t"""
    X = sample.X
    
    g_values = []
    g_names = []
    
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
    
    irr_indices = [('TSI', tsi_idx), ('DNI', dni_idx), ('GHI', ghi_idx)]
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            g_values.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
            g_names.extend([f'{name}_mean', f'{name}_std', f'{name}_min', f'{name}_max'])
    
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            delta = np.diff(vals)
            if len(delta) > 0:
                g_values.extend([np.mean(np.abs(delta)), np.std(delta), np.max(np.abs(delta))])
                g_names.extend([f'{name}_delta_mean_abs', f'{name}_delta_std', f'{name}_delta_max_abs'])
    
    if power_idx is not None:
        power_pu = X[:, power_idx] / capacity
        g_values.extend([power_pu.mean(), power_pu.std()])
        g_names.extend(['Power_pu_mean', 'Power_pu_std'])
        delta_power = np.diff(power_pu)
        if len(delta_power) > 0:
            g_values.append(np.mean(np.abs(delta_power)))
            g_names.append('Power_pu_delta_mean_abs')
    
    if temp_idx is not None:
        temp = X[:, temp_idx]
        g_values.extend([temp.mean(), temp.std()])
        g_names.extend(['Temp_mean', 'Temp_std'])
    
    if atm_idx is not None:
        atm = X[:, atm_idx]
        g_values.extend([atm.mean(), atm.std()])
        g_names.extend(['Atm_mean', 'Atm_std'])
    
    return np.array(g_values, dtype=np.float32), g_names


def compute_fcm_membership(z: np.ndarray, centers: np.ndarray, m: float = 2.0) -> np.ndarray:
    """计算 FCM 隶属度向量 (与 B4 一致，使用平方距离)"""
    K = centers.shape[0]
    distances = np.zeros(K)
    for k in range(K):
        diff = z - centers[k]
        distances[k] = np.sum(diff ** 2)
    distances = np.maximum(distances, 1e-10)
    power = 2.0 / (m - 1)
    u = np.zeros(K)
    for k in range(K):
        u[k] = 1.0 / np.sum((distances[k] / distances) ** power)
    return u


def adjust_membership(u: np.ndarray, temperature: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    """调整隶属度向量: 先温度缩放，再幂次调整"""
    eps = 1e-10
    u_adjusted = u.copy()
    
    if temperature != 1.0:
        log_u = np.log(u_adjusted + eps)
        scaled = log_u / temperature
        scaled = scaled - np.max(scaled)
        exp_scaled = np.exp(scaled)
        u_adjusted = exp_scaled / np.sum(exp_scaled)
    
    if alpha != 1.0:
        u_pow = np.power(u_adjusted, alpha)
        u_adjusted = u_pow / np.sum(u_pow)
    
    return u_adjusted


# =============================================================================
# Warm-Start 专家训练
# =============================================================================

@dataclass
class ExpertTrainResult:
    cluster_id: int
    best_epoch: int
    best_val_loss: float
    train_samples: int
    val_samples: int


def train_expert_warm(
    cluster_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    global_model_path: Path,
    device: str,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 0.001,
    lr_scale_finetune: float = 0.1,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
    seed: int = 42
) -> Tuple[GlobalLSTM, ExpertTrainResult]:
    """
    Warm-start 训练单个专家模型
    
    Args:
        global_model_path: B2 全局模型路径
        lr_scale_finetune: 微调学习率缩放因子
    """
    input_size = X_train.shape[2]
    output_steps = y_train.shape[1]
    
    # 创建模型 (与 B2 全局模型结构一致)
    model = GlobalLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_steps=output_steps,
        dropout=dropout
    )
    
    # Warm-start: 加载全局模型权重
    if global_model_path.exists():
        global_state = torch.load(global_model_path, map_location=device, weights_only=True)
        model.load_state_dict(global_state)
        print(f"  [Warm-start] Loaded global model from {global_model_path.name}")
    else:
        print(f"  [Warning] Global model not found: {global_model_path}, training from scratch")
    
    model.to(device)
    
    # DataLoader
    generator = get_dataloader_generator(seed + cluster_id)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=generator, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 微调学习率
    finetune_lr = lr * lr_scale_finetune
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0
    
    print(f"\n  [Expert {cluster_id}] Train: {len(X_train)}, Val: {len(X_val)}, LR: {finetune_lr:.6f}")
    print(f"   {'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'4h RMSE(pu)':>12}")
    print(f"  {'-'*50}")
    
    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        rmse_4h_pu = np.sqrt(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"   {epoch:>5} | {train_loss:>12.6f} | {val_loss:>12.6f} | {rmse_4h_pu:>12.6f} *")
        else:
            no_improve += 1
            if epoch <= 5 or epoch % 10 == 0:
                print(f"   {epoch:>5} | {train_loss:>12.6f} | {val_loss:>12.6f} | {rmse_4h_pu:>12.6f}")
        
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # 加载最优权重
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [Checkpoint] 评估使用: best epoch 权重 (epoch={best_epoch})")
    
    print(f"  Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}")
    
    result = ExpertTrainResult(
        cluster_id=cluster_id,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_samples=len(X_train),
        val_samples=len(X_val)
    )
    
    return model, result


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("B5.5 阶段 - Warm-Start 专家模型 + Hard vs Soft 路由对比")
    print("=" * 70)
    
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    print(f"\n[1/10] 配置加载完成")
    
    data_config = config['data']
    split_config = config['split']
    window_config = config['window']
    eval_config = config['evaluation']
    output_config = config['output']
    daylight_config = config.get('daylight_filter', {'enabled': False})
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    
    daylight_enabled = daylight_config.get('enabled', False)
    daylight_mode = daylight_config.get('mode', 'mask') if daylight_enabled else None
    
    target_col = data_config['target_column']
    capacity = data_config['nominal_capacity_mw']
    n_clusters = 3
    horizons = eval_config['horizons']
    
    # 软路由配置
    soft_routing_config = config.get('soft_routing', {})
    routing_mode = soft_routing_config.get('routing_mode', 'soft')  # "hard" or "soft"
    soft_temperature = soft_routing_config.get('temperature', 1.0)
    soft_alpha = soft_routing_config.get('alpha', 0.7)
    
    # 验证 routing_mode
    if routing_mode not in ('hard', 'soft'):
        print(f"  [Warning] Invalid routing_mode '{routing_mode}', defaulting to 'soft'")
        routing_mode = 'soft'
    
    # Warm-start 配置 (可在 config.yaml 中添加)
    warm_start_config = config.get('warm_start', {})
    warm_start_enabled = warm_start_config.get('enabled', True)
    warm_start_path = warm_start_config.get('path', 'experiments/results/model_global.pt')
    lr_scale_finetune = warm_start_config.get('lr_scale_finetune', 0.1)
    
    # 随机种子设置
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
    
    print(f"  - Device: {device}")
    if device == "cuda":
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - n_clusters: {n_clusters}")
    print(f"  - Warm-start: {warm_start_enabled} (lr_scale={lr_scale_finetune})")
    print(f"  - Routing mode: {routing_mode.upper()}")
    if routing_mode == 'soft':
        print(f"  - Soft routing params: T={soft_temperature}, alpha={soft_alpha}")
    else:
        print(f"  - Hard routing: argmax expert selection (T/alpha ignored)")
    
    # =========================================================================
    # 2. 加载 Router 组件 (从 B3 产物)
    # =========================================================================
    print("\n[2/10] 加载 Router 组件...")
    
    results_dir = PROJECT_ROOT / output_config['results_dir']
    
    with open(results_dir / "router_config.json", 'r', encoding='utf-8') as f:
        router_config_data = json.load(f)
    
    cluster_semantic_map = {int(k): v for k, v in router_config_data['cluster_semantic_map'].items()}
    
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
    
    # 簇语义映射
    print(f"\n  簇语义映射:")
    for k in range(n_clusters):
        print(f"    Cluster {k}: {cluster_semantic_map.get(k, 'unknown')}")
    
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
        if daylight_mode == 'drop':
            filtered_df, _, _ = filter_daylight_rows(daylight_result.df)
            df_for_split = filtered_df
        else:
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
    
    train_window_result = generate_windows(
        df=split_result.train_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=(daylight_mode == 'mask') if daylight_enabled else False
    )
    
    val_window_result = generate_windows(
        df=split_result.val_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=(daylight_mode == 'mask') if daylight_enabled else False
    )
    
    test_window_result = generate_windows(
        df=split_result.test_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=(daylight_mode == 'mask') if daylight_enabled else False
    )
    
    train_samples = train_window_result.samples
    val_samples = val_window_result.samples
    test_samples = test_window_result.samples
    
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # =========================================================================
    # 6. 构建状态向量 + 路由分簇
    # =========================================================================
    print("\n[6/10] 构建状态向量 + 路由分簇...")
    
    def route_samples(samples, scaler_g, pca, fcm_centers):
        labels = []
        memberships = []
        for sample in samples:
            g_t, _ = compute_state_vector(sample, feature_names, capacity)
            g_scaled = scaler_g.transform(g_t.reshape(1, -1))[0]
            z_t = pca.transform(g_scaled.reshape(1, -1))[0]
            u_t = compute_fcm_membership(z_t, fcm_centers, m=2.0)
            labels.append(np.argmax(u_t))
            memberships.append(u_t)
        return np.array(labels), np.array(memberships)
    
    labels_train, memb_train = route_samples(train_samples, scaler_g, pca, fcm_centers)
    labels_val, memb_val = route_samples(val_samples, scaler_g, pca, fcm_centers)
    labels_test, memb_test = route_samples(test_samples, scaler_g, pca, fcm_centers)
    
    # 统计
    print(f"\n  Cluster distribution:")
    print(f"  {'Cluster':<12} {'Semantic':>15} {'Train':>10} {'Val':>10} {'Test':>10}")
    print(f"  {'-'*60}")
    for k in range(n_clusters):
        semantic = cluster_semantic_map.get(k, 'unknown')
        n_train = np.sum(labels_train == k)
        n_val = np.sum(labels_val == k)
        n_test = np.sum(labels_test == k)
        print(f"  Cluster {k:<3} {semantic:>15} {n_train:>10} {n_val:>10} {n_test:>10}")
    
    # =========================================================================
    # 7. 准备训练数据
    # =========================================================================
    print("\n[7/10] 准备训练数据...")
    
    def prepare_data(samples, scaler_X, capacity):
        X_list, y_list = [], []
        for sample in samples:
            X_scaled = scaler_X.transform(sample.X)
            y_pu = sample.Y / capacity
            X_list.append(X_scaled)
            y_list.append(y_pu)
        return np.array(X_list), np.array(y_list)
    
    X_train_all, y_train_all = prepare_data(train_samples, scaler_X, capacity)
    X_val_all, y_val_all = prepare_data(val_samples, scaler_X, capacity)
    X_test_all, y_test_all = prepare_data(test_samples, scaler_X, capacity)
    
    # =========================================================================
    # 8. Warm-Start 训练专家模型
    # =========================================================================
    print("\n[8/10] Warm-Start 训练专家模型...")
    
    global_model_path = PROJECT_ROOT / warm_start_path
    
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.1)
    lr = train_config.get('lr', 0.001)
    batch_size = train_config.get('batch_size', 256)
    max_epochs = train_config.get('max_epochs', 100)
    patience = train_config.get('patience', 10)
    
    experts = {}
    train_results = []
    
    for k in range(n_clusters):
        print(f"\n{'='*60}")
        print(f"Training Expert {k} ({cluster_semantic_map.get(k, 'unknown')})")
        print(f"{'='*60}")
        
        # 按簇筛选数据
        train_mask = labels_train == k
        val_mask = labels_val == k
        
        X_train_k = X_train_all[train_mask]
        y_train_k = y_train_all[train_mask]
        X_val_k = X_val_all[val_mask]
        y_val_k = y_val_all[val_mask]
        
        model, result = train_expert_warm(
            cluster_id=k,
            X_train=X_train_k,
            y_train=y_train_k,
            X_val=X_val_k,
            y_val=y_val_k,
            global_model_path=global_model_path,
            device=device,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            lr_scale_finetune=lr_scale_finetune,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed
        )
        
        experts[k] = model
        train_results.append(result)
    
    # =========================================================================
    # 9. Rolling Evaluation (根据 routing_mode 选择评估方式)
    # =========================================================================
    print(f"\n[9/10] Rolling Evaluation (routing_mode={routing_mode.upper()})...")
    
    preds = []
    targets = []
    
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            k_t = labels_test[i]  # argmax 选择的专家
            
            # 准备输入
            X_tensor = torch.FloatTensor(X_test_all[i:i+1]).to(device)
            
            if routing_mode == 'hard':
                # Hard 模式: 仅用 argmax 选择的专家输出
                pred_pu = experts[k_t](X_tensor).cpu().numpy().flatten()
            else:
                # Soft 模式: 加权融合多专家输出
                u_t = memb_test[i]
                u_adjusted = adjust_membership(u_t, temperature=soft_temperature, alpha=soft_alpha)
                pred_pu = np.zeros(y_test_all.shape[1])
                for k in range(n_clusters):
                    expert_pred_pu = experts[k](X_tensor).cpu().numpy().flatten()
                    pred_pu += u_adjusted[k] * expert_pred_pu
            
            pred_mw = pred_pu * capacity
            y_true_mw = sample.Y
            
            preds.append(pred_mw)
            targets.append(y_true_mw)
    
    preds = np.array(preds)
    targets = np.array(targets)
    
    print(f"  Completed: {len(targets)} samples")
    
    # =========================================================================
    # 10. 计算指标 + 汇总表 + 保存
    # =========================================================================
    print("\n[10/10] 计算指标 + 汇总表 + 保存产物...")
    
    def compute_metrics(preds, targets, horizons, capacity):
        horizon_names = {4: '1h', 8: '2h', 16: '4h'}
        results = {}
        for h in horizons:
            name = horizon_names.get(h, f"{h}step")
            pred_h = preds[:, :h]
            target_h = targets[:, :h]
            errors = pred_h - target_h
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            nrmse = rmse / capacity
            results[name] = {'mae': mae, 'rmse': rmse, 'nrmse': nrmse}
        return results
    
    # 整体指标 (仅计算当前 routing_mode)
    overall_metrics = compute_metrics(preds, targets, horizons, capacity)
    
    # 按簇指标
    by_cluster = {}
    for k in range(n_clusters):
        mask = labels_test == k
        n_samples = np.sum(mask)
        if n_samples > 0:
            by_cluster[k] = {
                'n_samples': int(n_samples),
                'semantic': cluster_semantic_map.get(k, 'unknown'),
                'avg_membership': float(np.mean(memb_test[mask, k])),
                **compute_metrics(preds[mask], targets[mask], horizons, capacity)
            }
    
    # 打印汇总表
    print("\n" + "=" * 80)
    print(f"B5.5 汇总表 - Warm-Start 专家模型 (routing_mode={routing_mode.upper()})")
    print("=" * 80)
    
    print(f"\n[整体误差] (routing_mode={routing_mode.upper()})")
    print(f"{'Horizon':<10} {'MAE (MW)':>12} {'RMSE (MW)':>12} {'nRMSE':>12}")
    print("-" * 50)
    
    for h in horizons:
        horizon_names = {4: '1h', 8: '2h', 16: '4h'}
        name = horizon_names.get(h, f"{h}step")
        m = overall_metrics[name]
        print(f"{name:<10} {m['mae']:>12.4f} {m['rmse']:>12.4f} {m['nrmse']:>12.4f}")
    
    print(f"\n[按簇误差] (routing_mode={routing_mode.upper()})")
    print(f"{'Cluster':<10} {'Semantic':>15} {'N':>8} {'1h nRMSE':>12} {'2h nRMSE':>12} {'4h nRMSE':>12} {'Avg Memb':>10}")
    print("-" * 85)
    for k in range(n_clusters):
        if k in by_cluster:
            m = by_cluster[k]
            print(f"Cluster {k:<2} {m['semantic']:>15} {m['n_samples']:>8} "
                  f"{m['1h']['nrmse']:>12.4f} {m['2h']['nrmse']:>12.4f} {m['4h']['nrmse']:>12.4f} "
                  f"{m['avg_membership']:>10.4f}")
    
    print("=" * 80)
    
    # 保存专家模型
    for k in range(n_clusters):
        model_path = results_dir / f"model_expert_{k}_warm.pt"
        torch.save(experts[k].state_dict(), model_path)
        print(f"  Saved: {model_path.name}")
    
    # 保存 JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metrics_data = {
        'timestamp': timestamp,
        'routing_mode': routing_mode,
        'config': {
            'warm_start_enabled': warm_start_enabled,
            'warm_start_path': str(warm_start_path),
            'lr_scale_finetune': lr_scale_finetune,
            'routing_mode': routing_mode,
            'temperature_T': soft_temperature if routing_mode == 'soft' else None,
            'alpha': soft_alpha if routing_mode == 'soft' else None,
            'n_clusters': n_clusters,
            'n_test_samples': len(targets)
        },
        'cluster_semantic_map': cluster_semantic_map,
        'expert_training': {
            str(r.cluster_id): {
                'best_epoch': r.best_epoch,
                'best_val_loss': r.best_val_loss,
                'train_samples': r.train_samples,
                'val_samples': r.val_samples
            }
            for r in train_results
        },
        'overall': overall_metrics,
        'by_cluster': {str(k): v for k, v in by_cluster.items()}
    }
    
    json_path = results_dir / "metrics_B5_5.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path.name}")
    
    # 保存 CSV
    csv_path = results_dir / "metrics_B5_5.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("routing_mode,Cluster,Semantic,N,1h_MAE,1h_RMSE,1h_nRMSE,2h_MAE,2h_RMSE,2h_nRMSE,4h_MAE,4h_RMSE,4h_nRMSE,Avg_Membership\n")
        
        # Overall
        o = overall_metrics
        f.write(f"{routing_mode},Overall,,-,"
                f"{o['1h']['mae']:.4f},{o['1h']['rmse']:.4f},{o['1h']['nrmse']:.4f},"
                f"{o['2h']['mae']:.4f},{o['2h']['rmse']:.4f},{o['2h']['nrmse']:.4f},"
                f"{o['4h']['mae']:.4f},{o['4h']['rmse']:.4f},{o['4h']['nrmse']:.4f},-\n")
        
        # By cluster
        for k in range(n_clusters):
            if k in by_cluster:
                m = by_cluster[k]
                f.write(f"{routing_mode},{k},{m['semantic']},{m['n_samples']},"
                        f"{m['1h']['mae']:.4f},{m['1h']['rmse']:.4f},{m['1h']['nrmse']:.4f},"
                        f"{m['2h']['mae']:.4f},{m['2h']['rmse']:.4f},{m['2h']['nrmse']:.4f},"
                        f"{m['4h']['mae']:.4f},{m['4h']['rmse']:.4f},{m['4h']['nrmse']:.4f},{m['avg_membership']:.4f}\n")
    
    print(f"  Saved: {csv_path.name}")
    
    # 保存日志
    log_path = PROJECT_ROOT / output_config['logs_dir'] / f"train_log_b5_5_{timestamp}.json"
    log_data = {
        'timestamp': timestamp,
        'stage': 'B5.5',
        'routing_mode': routing_mode,
        'warm_start_enabled': warm_start_enabled,
        'lr_scale_finetune': lr_scale_finetune,
        'temperature_T': soft_temperature if routing_mode == 'soft' else None,
        'alpha': soft_alpha if routing_mode == 'soft' else None,
        'n_test_samples': len(targets),
        '4h_nrmse': overall_metrics['4h']['nrmse']
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {log_path.name}")
    
    # 最终总结
    print("\n" + "=" * 70)
    print("B5.5 阶段完成!")
    print(f"  - Routing mode: {routing_mode.upper()}")
    print(f"  - Warm-start: {warm_start_enabled} (lr_scale={lr_scale_finetune})")
    if routing_mode == 'soft':
        print(f"  - Soft routing params: T={soft_temperature}, alpha={soft_alpha}")
    else:
        print(f"  - Hard routing: argmax expert selection")
    print(f"  - Test 样本数: {len(targets)}")
    nrmse_4h = overall_metrics['4h']['nrmse']
    print(f"  - 4h nRMSE ({routing_mode.upper()}): {nrmse_4h:.4f} ({nrmse_4h*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
