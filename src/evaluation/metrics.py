# -*- coding: utf-8 -*-
"""
评估指标计算模块
包含 MAE、RMSE、nRMSE 的计算
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricsResult:
    """评估指标结果"""
    mae: float
    rmse: float
    nrmse: float  # 归一化 RMSE（按装机容量）
    n_samples: int


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nominal_capacity: float = 50.0
) -> MetricsResult:
    """
    计算评估指标
    
    公式：
    - MAE = mean(|y_true - y_pred|)
    - RMSE = sqrt(mean((y_true - y_pred)^2))
    - nRMSE = RMSE / nominal_capacity（按装机容量归一化）
    
    为什么用 nRMSE：
    - 项目技术路线要求"容量归一化"
    - 这样不同装机容量的电站可以横向比较
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        nominal_capacity: 装机容量（MW），用于归一化
    
    Returns:
        MetricsResult: 包含 MAE、RMSE、nRMSE 的结果
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    assert len(y_true) == len(y_pred), "真实值和预测值长度必须一致"
    
    n_samples = len(y_true)
    
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # nRMSE: Normalized RMSE（按装机容量归一化）
    nrmse = rmse / nominal_capacity
    
    return MetricsResult(
        mae=float(mae),
        rmse=float(rmse),
        nrmse=float(nrmse),
        n_samples=n_samples
    )
