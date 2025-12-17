# -*- coding: utf-8 -*-
"""
Persistence Baseline 模型
最简单的时间序列预测基准：预测值 = 当前功率
"""

import numpy as np
from typing import List


class PersistenceBaseline:
    """
    Persistence Baseline 模型
    
    预测逻辑：
    - 对于任意时刻 t，预测未来所有步长的功率均等于当前功率 P_{t-1}
    - 即：P_hat(t+h) = P_{t-1}，对所有 h in [0, 1, ..., Hmax-1]
    
    为什么用 Persistence：
    1. 这是时间序列预测的最简单 baseline
    2. 用于验证评估流程是否正确
    3. 后续所有模型必须超过 Persistence 才有意义
    """
    
    def __init__(self, max_horizon: int = 16):
        """
        Args:
            max_horizon: 最大预测步长
        """
        self.max_horizon = max_horizon
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        基于输入窗口进行预测
        
        Persistence 策略：取输入窗口最后一个时刻的功率作为所有步长的预测值
        
        Args:
            X: 输入窗口 [Lx, features]，假设功率是最后一列（或第 -1 列）
        
        Returns:
            预测值数组 [max_horizon]
        """
        # 假设功率是最后一列
        # X 的形状为 [Lx, features]，取最后一行最后一列作为当前功率
        current_power = X[-1, -1]
        
        # 预测未来所有步长的功率均等于当前功率
        predictions = np.full(self.max_horizon, current_power)
        
        return predictions
    
    def predict_at_horizon(self, X: np.ndarray, horizon: int) -> float:
        """
        预测指定步长的功率
        
        Args:
            X: 输入窗口
            horizon: 预测步长（1-indexed，即 horizon=1 表示下一个时刻）
        
        Returns:
            预测值
        """
        current_power = X[-1, -1]
        return current_power
    
    def __repr__(self):
        return f"PersistenceBaseline(max_horizon={self.max_horizon})"
