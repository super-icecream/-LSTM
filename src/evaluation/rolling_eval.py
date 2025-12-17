# -*- coding: utf-8 -*-
"""
Rolling Forecasting 评估器
在测试集上对每个锚点进行预测，并分步长计算指标
"""

import numpy as np
from typing import List, Dict, Callable, Any
from dataclasses import dataclass

from .metrics import compute_metrics, MetricsResult
from ..data.window import WindowSample


@dataclass
class HorizonMetrics:
    """单个步长的评估结果"""
    horizon: int
    horizon_name: str
    metrics: MetricsResult


@dataclass
class RollingEvalResult:
    """Rolling 评估结果"""
    horizon_results: List[HorizonMetrics]
    overall_metrics: MetricsResult
    total_samples: int


class RollingEvaluator:
    """
    Rolling Forecasting 评估器
    
    为什么用 rolling forecasting：
    1. 项目技术路线明确要求"rolling forecasting 作为唯一评估方式"
    2. 这模拟了真实场景：每个时刻都做一次预测，而不是只在固定点预测
    3. 这样评估更公平、更接近实际部署
    
    评估逻辑：
    1. 遍历 test 集中每一个锚点 t
    2. 对每个锚点：调用模型预测，得到预测值
    3. 按步长分组（1h=4步, 2h=8步, 4h=16步），分别计算指标
    4. 同时计算 overall（所有步长平均）
    
    严格约束：
    - 禁止使用任何未来真实观测信息
    - 只能用历史数据做预测
    """
    
    def __init__(
        self,
        horizons: List[int],
        horizon_names: Dict[int, str],
        nominal_capacity: float = 50.0
    ):
        """
        Args:
            horizons: 评估的步长列表，如 [4, 8, 16]
            horizon_names: 步长名称映射，如 {4: "1h", 8: "2h", 16: "4h"}
            nominal_capacity: 装机容量（MW）
        """
        self.horizons = sorted(horizons)
        self.horizon_names = horizon_names
        self.nominal_capacity = nominal_capacity
    
    def evaluate(
        self,
        samples: List[WindowSample],
        predict_fn: Callable[[np.ndarray], np.ndarray]
    ) -> RollingEvalResult:
        """
        在样本集上进行 rolling evaluation
        
        Args:
            samples: 滑窗样本列表
            predict_fn: 预测函数，输入 X [Lx, features]，输出 [max_horizon]
        
        Returns:
            RollingEvalResult: 包含各步长和整体指标
        """
        # 收集每个步长的真实值和预测值
        horizon_y_true: Dict[int, List[float]] = {h: [] for h in self.horizons}
        horizon_y_pred: Dict[int, List[float]] = {h: [] for h in self.horizons}
        
        # 遍历所有样本
        for sample in samples:
            # 调用模型预测
            predictions = predict_fn(sample.X)
            
            # 按步长收集
            for h in self.horizons:
                # horizon h 表示预测第 h 步（0-indexed: h-1）
                # 但这里 h 直接对应步数，如 h=4 表示预测 4 步后
                # Y[h-1] 是第 h 步的真实值
                if h <= len(sample.Y):
                    horizon_y_true[h].append(sample.Y[h - 1])
                    horizon_y_pred[h].append(predictions[h - 1])
        
        # 计算每个步长的指标
        horizon_results = []
        all_y_true = []
        all_y_pred = []
        
        for h in self.horizons:
            y_true = np.array(horizon_y_true[h])
            y_pred = np.array(horizon_y_pred[h])
            
            metrics = compute_metrics(y_true, y_pred, self.nominal_capacity)
            
            horizon_results.append(HorizonMetrics(
                horizon=h,
                horizon_name=self.horizon_names.get(h, f"{h}step"),
                metrics=metrics
            ))
            
            all_y_true.extend(y_true.tolist())
            all_y_pred.extend(y_pred.tolist())
        
        # 计算整体指标
        overall_metrics = compute_metrics(
            np.array(all_y_true),
            np.array(all_y_pred),
            self.nominal_capacity
        )
        
        return RollingEvalResult(
            horizon_results=horizon_results,
            overall_metrics=overall_metrics,
            total_samples=len(samples)
        )


def print_eval_table(result: RollingEvalResult, model_name: str = "Model") -> None:
    """以表格形式打印评估结果"""
    print("\n" + "=" * 70)
    print(f"{model_name} - Rolling Evaluation Results")
    print("=" * 70)
    print(f"Total test samples: {result.total_samples}")
    print("-" * 70)
    print(f"{'Horizon':<12} {'MAE (MW)':<15} {'RMSE (MW)':<15} {'nRMSE':<15}")
    print("-" * 70)
    
    for hr in result.horizon_results:
        print(f"{hr.horizon_name:<12} {hr.metrics.mae:<15.4f} {hr.metrics.rmse:<15.4f} {hr.metrics.nrmse:<15.4f}")
    
    print("-" * 70)
    om = result.overall_metrics
    print(f"{'Overall':<12} {om.mae:<15.4f} {om.rmse:<15.4f} {om.nrmse:<15.4f}")
    print("=" * 70)
