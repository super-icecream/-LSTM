# -*- coding: utf-8 -*-
"""
结果输出模块
将评估结果保存为 JSON / CSV 文件
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..evaluation.rolling_eval import RollingEvalResult


def save_results(
    result: RollingEvalResult,
    output_dir: str,
    model_name: str = "persistence",
    experiment_name: str = "b0",
    daylight_config: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    将评估结果保存为 JSON 和 CSV 文件
    
    Args:
        result: Rolling 评估结果
        output_dir: 输出目录
        model_name: 模型名称
        experiment_name: 实验名称
        daylight_config: 白天筛选配置（可选）
    
    Returns:
        保存的文件路径字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 根据白天筛选配置生成文件名后缀
    daylight_suffix = ""
    if daylight_config and daylight_config.get('enabled', False):
        threshold = daylight_config.get('threshold', 5)
        mode = daylight_config.get('mode', 'mask')
        # 根据列名生成简称：Total solar irradiance -> tsi, Direct normal irradiance -> dni
        col_name = daylight_config.get('dni_col', '')
        if 'Total solar irradiance' in col_name:
            col_abbr = 'tsi'
        elif 'Direct normal irradiance' in col_name:
            col_abbr = 'dni'
        else:
            col_abbr = 'irr'
        daylight_suffix = f"_daylight_{col_abbr}{threshold}_{mode}"
    
    base_name = f"{experiment_name}_{model_name}{daylight_suffix}_{timestamp}"
    
    # 构建结果字典
    results_dict = {
        "model_name": model_name,
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "total_samples": result.total_samples,
        "daylight_filter": daylight_config if daylight_config else {"enabled": False},
        "horizons": {},
        "overall": {
            "mae": result.overall_metrics.mae,
            "rmse": result.overall_metrics.rmse,
            "nrmse": result.overall_metrics.nrmse,
            "n_samples": result.overall_metrics.n_samples
        }
    }
    
    for hr in result.horizon_results:
        results_dict["horizons"][hr.horizon_name] = {
            "horizon_steps": hr.horizon,
            "mae": hr.metrics.mae,
            "rmse": hr.metrics.rmse,
            "nrmse": hr.metrics.nrmse,
            "n_samples": hr.metrics.n_samples
        }
    
    # 保存 JSON
    json_path = output_path / f"{base_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    # 保存 CSV
    csv_path = output_path / f"{base_name}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Horizon', 'Steps', 'MAE', 'RMSE', 'nRMSE', 'N_Samples'])
        
        for hr in result.horizon_results:
            writer.writerow([
                hr.horizon_name,
                hr.horizon,
                f"{hr.metrics.mae:.4f}",
                f"{hr.metrics.rmse:.4f}",
                f"{hr.metrics.nrmse:.4f}",
                hr.metrics.n_samples
            ])
        
        writer.writerow([
            'Overall',
            '-',
            f"{result.overall_metrics.mae:.4f}",
            f"{result.overall_metrics.rmse:.4f}",
            f"{result.overall_metrics.nrmse:.4f}",
            result.overall_metrics.n_samples
        ])
    
    print(f"\nResults saved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")
    
    return {
        "json": str(json_path),
        "csv": str(csv_path)
    }
