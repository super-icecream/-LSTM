"""
评估指标计算模块
实现DLFE-LSTM-WSI系统的性能评估指标计算
支持GPU加速和多时间尺度评估"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """评估指标结果数据类"""

    rmse: float
    mae: float
    nrmse: float
    confidence_interval: Tuple[float, float]
    std_error: float = 0.0
    max_error: float = 0.0
    min_error: float = 0.0
    nrmse_cap: Optional[float] = None  # 以装机容量归一化的 NRMSE（物理量纲）
    nmae_cap: Optional[float] = None  # 以装机容量归一化的 MAE（物理量纲）

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        data = {
            "RMSE": self.rmse,
            "MAE": self.mae,
            "NRMSE": self.nrmse,
            "CI_95%": self.confidence_interval,
            "STD": self.std_error,
            "MAX_ERROR": self.max_error,
            "MIN_ERROR": self.min_error,
        }
        if self.nrmse_cap is not None:
            data["NRMSE_cap"] = self.nrmse_cap
        if self.nmae_cap is not None:
            data["NMAE_cap"] = self.nmae_cap
        return data

    def __str__(self) -> str:
        """格式化输出"""
        parts = [f"RMSE: {self.rmse:.4f}", f"MAE: {self.mae:.4f}", f"NRMSE: {self.nrmse:.4f}"]
        if self.nrmse_cap is not None:
            parts.append(f"NRMSE_cap: {self.nrmse_cap:.4f}")
        if self.nmae_cap is not None:
            parts.append(f"NMAE_cap: {self.nmae_cap:.4f}")
        return " | ".join(parts)


class PerformanceMetrics:
    """
    GPU加速的性能评估指标计算类
    主输出（RMSE/MAE/NRMSE）规则：
    - 先在归一化尺度计算 raw 指标
    - 若同时提供 power_scale 与 prated，则将误差还原到物理功率并按装机容量归一，主输出为容量归一结果（此时 NRMSE=RMSE）
    - 否则主输出保留为归一化尺度 raw 值并给出告警
    兼容字段：
    - NRMSE_cap/NMAE_cap 保留用于兼容，恒等于主输出
    """

    def __init__(self, device: str = "cuda", epsilon: float = 1e-8, prated: Optional[float] = None, power_scale: Optional[float] = None):
        """
        初始化评估器

        Args:
            device: 计算设备 ('cuda' 或 'cpu')
            epsilon: 防止除零的小数
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.prated = prated
        self.power_scale = power_scale
        self.results_history: List[MetricsResult] = []

        if self.device.type == "cuda":
            logger.info(f"使用GPU加速: {torch.cuda.get_device_name()}")
        else:
            logger.warning("CUDA不可用，使用CPU计算")

    def _ensure_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """确保数据为GPU张量"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if not data.is_cuda and self.device.type == "cuda":
            data = data.to(self.device, non_blocking=True)
        return data

    @torch.no_grad()
    def calculate_rmse(self, predictions: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> float:
        """计算RMSE"""
        predictions = self._ensure_tensor(predictions)
        targets = self._ensure_tensor(targets)
        mse = torch.mean((predictions - targets) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()

    @torch.no_grad()
    def calculate_mae(self, predictions: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> float:
        """计算MAE"""
        predictions = self._ensure_tensor(predictions)
        targets = self._ensure_tensor(targets)
        mae = torch.mean(torch.abs(predictions - targets))
        return mae.item()

    @torch.no_grad()
    def calculate_nrmse(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        normalization: str = "range",
    ) -> float:
        """
        计算NRMSE

        Args:
            predictions: 预测值
            targets: 真实值
            normalization: 'range' | 'mean' | 'std'
        """
        predictions = self._ensure_tensor(predictions)
        targets = self._ensure_tensor(targets)

        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))

        if normalization == "range":
            denominator = torch.max(targets) - torch.min(targets)
        elif normalization == "mean":
            denominator = torch.mean(torch.abs(targets))
        elif normalization == "std":
            denominator = torch.std(targets)
        else:
            raise ValueError(f"未知的归一化方式: {normalization}")

        denominator = torch.clamp(denominator, min=self.epsilon)
        nrmse = rmse / denominator
        return nrmse.item()

    @torch.no_grad()
    def calculate_all_metrics(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        calculate_ci: bool = True,
        prated: Optional[float] = None,
        power_scale: Optional[float] = None,
    ) -> MetricsResult:
        """
        计算所有评估指标
        主输出规则：
        - 先在归一化尺度计算 rmse_raw/mae_raw/nrmse_raw；
        - 若同时提供 power_scale 与 prated，则误差还原到物理功率并按容量归一：
          rmse = rmse_raw * power_scale / prated；mae = mae_raw * power_scale / prated；nrmse = rmse；
        - 否则保留 raw 值并给出告警；
        兼容：nrmse_cap/nmae_cap 与主输出同值。
        Args:
            predictions: 预测值
            targets: 真实值
            calculate_ci: 是否计算置信区间
            prated: 装机容量（建议单位 kW）
            power_scale: 训练集功率缩放因子（如 MinMax 的 max-min）
        """
        predictions = self._ensure_tensor(predictions)
        targets = self._ensure_tensor(targets)

        # 1) 归一化尺度的 raw 指标
        rmse_raw = self.calculate_rmse(predictions, targets)
        mae_raw = self.calculate_mae(predictions, targets)
        nrmse_raw = self.calculate_nrmse(predictions, targets)

        # 2) 还原到物理功率并按容量归一（若可用）
        effective_prated = prated if prated is not None else self.prated
        effective_power_scale = power_scale if power_scale is not None else getattr(self, "power_scale", None)

        use_capacity_norm = (
            effective_prated is not None
            and effective_power_scale is not None
            and effective_prated > self.epsilon
        )

        if use_capacity_norm:
            rmse_phys = rmse_raw * float(effective_power_scale)
            mae_phys = mae_raw * float(effective_power_scale)
            rmse = rmse_phys / float(effective_prated)
            mae = mae_phys / float(effective_prated)
            nrmse = rmse  # 按容量归一后，NRMSE 等同于 RMSE
        else:
            # 缺少 prated 或 power_scale，或 prated 非法：保留 raw
            rmse = rmse_raw
            mae = mae_raw
            nrmse = nrmse_raw
            if effective_prated is None or effective_power_scale is None:
                logger.warning(
                    "缺少 prated 或 power_scale，主指标保留为归一化尺度（raw）。prated=%s, power_scale=%s",
                    str(effective_prated), str(effective_power_scale)
                )
            elif effective_prated <= self.epsilon:
                logger.warning("prated 无效（%.6f），主指标保留为归一化尺度（raw）", float(effective_prated))
        errors = (predictions - targets).cpu().numpy()
        std_error = float(np.std(errors))
        max_error = float(np.max(np.abs(errors)))
        min_error = float(np.min(np.abs(errors)))

        if calculate_ci:
            ci = self.calculate_confidence_interval(torch.from_numpy(errors).to(self.device))
        else:
            ci = (0.0, 0.0)

        result = MetricsResult(
            rmse=rmse,
            mae=mae,
            nrmse=nrmse,
            confidence_interval=ci,
            std_error=std_error,
            max_error=max_error,
            min_error=min_error,
            nrmse_cap=nrmse,  # 兼容字段：与主输出一致
            nmae_cap=mae,      # 兼容字段：与主输出一致
        )
        self.results_history.append(result)
        return result

    @torch.no_grad()
    def evaluate_multi_horizon(
        self,
        model_dict: Dict[str, torch.nn.Module],
        sequence_sets: Dict[str, np.ndarray],
        horizons: List[int] = [1, 2, 4],
        prated: Optional[float] = None,
        power_scale: Optional[float] = None,
    ) -> Dict[int, MetricsResult]:
        """
        多时间尺度评估
        Args:
            model_dict: 三个天气子模型{'sunny','cloudy','overcast'}
            sequence_sets: 包含features/targets/weather的测试数据
            horizons: 预测时域列表
        """
        features = sequence_sets["features"]
        targets = sequence_sets["targets"]
        weather = sequence_sets.get("weather")

        results: Dict[int, MetricsResult] = {}

        # 按 horizons 的索引取列，避免将“步长值”误当作列号导致高步长被跳过
        horizons = list(horizons)
        for idx, horizon in enumerate(horizons):
            horizon_preds = []
            horizon_targets = []

            for weather_idx, weather_name in enumerate(["sunny", "cloudy", "overcast"]):
                mask = weather == weather_idx if weather is not None else slice(None)
                if weather is not None and not np.any(mask):
                    continue

                model = model_dict[weather_name].to(self.device)
                model.eval()

                feature_slice = torch.from_numpy(features[mask]).float().to(self.device)
                target_slice = torch.from_numpy(targets[mask]).float().to(self.device)

                if target_slice.dim() > 1:
                    if target_slice.shape[1] <= idx:
                        logger.warning(
                            "multi-horizon 跳过目标列，weather=%s horizon=%d 目标维度不足(%d)",
                            weather_name,
                            horizon,
                            target_slice.shape[1],
                        )
                        continue
                    target_slice = target_slice[:, idx]

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        output, _ = model(feature_slice)

                if output.dim() > 1:
                    if output.shape[1] <= idx:
                        logger.warning(
                            "multi-horizon 跳过预测列，weather=%s horizon=%d 预测维度不足(%d)",
                            weather_name,
                            horizon,
                            output.shape[1],
                        )
                        continue
                    pred_slice = output[:, idx]
                else:
                    pred_slice = output
                horizon_preds.append(pred_slice)
                horizon_targets.append(target_slice)
            if horizon_preds:
                all_predictions = torch.cat(horizon_preds, dim=0)
                all_targets = torch.cat(horizon_targets, dim=0)
                results[horizon] = self.calculate_all_metrics(
                    all_predictions, all_targets, prated=prated, power_scale=power_scale
                )
                logger.info(f"multi-horizon step {horizon}: {results[horizon]}")

        return results

    @torch.no_grad()
    def evaluate_by_weather(
        self,
        model_dict: Dict[str, torch.nn.Module],
        sequence_sets: Dict[str, np.ndarray],
        prated: Optional[float] = None,
        power_scale: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, MetricsResult]]:
        """分天气类型评估"""
        results_list = []
        metrics_map: Dict[str, MetricsResult] = {}
        features = sequence_sets["features"]
        targets = sequence_sets["targets"]
        weather = sequence_sets.get("weather")

        for weather_idx, weather_name in enumerate(["sunny", "cloudy", "overcast"]):
            mask = weather == weather_idx if weather is not None else slice(None)
            if weather is not None and not np.any(mask):
                logger.warning(f"缺少 {weather_name} 的测试数据")
                continue

            model = model_dict[weather_name].to(self.device)
            model.eval()

            feature_slice = torch.from_numpy(features[mask]).float().to(self.device)
            target_slice = torch.from_numpy(targets[mask]).float().to(self.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    predictions, _ = model(feature_slice)

            metrics = self.calculate_all_metrics(
                predictions, target_slice, prated=prated, power_scale=power_scale
            )
            metrics_map[weather_name] = metrics

            for metric_name, metric_value in metrics.to_dict().items():
                if metric_name != "CI_95%":
                    results_list.append({"weather_type": weather_name, "metric": metric_name, "value": metric_value})

        results_df = pd.DataFrame(results_list)
        pivot_df = results_df.pivot_table(index="weather_type", columns="metric", values="value", aggfunc="first")
        return pivot_df, metrics_map

    @torch.no_grad()
    def calculate_confidence_interval(
        self, errors: torch.Tensor, confidence: float = 0.95, n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """计算置信区间（Bootstrap）"""
        errors = self._ensure_tensor(errors)
        n_samples = errors.shape[0]

        bootstrap_means = []

        for _ in range(n_bootstrap):
            indices = torch.randint(0, n_samples, (n_samples,), device=self.device)
            bootstrap_sample = errors[indices]
            bootstrap_means.append(torch.mean(torch.abs(bootstrap_sample)).item())

        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return (float(lower), float(upper))

    def statistical_significance_test(
        self,
        model1_errors: Union[torch.Tensor, np.ndarray],
        model2_errors: Union[torch.Tensor, np.ndarray],
        test_type: str = "paired_t",
    ) -> Dict:
        """统计显著性检验"""
        if isinstance(model1_errors, torch.Tensor):
            model1_errors = model1_errors.cpu().numpy()
        if isinstance(model2_errors, torch.Tensor):
            model2_errors = model2_errors.cpu().numpy()

        if test_type == "paired_t":
            statistic, p_value = stats.ttest_rel(np.abs(model1_errors), np.abs(model2_errors))
            test_name = "配对t检验"
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(np.abs(model1_errors), np.abs(model2_errors))
            test_name = "Wilcoxon检验"
        else:
            raise ValueError(f"不支持的检验类型: {test_type}")

        alpha = 0.05
        is_significant = p_value < alpha

        diff = np.abs(model1_errors) - np.abs(model2_errors)
        cohens_d = np.mean(diff) / np.std(diff)

        return {
            "test_type": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_significant": bool(is_significant),
            "cohens_d": float(cohens_d),
            "conclusion": "显著差异" if is_significant else "无显著差异",
            "model1_mean_error": float(np.mean(np.abs(model1_errors))),
            "model2_mean_error": float(np.mean(np.abs(model2_errors))),
        }

    def compare_weather_significance(
        self, per_weather_errors: Dict[str, np.ndarray], baseline: str = "sunny", test_type: str = "paired_t"
    ) -> Dict[str, Dict]:
        """对比不同天气模型误差显著性"""
        significance_summary: Dict[str, Dict] = {}
        baseline_errors = per_weather_errors.get(baseline)
        if baseline_errors is None:
            return significance_summary

        for weather, errors in per_weather_errors.items():
            if weather == baseline or errors is None or len(errors) == 0:
                continue
            try:
                result = self.statistical_significance_test(baseline_errors, errors, test_type=test_type)
                significance_summary[f"{baseline}_vs_{weather}"] = result
            except ValueError:
                continue
        return significance_summary

    def get_summary_statistics(self) -> pd.DataFrame:
        """获取历史评估结果的汇总统计"""
        if not self.results_history:
            return pd.DataFrame()

        data = []
        for i, result in enumerate(self.results_history):
            row = result.to_dict()
            row["evaluation_id"] = i
            data.append(row)

        df = pd.DataFrame(data)
        summary = df.describe()
        return summary


def export_metrics_bundle(
    overall: MetricsResult, per_weather: Dict[str, MetricsResult], multi_horizon: Dict[int, MetricsResult]
) -> Dict[str, Dict]:
    """整理评估指标，输出 JSON 友好格式"""
    bundle = {
        "overall": overall.to_dict() if isinstance(overall, MetricsResult) else overall,
        "per_weather": {
            weather: metrics.to_dict() if isinstance(metrics, MetricsResult) else metrics
            for weather, metrics in per_weather.items()
        },
        "multi_horizon": {
            int(h): metrics.to_dict() if isinstance(metrics, MetricsResult) else metrics
            for h, metrics in multi_horizon.items()
        },
    }
    return bundle


def export_weather_distribution(weather_array: np.ndarray) -> Dict[str, int]:
    """根据天气标签统计分布"""
    distribution: Dict[str, int] = {}
    for idx, label in enumerate(["sunny", "cloudy", "overcast"]):
        count = int(np.sum(weather_array == idx))
        if count > 0:
            distribution[label] = count
    return distribution


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 1000
    true_values = torch.randn(n_samples, 1) * 100 + 500
    predictions = true_values + torch.randn_like(true_values) * 20

    evaluator = PerformanceMetrics(device="cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("性能评估测试")
    print("=" * 50)

    results = evaluator.calculate_all_metrics(predictions, true_values)
    print(f"\n评估结果: {results}")
    print("\n详细指标:")
    for key, value in results.to_dict().items():
        if key == "CI_95%":
            print(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}]")
        else:
            print(f"  {key}: {value:.4f}")

    predictions2 = true_values + torch.randn_like(true_values) * 25
    errors1 = (predictions - true_values).numpy()
    errors2 = (predictions2 - true_values).numpy()

    sig_test = evaluator.statistical_significance_test(errors1, errors2)
    print("\n统计显著性检验")
    for key, value in sig_test.items():
        print(f"  {key}: {value}")

    print("\n测试完成!")
