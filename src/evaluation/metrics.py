"""
璇勪及鎸囨爣璁＄畻妯″潡
瀹炵幇DLFE-LSTM-WSI绯荤粺鐨勬€ц兘璇勪及鎸囨爣璁＄畻
鏀寔GPU鍔犻€熷拰澶氭椂闂村昂搴﹁瘎浼?"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

# 璁剧疆鏃ュ織
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """璇勪及鎸囨爣缁撴灉鏁版嵁绫?""

    rmse: float
    mae: float
    nrmse: float
    confidence_interval: Tuple[float, float]
    std_error: float = 0.0
    max_error: float = 0.0
    min_error: float = 0.0

    def to_dict(self) -> Dict:
        """杞崲涓哄瓧鍏告牸寮?""
        return {
            "RMSE": self.rmse,
            "MAE": self.mae,
            "NRMSE": self.nrmse,
            "CI_95%": self.confidence_interval,
            "STD": self.std_error,
            "MAX_ERROR": self.max_error,
            "MIN_ERROR": self.min_error,
        }

    def __str__(self) -> str:
        """鏍煎紡鍖栬緭鍑?""
        return f"RMSE: {self.rmse:.4f} | MAE: {self.mae:.4f} | NRMSE: {self.nrmse:.4f}"


class PerformanceMetrics:
    """
    GPU鍔犻€熺殑鎬ц兘璇勪及鎸囨爣璁＄畻绫?
    鏀寔鐨勬寚鏍囷細
    - RMSE: 鍧囨柟鏍硅宸?    - MAE: 骞冲潎缁濆璇樊
    - NRMSE: 褰掍竴鍖栧潎鏂规牴璇樊
    """

    def __init__(self, device: str = "cuda", epsilon: float = 1e-8):
        """
        鍒濆鍖栬瘎浼板櫒

        Args:
            device: 璁＄畻璁惧 ('cuda' 鎴?'cpu')
            epsilon: 闃叉闄ら浂鐨勫皬鏁?        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.results_history: List[MetricsResult] = []

        if self.device.type == "cuda":
            logger.info(f"浣跨敤GPU鍔犻€? {torch.cuda.get_device_name()}")
        else:
            logger.warning("CUDA涓嶅彲鐢紝浣跨敤CPU璁＄畻")

    def _ensure_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """纭繚鏁版嵁涓篏PU寮犻噺"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if not data.is_cuda and self.device.type == "cuda":
            data = data.to(self.device, non_blocking=True)
        return data

    @torch.no_grad()
    def calculate_rmse(self, predictions: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> float:
        """璁＄畻RMSE"""
        predictions = self._ensure_tensor(predictions)
        targets = self._ensure_tensor(targets)
        mse = torch.mean((predictions - targets) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()

    @torch.no_grad()
    def calculate_mae(self, predictions: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> float:
        """璁＄畻MAE"""
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
        璁＄畻NRMSE

        Args:
            predictions: 棰勬祴鍊?            targets: 鐪熷疄鍊?            normalization: 'range' | 'mean' | 'std'
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
            raise ValueError(f"鏈煡鐨勫綊涓€鍖栨柟寮?{normalization}")

        denominator = torch.clamp(denominator, min=self.epsilon)
        nrmse = rmse / denominator
        return nrmse.item()

    @torch.no_grad()
    def calculate_all_metrics(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        calculate_ci: bool = True,
    ) -> MetricsResult:
        """
        璁＄畻鎵€鏈夎瘎浼版寚鏍?
        Args:
            predictions: 棰勬祴鍊?            targets: 鐪熷疄鍊?            calculate_ci: 鏄惁璁＄畻缃俊鍖洪棿
        """
        predictions = self._ensure_tensor(predictions)
        targets = self._ensure_tensor(targets)

        rmse = self.calculate_rmse(predictions, targets)
        mae = self.calculate_mae(predictions, targets)
        nrmse = self.calculate_nrmse(predictions, targets)

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
        )
        self.results_history.append(result)
        return result

    @torch.no_grad()
    def evaluate_multi_horizon(
        self,
        model_dict: Dict[str, torch.nn.Module],
        sequence_sets: Dict[str, np.ndarray],
        horizons: List[int] = [1, 2, 4],
    ) -> Dict[int, MetricsResult]:
        """
        澶氭椂闂村昂搴﹁瘎浼?
        Args:
            model_dict: 涓変釜澶╂皵瀛愭ā鍨媨'sunny','cloudy','overcast'}
            sequence_sets: 鍖呭惈features/targets/weather鐨勬祴璇曟暟鎹?            horizons: 棰勬祴鏃跺煙鍒楄〃
        """
        features = sequence_sets["features"]
        targets = sequence_sets["targets"]
        weather = sequence_sets.get("weather")

        results: Dict[int, MetricsResult] = {}

        for horizon in horizons:
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
                    if horizon <= target_slice.shape[1]:
                        target_slice = target_slice[:, horizon - 1]
                    else:
                        continue

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        output, _ = model(feature_slice)

                if output.dim() > 1 and output.shape[1] < horizon:
                    continue
                pred_slice = output[:, horizon - 1] if output.dim() > 1 else output
                horizon_preds.append(pred_slice)
                horizon_targets.append(target_slice)
            if horizon_preds:
                all_predictions = torch.cat(horizon_preds, dim=0)
                all_targets = torch.cat(horizon_targets, dim=0)
                results[horizon] = self.calculate_all_metrics(all_predictions, all_targets)
                logger.info(f"multi-horizon step {horizon}: {results[horizon]}")

        return results

    @torch.no_grad()
    def evaluate_by_weather(
        self, model_dict: Dict[str, torch.nn.Module], sequence_sets: Dict[str, np.ndarray]
    ) -> Tuple[pd.DataFrame, Dict[str, MetricsResult]]:
        """
        鍒嗗ぉ姘旂被鍨嬭瘎浼?        """
        results_list = []
        metrics_map: Dict[str, MetricsResult] = {}
        features = sequence_sets["features"]
        targets = sequence_sets["targets"]
        weather = sequence_sets.get("weather")

        for weather_idx, weather_name in enumerate(["sunny", "cloudy", "overcast"]):
            mask = weather == weather_idx if weather is not None else slice(None)
            if weather is not None and not np.any(mask):
                logger.warning(f"缂哄皯 {weather_name} 鐨勬祴璇曟暟鎹?)
                continue

            model = model_dict[weather_name].to(self.device)
            model.eval()

            feature_slice = torch.from_numpy(features[mask]).float().to(self.device)
            target_slice = torch.from_numpy(targets[mask]).float().to(self.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    predictions, _ = model(feature_slice)

            metrics = self.calculate_all_metrics(predictions, target_slice)
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
        """
        璁＄畻缃俊鍖洪棿锛圔ootstrap锛?        """
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
        """
        缁熻鏄捐憲鎬ф楠?        """
        if isinstance(model1_errors, torch.Tensor):
            model1_errors = model1_errors.cpu().numpy()
        if isinstance(model2_errors, torch.Tensor):
            model2_errors = model2_errors.cpu().numpy()

        if test_type == "paired_t":
            statistic, p_value = stats.ttest_rel(np.abs(model1_errors), np.abs(model2_errors))
            test_name = "閰嶅t妫€楠?
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(np.abs(model1_errors), np.abs(model2_errors))
            test_name = "Wilcoxon妫€楠?
        else:
            raise ValueError(f"涓嶆敮鎸佺殑妫€楠岀被鍨?{test_type}")

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
            "conclusion": "鏄捐憲宸紓" if is_significant else "鏃犳樉钁楀樊寮?,
            "model1_mean_error": float(np.mean(np.abs(model1_errors))),
            "model2_mean_error": float(np.mean(np.abs(model2_errors))),
        }

    def compare_weather_significance(
        self, per_weather_errors: Dict[str, np.ndarray], baseline: str = "sunny", test_type: str = "paired_t"
    ) -> Dict[str, Dict]:
        """瀵规瘮涓嶅悓澶╂皵妯″瀷璇樊鏄捐憲鎬?""
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
        """
        鑾峰彇鍘嗗彶璇勪及缁撴灉鐨勬眹鎬荤粺璁?        """
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
    """鏁寸悊璇勪及鎸囨爣锛岃緭鍑?JSON 鍙嬪ソ鏍煎紡"""
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
    """鏍规嵁澶╂皵鏍囩缁熻鍒嗗竷"""
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
    print("鎬ц兘璇勪及娴嬭瘯")
    print("=" * 50)

    results = evaluator.calculate_all_metrics(predictions, true_values)
    print(f"\n璇勪及缁撴灉: {results}")
    print("\n璇︾粏鎸囨爣:")
    for key, value in results.to_dict().items():
        if key == "CI_95%":
            print(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}]")
        else:
            print(f"  {key}: {value:.4f}")

    predictions2 = true_values + torch.randn_like(true_values) * 25
    errors1 = (predictions - true_values).numpy()
    errors2 = (predictions2 - true_values).numpy()

    sig_test = evaluator.statistical_significance_test(errors1, errors2)
    print("\n缁熻鏄捐憲鎬ф楠?)
    for key, value in sig_test.items():
        print(f"  {key}: {value}")

    print("\n娴嬭瘯瀹屾垚!")
