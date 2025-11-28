"""
WalkForwardTrainer
==================

Orchestrates walk-forward training, evaluation, and online learning.
"""



from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from ..data_processing.preprocessor import Preprocessor
from ..data_processing.vmd_decomposer import VMDDecomposer
from ..feature_engineering.weather_classifier import WeatherClassifier
from ..feature_engineering.dpsr import DPSR
from ..feature_engineering.dlfe import DLFE
from ..models.multi_weather_model import MultiWeatherModel
from ..evaluation import (
    PerformanceMetrics,
    export_weather_distribution,
    export_metrics_bundle,
)
from .trainer import GPUOptimizedTrainer
from ..utils.cluster_labels import (
    ClusterLabelBundle,
    load_cluster_label_bundle,
    assign_cluster_labels,
)


logger = logging.getLogger(__name__)


def _json_fallback(obj: Any):
    """Safe JSON fallback for custom metric objects."""
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _align_length(arrays: List[np.ndarray]) -> List[np.ndarray]:
    min_len = min(arr.shape[0] for arr in arrays)
    return [arr[:min_len] for arr in arrays]


def _sanitize_array(array: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float32)


class WalkForwardTrainer:
    """Scheduler for walk-forward training / evaluation / online learning."""

    def __init__(
        self,
        *,
        config: Dict,
        base_artifact_dir: Path,
        base_checkpoint_dir: Path,
        base_result_dir: Path,
        logger: logging.Logger,
        weather_map: Dict[int, str],
        build_sequence_sets: Callable[
            [Dict[str, Dict[str, np.ndarray]], int, List[int], Any], Dict[str, Dict[str, np.ndarray]]
        ],
        build_weather_dataloaders: Callable[..., Dict[str, Any]],
        evaluate_fn: Callable[..., Any],
        build_model_builder: Callable[[Dict, int, int], Any],
        online_learning_override: Optional[bool] = None,
    ):
        self.config = config
        self.logger = logger
        self.weather_map = weather_map
        self.build_sequence_sets = build_sequence_sets
        self.build_weather_dataloaders = build_weather_dataloaders
        self.evaluate_fn = evaluate_fn
        self.build_model_builder = build_model_builder

        self.base_artifact_dir = base_artifact_dir
        self.base_checkpoint_dir = base_checkpoint_dir
        self.base_result_dir = base_result_dir

        self.seq_length = config.get("data", {}).get("sequence_length", 24)
        self.horizons = list(config.get("evaluation", {}).get("horizons", [1, 2, 4]))
        training_cfg = dict(config.get("training", {}))
        if training_cfg.get("learning_rate") is None:
            training_cfg["learning_rate"] = 0.001
        self.training_cfg = training_cfg

        self.batch_size = training_cfg.get("batch_size", 64)
        self.num_workers = training_cfg.get("num_workers", 0)

        # 统一装机容量到 kW
        prated_cfg = self.config.get("evaluation", {}).get("prated")
        self.prated_kw = self._normalize_prated_kw(prated_cfg)
        self.power_scale: Optional[float] = None

        project_device = config.get("project", {}).get("device", "cuda")
        self.device = torch.device(
            "cuda" if project_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )

        wf_cfg = config.get("walk_forward", {})
        online_cfg = wf_cfg.get("online_learning", {})
        if online_learning_override is not None:
            online_cfg = dict(online_cfg)
            online_cfg["enable"] = bool(online_learning_override)
        self.online_cfg = online_cfg
        self.weight_cfg = wf_cfg.get("weight_inheritance", {"enable": True, "strategy": "full"})

        fe_cfg = config.get("feature_engineering", {})
        reuse_cfg = fe_cfg.get("reuse_cluster_labels", {})
        self.reuse_labels_enabled = bool(reuse_cfg.get("enabled", False))
        self.reuse_label_bundles: Dict[str, ClusterLabelBundle] = {}
        self.reuse_label_bundle_combined: Optional[ClusterLabelBundle] = None
        self.reuse_label_fallback = str(reuse_cfg.get("fallback", "classify")).lower()
        self.reuse_label_min_coverage = float(reuse_cfg.get("min_coverage", 0.95))
        if self.reuse_labels_enabled:
            split_cfgs = reuse_cfg.get("splits", {})
            for split_name, cfg in split_cfgs.items():
                self.reuse_label_bundles[split_name] = load_cluster_label_bundle(cfg)
            if self.reuse_label_bundles:
                combined_series = pd.concat([b.series for b in self.reuse_label_bundles.values()]).sort_index()
                combined_series = combined_series[~combined_series.index.duplicated(keep="first")]
                self.reuse_label_bundle_combined = ClusterLabelBundle(
                    series=combined_series,
                    meta={"source": "combined"},
                    source="combined",
                )
            if self.reuse_label_fallback not in {"classify"}:
                raise ValueError("reuse_cluster_labels.fallback only supports 'classify'")
            self.logger.info(
                "Walk-Forward 标签复用已启用，可用标签来源: %s",
                {k: v.source for k, v in self.reuse_label_bundles.items()},
            )
            self.logger.info("标签复用已启用，将跳过阈值微调。")

        # 全局训练历史（跨所有fold累积，使用 NRMSE 统一指标）
        self.global_train_history: Dict[str, Dict[str, List]] = {
            weather: {
                'train_nrmse': [],
                'val_nrmse': [],
                'grad_last': [],
                'grad_avg': [],
                'grad_max': [],
                'lr': [],
            }
            for weather in ['sunny', 'cloudy', 'overcast']
        }
        self.fold_boundaries: List[int] = []  # 记录每个fold结束时的全局epoch编号
        self.fold_best_epochs: Dict[str, List[Dict]] = {
            weather: [] for weather in ['sunny', 'cloudy', 'overcast']
        }  # 记录每个fold内的最佳验证 NRMSE epoch

    def _log_weather_distribution(self, split_name: str, weather_bundle: Dict[str, np.ndarray]) -> None:
        labels = np.asarray(weather_bundle.get("labels"), dtype=np.int64)
        dist = {self.weather_map[idx]: int(np.sum(labels == idx)) for idx in self.weather_map}
        mask_src = weather_bundle.get("day_mask")
        if mask_src is None:
            day_mask_local = np.ones_like(labels, dtype=bool)
        else:
            day_mask_local = np.asarray(mask_src, dtype=bool)
        night_count = int((~day_mask_local).sum()) if day_mask_local.size else 0
        day_labels = labels[day_mask_local]
        day_total = max(int(day_labels.size), 1)
        dist_percent = {name: (dist[name] / day_total * 100.0) if day_total else 0.0 for name in dist}
        parts = [f"{name}={dist[name]} ({dist_percent[name]:.1f}%)" for name in dist]
        self.logger.info("天气分布(%s): %s, 夜间样本数=%d", split_name, ", ".join(parts), night_count)

    def _apply_cluster_labels_if_needed(
        self,
        df: pd.DataFrame,
        weather_bundle: Dict[str, np.ndarray],
        split_name: str,
    ) -> Dict[str, np.ndarray]:
        reuse_bundle = None
        if self.reuse_labels_enabled:
            canonical = split_name.split("-")[-1].lower()
            reuse_bundle = self.reuse_label_bundles.get(canonical)
            if reuse_bundle is None:
                reuse_bundle = self.reuse_label_bundle_combined
        if not self.reuse_labels_enabled or reuse_bundle is None:
            self._log_weather_distribution(split_name, weather_bundle)
            return weather_bundle
        mask_src = weather_bundle.get("day_mask")
        day_mask_local = (
            np.asarray(mask_src, dtype=bool) if mask_src is not None else np.ones(len(df), dtype=bool)
        )
        final_labels, stats = assign_cluster_labels(
            df.index,
            np.asarray(weather_bundle.get("labels"), dtype=np.int64),
            day_mask_local,
            reuse_bundle,
            self.reuse_label_fallback,
            split_name,
            self.logger,
        )
        weather_bundle["labels"] = final_labels
        weather_bundle["cluster_label_source"] = reuse_bundle.source
        weather_bundle["cluster_label_meta"] = reuse_bundle.meta
        weather_bundle["cluster_label_stats"] = stats
        cov_day = stats.get("day_coverage")
        # If coverage is low, fall back to combined labels to handle foldX prefixes
        if (
            self.reuse_label_bundle_combined is not None
            and reuse_bundle is not self.reuse_label_bundle_combined
            and np.isfinite(cov_day)
            and cov_day < float(self.reuse_label_min_coverage)
        ):
            self.logger.warning(
                "聚类标签[%s] 的白天覆盖率 %.3f 低于阈值 %.3f，将回退到合并标签",
                split_name,
                cov_day,
                self.reuse_label_min_coverage,
            )
            final_labels, stats = assign_cluster_labels(
                df.index,
                np.asarray(weather_bundle.get("labels"), dtype=np.int64),
                day_mask_local,
                self.reuse_label_bundle_combined,
                self.reuse_label_fallback,
                f"{split_name}-combined",
                self.logger,
            )
            weather_bundle["labels"] = final_labels
            weather_bundle["cluster_label_source"] = self.reuse_label_bundle_combined.source
            weather_bundle["cluster_label_meta"] = self.reuse_label_bundle_combined.meta
            weather_bundle["cluster_label_stats"] = stats
            cov_day = stats.get("day_coverage")
        if (
            self.reuse_label_min_coverage
            and np.isfinite(cov_day)
            and cov_day < float(self.reuse_label_min_coverage)
        ):
            self.logger.warning(
                "聚类标签[%s] 的白天覆盖率 %.3f 低于告警阈值 %.3f",
                split_name,
                cov_day,
                self.reuse_label_min_coverage,
            )
        self._log_weather_distribution(split_name, weather_bundle)
        return weather_bundle

    def train_all_folds(self, folds: List[Dict]) -> Dict[str, Any]:
        self.base_artifact_dir.mkdir(parents=True, exist_ok=True)
        self.base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.base_result_dir.mkdir(parents=True, exist_ok=True)

        prev_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        fold_results: List[Dict[str, Any]] = []

        for fold in folds:
            fold_id = fold["id"]
            self.logger.info("=" * 80)
            self.logger.info("开始训练第 %d/%d 折", fold_id, len(folds))
            self.logger.info("=" * 80)

            artifact_dir = self.base_artifact_dir / f"fold_{fold_id:02d}"
            checkpoint_dir = self.base_checkpoint_dir / f"fold_{fold_id:02d}"
            result_dir = self.base_result_dir / f"fold_{fold_id:02d}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)

            feature_sets = self._prepare_fold_features(fold, artifact_dir)
            sequence_sets = self.build_sequence_sets(feature_sets, self.seq_length, self.horizons, self.logger)
            daytime_stats: Dict[str, Dict[str, Any]] = {}
            for split_name in ("train", "val", "test"):
                meta = sequence_sets[split_name].get("meta", {}) if sequence_sets.get(split_name) else {}
                daytime_stats[split_name] = meta
                day_ratio = meta.get("day_ratio")
                if day_ratio is not None:
                    self.logger.info(
                        "折 %d %s 白天序列占比: %.2f%% (样本数=%s)",
                        fold_id,
                        split_name,
                        day_ratio * 100,
                        meta.get("kept_sequences", "N/A"),
                    )

            train_loaders = self.build_weather_dataloaders(
                sequence_sets["train"], self.batch_size, self.num_workers, shuffle=True
            )
            val_loaders = self.build_weather_dataloaders(
                sequence_sets["val"], self.batch_size, self.num_workers, shuffle=False
            )
            test_loaders = self.build_weather_dataloaders(
                sequence_sets["test"], self.batch_size, self.num_workers, shuffle=False
            )

            input_dim = sequence_sets["train"]["features"].shape[2]
            model_builder = self.build_model_builder(self.config, input_dim, self.seq_length)
            multi_model = MultiWeatherModel(
                model_builder,
                use_model_parallel=self.config.get("model", {}).get("use_model_parallel", False),
            )

            if prev_state and self.weight_cfg.get("enable", True):
                self._load_previous_state(multi_model, prev_state)
                self.logger.info("折 %d 已使用上一折的权重进行初始化", fold_id)

            training_cfg = dict(self.training_cfg)
            training_cfg["checkpoint_dir"] = str(checkpoint_dir)

            trainer = GPUOptimizedTrainer(
                multi_model.models,
                training_cfg,
                device=str(self.device),
                log_dir=str(result_dir),
                power_scale=self.power_scale,
                prated=self.prated_kw,
            )
            epochs = training_cfg.get("epochs", 100)
            trainer.train_all_models(train_loaders, val_loaders, epochs=epochs)

            # 追加当前fold的训练历史到全局历史
            self._append_fold_history(trainer.train_history, fold_id, epochs)

            self._load_best_checkpoints(multi_model, checkpoint_dir)
            test_metrics = self._evaluate_fold(
                fold_id,
                multi_model,
                sequence_sets["test"],
                result_dir,
                trainer.device,
            )

            online_metrics = None
            if self.online_cfg.get("enable", True):
                online_metrics = self._perform_online_learning(
                    fold_id,
                    multi_model,
                    sequence_sets["test"],
                    test_loaders,
                    trainer.device,
                    checkpoint_dir,
                )

            prev_state = self._capture_state(multi_model)
            self._save_final_state(multi_model, checkpoint_dir)

            fold_record = {
                "fold_id": fold_id,
                "time_ranges": fold["time_ranges"],
                "sizes": fold["size"],
                "daytime_stats": daytime_stats,
                "test_metrics": test_metrics["overall"],
                "per_weather_metrics": test_metrics["per_weather"],
                "online_metrics": online_metrics,
                "weather_distribution": test_metrics["weather_distribution"],
                "checkpoint_dir": str(checkpoint_dir),
                "result_dir": str(result_dir),
            }
            fold_results.append(fold_record)

            with open(result_dir / "metrics.json", "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "test_metrics": test_metrics,
                        "online_metrics": online_metrics,
                        "time_ranges": fold["time_ranges"],
                    },
                    fp,
                    ensure_ascii=False,
                    indent=2,
                    default=_json_fallback,
                )

        # 保存并绘制全局训练历史曲线
        self.save_overall_loss_history()
        self.plot_overall_loss_curves()

        aggregate = self._aggregate_results(fold_results)
        summary = {"folds": fold_results, "aggregate": aggregate}
        with open(self.base_result_dir / "summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2, default=_json_fallback)
        self.logger.info("Walk-Forward 运行完成，汇总已保存到 %s", self.base_result_dir / "summary.json")
        return summary

    # ------------------------------------------------------------------ helpers
    def _normalize_prated_kw(self, prated_value: Optional[float]) -> Optional[float]:
        """将装机容量统一转换为 kW；若值较小（<=1000）则视为 MW 并乘以1000。"""
        if prated_value is None:
            return None
        try:
            val = float(prated_value)
        except (TypeError, ValueError):
            return None
        return val * 1000.0 if val <= 1000 else val

    def _compute_power_scale(self, preprocessor: Preprocessor) -> Optional[float]:
        """根据预处理器参数提取功率缩放因子（用于将归一化误差还原到物理功率尺度）。"""
        method = getattr(preprocessor, "method", None)
        params = getattr(preprocessor, "scaler_params", {}) or {}
        key = "power"
        if method == "minmax":
            mins = params.get("min", {})
            maxs = params.get("max", {})
            if key in mins and key in maxs:
                return float(maxs[key] - mins[key])
        elif method == "standard":
            stds = params.get("std", {})
            if key in stds:
                return float(stds[key])
        elif method == "robust":
            iqrs = params.get("iqr", {})
            if key in iqrs:
                return float(iqrs[key])
        elif method == "maxabs":
            max_abs = params.get("max_abs", {})
            if key in max_abs:
                return float(max_abs[key])
        return None

    def _prepare_fold_features(self, fold: Dict, artifact_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
        train_df: Any = fold["train"]
        val_df: Any = fold["val"]
        test_df: Any = fold["test"]

        norm_cfg = self.config.get("preprocessing", {}).get("normalization", {})
        preprocessor = Preprocessor(
            method=norm_cfg.get("method", "minmax"),
            feature_range=tuple(norm_cfg.get("feature_range", [0, 1])),
        )
        preprocessor.fit(train_df)
        train_proc = preprocessor.transform(train_df)
        val_proc = preprocessor.transform(val_df)
        test_proc = preprocessor.transform(test_df)
        preprocessor.save_params(artifact_dir / "preprocessor.json")
        # 记录功率缩放因子供评估还原物理量纲
        self.power_scale = self._compute_power_scale(preprocessor)

        fe_cfg = self.config.get("feature_engineering", {})
        daytime_cfg = fe_cfg.get("daytime", {})
        loc_cfg = self.config.get("data", {}).get("location", {}) or {}
        weather_classifier = WeatherClassifier(
            location_lat=float(loc_cfg.get("lat", 38.5)),
            location_lon=float(loc_cfg.get("lon", 105.0)),
            elevation=float(loc_cfg.get("elevation", 1500.0)),
            time_zone_hours=float(loc_cfg.get("time_zone_hours", 8.0)),
            ci_thresholds=fe_cfg.get("ci_thresholds", [0.2, 0.6]),
            wsi_thresholds=fe_cfg.get("wsi_thresholds", [0.3, 0.7]),
            fusion_weights=fe_cfg.get("fusion_weights", {"ci": 0.7, "wsi": 0.3}),
            daytime_ge_min=daytime_cfg.get("ge_min_wm2", 20.0),
            daytime_mode=str(daytime_cfg.get("mode", "ghi")).lower(),
            daytime_ghi_min=float(daytime_cfg.get("ghi_min_wm2", 5.0)),
            night_handling=daytime_cfg.get("night_handling", "exclude"),
        )
        train_weather = weather_classifier.classify(train_df)
        val_weather = weather_classifier.classify(val_df)
        test_weather = weather_classifier.classify(test_df)
        train_weather = self._apply_cluster_labels_if_needed(train_df, train_weather, f"fold{fold['id']}-train")
        val_weather = self._apply_cluster_labels_if_needed(val_df, val_weather, f"fold{fold['id']}-val")
        test_weather = self._apply_cluster_labels_if_needed(test_df, test_weather, f"fold{fold['id']}-test")
        def _extract_day_mask(bundle: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
            mask = bundle.get("day_mask")
            return np.asarray(mask, dtype=bool) if mask is not None else None

        train_day_mask = _extract_day_mask(train_weather)
        val_day_mask = _extract_day_mask(val_weather)
        test_day_mask = _extract_day_mask(test_weather)

        with open(artifact_dir / "weather_classifier.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "ci_thresholds": weather_classifier.ci_thresholds,
                    "wsi_thresholds": weather_classifier.wsi_thresholds,
                    "fusion_weights": weather_classifier.fusion_weights,
                    "daytime_ge_min": weather_classifier.daytime_ge_min,
                    "night_handling": weather_classifier.night_handling,
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )

        vmd_cfg = self.config.get("preprocessing", {}).get("vmd", {})
        vmd = VMDDecomposer(
            n_modes=vmd_cfg.get("n_modes", 5),
            alpha=vmd_cfg.get("alpha", 2000),
            tau=vmd_cfg.get("tau", 2.0),
            DC=vmd_cfg.get("DC", 0),
            init=vmd_cfg.get("init", 1),
            tol=vmd_cfg.get("tolerance", 1e-6),
            max_iter=vmd_cfg.get("max_iter", 500),
            verbose=False,  # disable per-segment progress spam; handled by process_dataset
        )
        apply_mask_vmd = daytime_cfg.get("apply_mask_in_vmd", False)
        train_vmd = vmd.process_dataset(
            train_proc,
            day_mask=train_day_mask if apply_mask_vmd else None,
        )
        val_vmd = vmd.process_dataset(
            val_proc,
            day_mask=val_day_mask if apply_mask_vmd else None,
        )
        test_vmd = vmd.process_dataset(
            test_proc,
            day_mask=test_day_mask if apply_mask_vmd else None,
        )
        vmd.save_params(artifact_dir / "vmd.pkl")

        dpsr_cfg = fe_cfg.get("dpsr", {})
        dpsr = DPSR(
            embedding_dim=dpsr_cfg.get("embedding_dim", 30),
            neighborhood_size=dpsr_cfg.get("neighborhood_size", 50),
            regularization=dpsr_cfg.get("regularization", 0.01),
            time_delay=dpsr_cfg.get("time_delay", 1),
            max_iter=dpsr_cfg.get("max_iter", 100),
            learning_rate=dpsr_cfg.get("learning_rate", 0.01),
        )
        apply_mask_dpsr = daytime_cfg.get("apply_mask_in_dpsr", False)
        train_dpsr, _ = dpsr.fit_transform(
            train_vmd,
            day_mask=train_day_mask if apply_mask_dpsr else None,
        )
        val_dpsr = dpsr.transform(
            val_vmd,
            day_mask=val_day_mask if apply_mask_dpsr else None,
        )
        test_dpsr = dpsr.transform(
            test_vmd,
            day_mask=test_day_mask if apply_mask_dpsr else None,
        )
        dpsr.save_weights(artifact_dir / "dpsr_weights.pkl")

        dlfe_cfg = fe_cfg.get("dlfe", {})
        dlfe = DLFE(
            target_dim=dlfe_cfg.get("target_dim", 30),
            sigma=dlfe_cfg.get("sigma", 1.0),
            alpha=dlfe_cfg.get("alpha", 2 ** -10),
            beta=dlfe_cfg.get("beta", 0.1),
            max_iter=dlfe_cfg.get("max_iter", 100),
            tol=dlfe_cfg.get("tol", 1e-6),
        )
        apply_mask_dlfe = daytime_cfg.get("apply_mask_in_dlfe", False)
        train_dlfe = dlfe.fit_transform(
            train_dpsr,
            dpsr_weights=None,
            day_mask=train_day_mask if apply_mask_dlfe else None,
        )
        val_dlfe = dlfe.transform(
            val_dpsr,
            day_mask=val_day_mask if apply_mask_dlfe else None,
        )
        test_dlfe = dlfe.transform(
            test_dpsr,
            day_mask=test_day_mask if apply_mask_dlfe else None,
        )
        dlfe.save_mapping(artifact_dir / "dlfe_mapping.pkl")

        def build_feature_set(features: np.ndarray, processed_df: Any, weather_bundle: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            feature_array = _sanitize_array(features)
            target_array = _sanitize_array(processed_df["power"].values.astype(np.float32))
            weather_array = np.asarray(weather_bundle["labels"], dtype=np.int64)
            day_mask_array = np.asarray(
                weather_bundle.get("day_mask", np.ones_like(weather_array, dtype=bool)),
                dtype=bool,
            )
            feature_array, target_array, weather_array, day_mask_array = _align_length(
                [feature_array, target_array, weather_array, day_mask_array]
            )
            return {
                "features": feature_array,
                "targets": _sanitize_array(target_array).reshape(-1, 1),
                "weather": weather_array,
                "day_mask": day_mask_array.astype(bool),
            }

        return {
            "train": build_feature_set(train_dlfe, train_proc, train_weather),
            "val": build_feature_set(val_dlfe, val_proc, val_weather),
            "test": build_feature_set(test_dlfe, test_proc, test_weather),
        }

    def _load_best_checkpoints(self, multi_model: MultiWeatherModel, checkpoint_dir: Path) -> None:
        for _, weather_name in self.weather_map.items():
            best_path = checkpoint_dir / f"best_{weather_name}_model.pth"
            if not best_path.exists():
                continue
            state = torch.load(best_path, map_location=self.device)
            multi_model.models[weather_name].load_state_dict(state["model_state_dict"])
            logger.info("已加载天气类别 %s 的最佳权重 (%s)", weather_name, best_path.name)

    def _evaluate_fold(
        self,
        fold_id: int,
        multi_model: MultiWeatherModel,
        test_sequence: Dict[str, np.ndarray],
        result_dir: Path,
        device: torch.device,
    ) -> Dict[str, Any]:
        batch_size = self.batch_size
        num_workers = self.num_workers
        overall_metrics, per_weather_metrics, predictions, targets, per_weather_errors = self.evaluate_fn(
            multi_model,
            test_sequence,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            logger=self.logger,
            prated=self.prated_kw,
            power_scale=self.power_scale,
        )

        weather_distribution = export_weather_distribution(test_sequence.get("weather", np.array([])))
        evaluation_tool = PerformanceMetrics(
            device=str(device), prated=self.prated_kw, power_scale=self.power_scale
        )
        multi_horizon_metrics = evaluation_tool.evaluate_multi_horizon(
            multi_model.models,
            test_sequence,
            horizons=self.horizons,
            prated=self.prated_kw,
            power_scale=self.power_scale,
        )
        freq_min = int(self.config.get("data", {}).get("frequency_minutes", 1))
        for h, metrics in multi_horizon_metrics.items():
            self.logger.info("多步预测 %d 步(%d 分钟): %s", h, h * freq_min, metrics)
        significance_results = evaluation_tool.compare_weather_significance(per_weather_errors)

        payload = {
            "overall": overall_metrics,
            "per_weather": per_weather_metrics,
            "multi_horizon": multi_horizon_metrics,
            "significance": significance_results,
            "weather_distribution": weather_distribution,
        }

        metrics_bundle = export_metrics_bundle(overall_metrics, per_weather_metrics, multi_horizon_metrics)
        metrics_bundle["significance"] = significance_results
        metrics_bundle["weather_distribution"] = weather_distribution

        with open(result_dir / "test_metrics.json", "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2, default=_json_fallback)

        np.save(result_dir / "predictions.npy", predictions)
        np.save(result_dir / "targets.npy", targets)
        self.logger.info("折 %d 测试评估完成", fold_id)
        return payload

    def _capture_state(self, multi_model: MultiWeatherModel) -> Dict[str, Dict[str, torch.Tensor]]:
        state: Dict[str, Dict[str, torch.Tensor]] = {}
        for weather_name, model in multi_model.models.items():
            state[weather_name] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        return state

    def _load_previous_state(self, multi_model: MultiWeatherModel, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        for weather_name, model in multi_model.models.items():
            if weather_name in state:
                model.load_state_dict(state[weather_name])

    def _save_final_state(self, multi_model: MultiWeatherModel, checkpoint_dir: Path) -> None:
        for weather_name, model in multi_model.models.items():
            payload = {"model_state_dict": model.state_dict()}
            torch.save(payload, checkpoint_dir / f"final_{weather_name}_model.pth")

    def _perform_online_learning(
        self,
        fold_id: int,
        multi_model: MultiWeatherModel,
        test_sequence: Dict[str, np.ndarray],
        test_loaders: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        lr_multiplier = self.online_cfg.get("learning_rate_multiplier", 0.1)
        max_updates = int(self.online_cfg.get("max_updates_per_fold", 0))
        if max_updates <= 0:
            return None

        base_lr = self.training_cfg.get("learning_rate", 0.001)
        online_lr = base_lr * lr_multiplier
        weight_decay = self.training_cfg.get("weight_decay", 0.0)

        criterion = nn.MSELoss()
        optimizers: Dict[str, torch.optim.Optimizer] = {}
        losses: Dict[str, List[float]] = {}
        use_amp = device.type == "cuda"
        scaler = amp.GradScaler(enabled=use_amp)

        def _check_tensor_finite(name: str, tensor: torch.Tensor, weather: str, upd: int) -> None:
            """确保张量无 NaN/Inf，若有则立刻报错并给出统计信息。"""
            if not torch.is_floating_point(tensor):
                return
            total = tensor.numel()
            if total == 0:
                return
            finite_mask = torch.isfinite(tensor)
            finite_cnt = int(finite_mask.sum().item())
            if finite_cnt != total:
                self.logger.error(
                    "Online learning 非有限数值: %s weather=%s update=%d shape=%s dtype=%s finite=%d/%d",
                    name,
                    weather,
                    upd,
                    tuple(tensor.shape),
                    tensor.dtype,
                    finite_cnt,
                    total,
                )
                bad_indices = (finite_mask == 0).nonzero(as_tuple=False)[:5].cpu().tolist()
                self.logger.error("示例异常位置（最多5个）: %s", bad_indices)
                raise ValueError(f"online_learning {name} 含 NaN/Inf")

        for weather_name, model in multi_model.models.items():
            model.to(device)
            model.train()
            optimizers[weather_name] = torch.optim.AdamW(model.parameters(), lr=online_lr, weight_decay=weight_decay)
            losses[weather_name] = []

        updates = 0
        loader_iters = {w: iter(loader) for w, loader in test_loaders.items()}

        while updates < max_updates and loader_iters:
            for weather_name, model in multi_model.models.items():
                if weather_name not in loader_iters:
                    continue
                loader = loader_iters[weather_name]
                try:
                    batch = next(loader)
                except StopIteration:
                    # 当前天气的数据已耗尽，移除迭代器，避免死循环
                    del loader_iters[weather_name]
                    continue

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    features, targets, _ = batch
                else:
                    features, targets = batch

                param_dtype = next(model.parameters()).dtype
                features = features.to(device, dtype=param_dtype)
                targets = targets.to(device, dtype=param_dtype)

                _check_tensor_finite("features", features, weather_name, updates)
                _check_tensor_finite("targets", targets, weather_name, updates)

                optimizer = optimizers[weather_name]
                optimizer.zero_grad()
                try:
                    with amp.autocast(enabled=use_amp):
                        preds, _ = model(features)
                        loss = criterion(preds, targets)
                except Exception:
                    self.logger.exception(
                        "Online learning 前向失败: weather=%s update=%d shape=%s dtype=%s",
                        weather_name,
                        updates,
                        tuple(features.shape),
                        features.dtype,
                    )
                    raise

                if use_amp:
                    try:
                        scaler.scale(loss).backward()
                    except Exception:
                        self.logger.exception(
                            "Online learning 反向失败(AMP): weather=%s update=%d loss_dtype=%s",
                            weather_name,
                            updates,
                            loss.dtype,
                        )
                        raise
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except Exception:
                        self.logger.exception(
                            "Online learning optimizer step 失败(AMP): weather=%s update=%d", weather_name, updates
                        )
                        raise
                else:
                    try:
                        loss.backward()
                    except Exception:
                        self.logger.exception(
                            "Online learning 反向失败: weather=%s update=%d loss_dtype=%s",
                            weather_name,
                            updates,
                            loss.dtype,
                        )
                        raise
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    try:
                        optimizer.step()
                    except Exception:
                        self.logger.exception(
                            "Online learning optimizer step 失败: weather=%s update=%d", weather_name, updates
                        )
                        raise

                losses[weather_name].append(float(loss.detach().cpu()))
                updates += 1
                if updates >= max_updates:
                    break

            if not loader_iters:
                break

        torch.save(
            {w: model.state_dict() for w, model in multi_model.models.items()},
            checkpoint_dir / "online_learning_state.pth",
        )
        self.logger.info('Fold %d online learning finished, updates=%d', fold_id, updates)
        return {
            "updates": updates,
            "average_loss": {w: float(np.mean(loss_list)) if loss_list else None for w, loss_list in losses.items()},
        }

    def _aggregate_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not fold_results:
            return {}

        def _as_dict(obj: Any) -> Dict[str, Any]:
            if hasattr(obj, "to_dict"):
                try:
                    return obj.to_dict()
                except Exception:
                    return {}
            return obj if isinstance(obj, dict) else {}

        first_metrics = _as_dict(fold_results[0]["test_metrics"])
        overall_keys = first_metrics.keys()
        aggregate_overall: Dict[str, float] = {}
        for key in overall_keys:
            values = []
            for fold in fold_results:
                metric_block = _as_dict(fold["test_metrics"])
                value = metric_block.get(key) if isinstance(metric_block, dict) else None
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                aggregate_overall[key] = float(np.mean(values))

        per_weather_agg: Dict[str, Dict[str, float]] = {}
        for fold in fold_results:
            per_weather = fold.get("per_weather_metrics", {})
            for weather, metric_map in per_weather.items():
                weather_bucket = per_weather_agg.setdefault(weather, {})
                metric_dict = _as_dict(metric_map)
                for metric_name, metric_value in metric_dict.items():
                    weather_bucket.setdefault(metric_name, []).append(metric_value)
        per_weather_summary = {
            weather: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            for weather, metrics in per_weather_agg.items()
        }

        return {
            "overall": aggregate_overall,
            "per_weather": per_weather_summary,
            "fold_count": len(fold_results),
        }

    # ------------------------------------------------------------------ helpers
    def _normalize_prated_kw(self, prated_value: Optional[float]) -> Optional[float]:
        """将装机容量统一转换为 kW；若值较小（<=1000）则视为 MW 并乘以1000。"""
        if prated_value is None:
            return None
        try:
            val = float(prated_value)
        except (TypeError, ValueError):
            return None
        return val * 1000.0 if val <= 1000 else val

    def _compute_power_scale(self, preprocessor: Preprocessor) -> Optional[float]:
        """根据预处理器参数提取功率缩放因子（用于将归一化误差还原到物理功率尺度）。"""
        method = getattr(preprocessor, "method", None)
        params = getattr(preprocessor, "scaler_params", {}) or {}
        key = "power"
        if method == "minmax":
            mins = params.get("min", {})
            maxs = params.get("max", {})
            if key in mins and key in maxs:
                return float(maxs[key] - mins[key])
        elif method == "standard":
            stds = params.get("std", {})
            if key in stds:
                return float(stds[key])
        elif method == "robust":
            iqrs = params.get("iqr", {})
            if key in iqrs:
                return float(iqrs[key])
        elif method == "maxabs":
            max_abs = params.get("max_abs", {})
            if key in max_abs:
                return float(max_abs[key])
        return None


    def _append_fold_history(
        self,
        fold_history: Dict[str, Dict[str, List]],
        fold_id: int,
        epochs: int,
    ) -> None:
        """将单个fold的训练历史追加到全局历史"""
        current_global_epoch = len(self.global_train_history['sunny']['train_nrmse'])

        for weather in ['sunny', 'cloudy', 'overcast']:
            if weather not in fold_history:
                continue
            hist = fold_history[weather]

            # 追加各指标（注意：trainer 现在使用 train_nrmse/val_nrmse）
            train_nrmse = hist.get('train_nrmse', [])
            val_nrmse = hist.get('val_nrmse', [])
            grad_last = hist.get('grad_last', [])
            grad_avg = hist.get('grad_avg', [])
            grad_max = hist.get('grad_max', [])
            lr = hist.get('lr', [])

            self.global_train_history[weather]['train_nrmse'].extend(train_nrmse)
            self.global_train_history[weather]['val_nrmse'].extend(val_nrmse)
            self.global_train_history[weather]['grad_last'].extend(grad_last)
            self.global_train_history[weather]['grad_avg'].extend(grad_avg)
            self.global_train_history[weather]['grad_max'].extend(grad_max)
            self.global_train_history[weather]['lr'].extend(lr)

            # 记录当前fold内的最佳验证 NRMSE epoch（全局编号）
            if val_nrmse:
                local_best_idx = int(np.argmin(val_nrmse))
                global_best_epoch = current_global_epoch + local_best_idx + 1
                self.fold_best_epochs[weather].append({
                    'fold_id': fold_id,
                    'global_epoch': global_best_epoch,
                    'local_epoch': local_best_idx + 1,
                    'val_nrmse': val_nrmse[local_best_idx],
                })

        # 记录fold边界（全局epoch编号）
        new_total = len(self.global_train_history['sunny']['train_nrmse'])
        self.fold_boundaries.append(new_total)
        self.logger.debug("Fold %d 历史已追加，全局epoch数: %d", fold_id, new_total)

    def save_overall_loss_history(self, save_dir: Optional[Path] = None) -> None:
        """保存全局训练历史到JSON文件"""
        save_dir = save_dir or self.base_result_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        history_data = {
            'train_history': self.global_train_history,
            'fold_boundaries': self.fold_boundaries,
            'fold_best_epochs': self.fold_best_epochs,
        }

        save_path = save_dir / 'overall_nrmse_history.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        print(f"[*] 全局NRMSE历史已保存至: {save_path}")
        sys.stdout.flush()

    def plot_overall_loss_curves(self, save_dir: Optional[Path] = None) -> None:
        """绘制全局 NRMSE 曲线图（跨所有fold）"""
        save_dir = save_dir or self.base_result_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        weather_names = {'sunny': '晴天', 'cloudy': '多云', 'overcast': '阴天'}
        weathers = ['sunny', 'cloudy', 'overcast']
        n_models = len(weathers)

        # EMA配置（默认关闭）
        grad_ema_alpha = float(self.training_cfg.get('grad_ema_alpha', 0.0))
        log_gradients = bool(self.training_cfg.get('log_gradients', True))

        # 创建图表：三个子模型并排一行
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), dpi=100)
        if n_models == 1:
            axes = [axes]

        for ax, weather in zip(axes, weathers):
            history = self.global_train_history.get(weather, {})
            train_nrmse = history.get('train_nrmse', [])
            val_nrmse = history.get('val_nrmse', [])
            grad_last = history.get('grad_last', [])

            if not train_nrmse:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
                ax.set_title(f'{weather_names.get(weather, weather)} 模型', fontsize=13, fontweight='bold')
                continue

            total_epochs = len(train_nrmse)
            epochs_range = range(1, total_epochs + 1)

            # 绘制训练 NRMSE 曲线
            ax.plot(epochs_range, train_nrmse, 'b-', label='训练NRMSE', linewidth=1.5, alpha=0.8)

            # 绘制验证 NRMSE 曲线
            if val_nrmse:
                ax.plot(epochs_range, val_nrmse, 'r--', label='验证NRMSE', linewidth=1.5, alpha=0.8)

            # 绘制fold边界（垂直虚线）和fold标签
            for i, boundary in enumerate(self.fold_boundaries):
                if boundary < total_epochs:
                    ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
                # 在fold中间位置添加标签
                start = self.fold_boundaries[i - 1] if i > 0 else 0
                mid = (start + boundary) / 2
                y_pos = ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else max(train_nrmse) * 0.95
                ax.text(mid, y_pos, f'Fold {i + 1}', ha='center', va='top', fontsize=9, color='dimgray', alpha=0.8)

            # 标注每个fold的最佳验证 NRMSE 点
            best_epochs_info = self.fold_best_epochs.get(weather, [])
            for info in best_epochs_info:
                epoch = info['global_epoch']
                nrmse_val = info['val_nrmse']
                ax.scatter(epoch, nrmse_val, color='red', s=80, zorder=5, marker='*')
                ax.annotate(
                    f"F{info['fold_id']}:{nrmse_val:.4f}",
                    xy=(epoch, nrmse_val),
                    xytext=(5, 8), textcoords='offset points',
                    fontsize=8, color='red',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6)
                )

            # 叠加梯度曲线（右轴）
            if log_gradients and grad_last:
                # 可选EMA平滑
                if 0.0 < grad_ema_alpha < 1.0:
                    ema_series = []
                    s = None
                    for v in grad_last:
                        s = v if s is None else (grad_ema_alpha * v + (1.0 - grad_ema_alpha) * s)
                        ema_series.append(s)
                    to_plot = ema_series
                else:
                    to_plot = grad_last

                ax2 = ax.twinx()
                ax2.plot(epochs_range, to_plot, color='green', linestyle='-', label='梯度范数', linewidth=1.2, alpha=0.7)
                ax2.set_ylabel('梯度范数', fontsize=10, color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.set_ylim(bottom=0)

                # 合并图例
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
            else:
                ax.legend(loc='upper right', fontsize=9)

            ax.set_xlabel('Epoch (全局)', fontsize=11)
            ax.set_ylabel('NRMSE (%)', fontsize=11)
            ax.set_title(f'{weather_names.get(weather, weather)} 模型', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=1, right=total_epochs)

        plt.suptitle('Walk-Forward 全局 NRMSE 曲线', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存图表
        save_path = save_dir / 'overall_nrmse_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[*] 全局NRMSE曲线图已保存至: {save_path}")
        sys.stdout.flush()
