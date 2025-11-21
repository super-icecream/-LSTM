"""
WalkForwardTrainer
==================

Orchestrates walk-forward training, evaluation, and online learning.
"""



from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.cuda.amp as amp

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
                "Walk-Forward reuse labels: %s",
                {k: v.source for k, v in self.reuse_label_bundles.items()},
            )
            self.logger.info("Label reuse enabled; threshold fine-tuning skipped.")

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
        self.logger.info("weather dist (%s): %s, night=%d", split_name, ", ".join(parts), night_count)

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
                "Cluster labels[%s] day coverage %.3f < threshold %.3f, fallback to combined labels",
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
                "Cluster labels[%s] day coverage %.3f below warn threshold %.3f",
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
            self.logger.info("Start training Fold %d/%d", fold_id, len(folds))
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
                        "Fold %d %s day sequence ratio: %.2f%% (samples=%s)",
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
                self.logger.info("Fold %d initialized with previous fold weights", fold_id)

            training_cfg = dict(self.training_cfg)
            training_cfg["checkpoint_dir"] = str(checkpoint_dir)

            trainer = GPUOptimizedTrainer(multi_model.models, training_cfg, device=str(self.device))
            epochs = training_cfg.get("epochs", 100)
            trainer.train_all_models(train_loaders, val_loaders, epochs=epochs)

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

        aggregate = self._aggregate_results(fold_results)
        summary = {"folds": fold_results, "aggregate": aggregate}
        with open(self.base_result_dir / "summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2, default=_json_fallback)
        self.logger.info("Walk-Forward run complete; summary saved to %s", self.base_result_dir / "summary.json")
        return summary

    # ------------------------------------------------------------------ helpers
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
            logger.info("Loaded best weights for %s (%s)", weather_name, best_path.name)

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
            prated=self.config.get("evaluation", {}).get("prated"),
        )

        weather_distribution = export_weather_distribution(test_sequence.get("weather", np.array([])))
        evaluation_tool = PerformanceMetrics(
            device=str(device), prated=self.config.get("evaluation", {}).get("prated")
        )
        multi_horizon_metrics = evaluation_tool.evaluate_multi_horizon(
            multi_model.models,
            test_sequence,
            horizons=self.horizons,
        )
        freq_min = int(self.config.get("data", {}).get("frequency_minutes", 1))
        for h, metrics in multi_horizon_metrics.items():
            self.logger.info("multi-horizon %d-step (%d min): %s", h, h * freq_min, metrics)
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
        self.logger.info("Fold %d test evaluation finished", fold_id)
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

