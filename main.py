#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DLFE-LSTM-WSI 项目主入口

该脚本串联配置加载、数据预处理、特征工程、模型训练以及评估流程，
按照《DLFE-LSTM-WSI-4 - 最终方案总结》的架构实现真实数据驱动的流水线。

当前实现支持以下模式：
- ``prepare``：只运行数据/特征流水线并缓存结果；
- ``train``：执行完整的训练 + 评估流程；
- ``test``：加载指定训练产物，对测试集重新评估。

后续可以在此基础上扩展 ``predict``、``serve``、``optimize`` 等模式。
"""

from __future__ import annotations

# ========================================
# 全局日志配置（必须在导入其他模块之前）
# ========================================
import logging
import sys

root_logger = logging.getLogger()
root_logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)
# ========================================

import argparse
import gc
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import (
    initialize_project,
    DataLoader as PVDataLoader,
    DataSplitter,
    Preprocessor,
    VMDDecomposer,
    WalkForwardSplitter,
    WeatherClassifier,
    DPSR,
    DLFE,
    ModelBuilder,
    MultiWeatherModel,
    GPUOptimizedTrainer,
    WalkForwardTrainer,
    PerformanceMetrics,
    PerformanceVisualizer,
    export_metrics_bundle,
    export_weather_distribution,
    generate_markdown_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WEATHER_MAP = {0: "sunny", 1: "cloudy", 2: "overcast"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DLFE-LSTM-WSI 主程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_shared_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            type=str,
            default=str(PROJECT_ROOT / "config" / "config.yaml"),
            help="配置文件路径",
        )
        p.add_argument(
            "--force-rebuild",
            action="store_true",
            help="强制重新执行数据/特征流水线，忽略缓存",
        )

    prepare_parser = subparsers.add_parser("prepare", help="仅构建并缓存特征")
    add_shared_arguments(prepare_parser)
    prepare_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="实验运行名称（默认使用时间戳）",
    )

    train_parser = subparsers.add_parser("train", help="训练并评估模型")
    add_shared_arguments(train_parser)
    train_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="实验运行名称（默认使用时间戳）",
    )

    test_parser = subparsers.add_parser("test", help="加载已训练模型并在测试集评估")
    add_shared_arguments(test_parser)
    test_parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="需要评估的运行名称（用于定位检查点和结果目录）",
    )

    walk_forward_parser = subparsers.add_parser("walk-forward", help="运行 Walk-Forward 验证流程")
    add_shared_arguments(walk_forward_parser)
    walk_forward_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="实验运行名称（默认使用时间戳）",
    )
    walk_forward_parser.add_argument(
        "--disable-online-learning",
        action="store_true",
        help="禁用测试阶段的在线学习微调",
    )

    return parser.parse_args()


@dataclass
class PipelinePaths:
    raw: Path
    processed: Path
    features: Path
    splits: Path
    artifacts: Path
    runs: Path
    run_dir: Optional[Path] = None
    checkpoints: Optional[Path] = None
    results: Optional[Path] = None


def resolve_paths(config: Dict, run_name: Optional[str]) -> PipelinePaths:
    data_cfg = config.get("data", {})
    project_cfg = config.get("project", {})

    root_dir = Path(data_cfg.get("root_dir", PROJECT_ROOT / "data")).resolve()
    raw_dir = Path(data_cfg.get("raw_dir", root_dir / "raw")).resolve()
    processed_dir = Path(data_cfg.get("processed_dir", root_dir / "processed")).resolve()
    features_dir = Path(data_cfg.get("features_dir", root_dir / "features")).resolve()
    splits_dir = Path(data_cfg.get("splits_dir", root_dir / "splits")).resolve()

    artifacts_root = Path(project_cfg.get("artifacts_dir", PROJECT_ROOT / "experiments" / "artifacts"))
    runs_root = Path(project_cfg.get("runs_dir", PROJECT_ROOT / "experiments" / "runs"))

    run_dir = runs_root / run_name if run_name else None
    checkpoints_dir = run_dir / "checkpoints" if run_dir else None
    results_dir = run_dir / "results" if run_dir else None
    artifacts_dir = (run_dir / "artifacts") if run_dir else artifacts_root

    # 创建基础目录
    for directory in [processed_dir, features_dir, splits_dir, artifacts_dir, runs_root]:
        directory.mkdir(parents=True, exist_ok=True)
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        if checkpoints_dir:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if results_dir:
            results_dir.mkdir(parents=True, exist_ok=True)

    return PipelinePaths(
        raw=raw_dir,
        processed=processed_dir,
        features=features_dir,
        splits=splits_dir,
        artifacts=artifacts_dir,
        runs=runs_root,
        run_dir=run_dir,
        checkpoints=checkpoints_dir,
        results=results_dir,
    )


def load_cached_feature_split(directory: Path, split: str) -> Optional[Dict[str, np.ndarray]]:
    file_path = directory / f"{split}_features.npz"
    if not file_path.exists():
        return None

    data = np.load(file_path, allow_pickle=True)
    required_keys = {"features", "targets", "weather"}
    if not required_keys.issubset(data.files):
        return None

    if "day_mask" not in data.files:
        return None
    day_mask = data["day_mask"]

    return {
        "features": data["features"],
        "targets": data["targets"],
        "weather": data["weather"],
        "day_mask": day_mask.astype(bool),
    }


def save_feature_split(directory: Path, split: str, features: np.ndarray, targets: np.ndarray, weather: np.ndarray,
                       day_mask: Optional[np.ndarray] = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{split}_features.npz"
    save_kwargs = {
        "features": features,
        "targets": targets,
        "weather": weather,
    }
    if day_mask is not None:
        save_kwargs["day_mask"] = day_mask
    np.savez_compressed(file_path, **save_kwargs)


def align_length(arrays: Iterable[np.ndarray]) -> Tuple[np.ndarray, ...]:
    lengths = [arr.shape[0] for arr in arrays]
    min_len = min(lengths)
    return tuple(arr[:min_len] for arr in arrays)


def sanitize_feature_array(array: np.ndarray) -> np.ndarray:
    sanitized = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if sanitized.ndim == 1:
        sanitized = sanitized.reshape(-1, 1)
    return sanitized.astype(np.float32)


def load_or_prepare_merged_dataset(
    config: Dict,
    paths: PipelinePaths,
    logger,
    force_rebuild: bool,
):
    """
    加载或重新构建合并后的原始数据集，返回DataFrame及DataLoader实例。
    """

    loader_params_path = paths.artifacts / "dataloader_params.json"
    data_loader = PVDataLoader(data_path=str(paths.raw))
    loader_meta_path = paths.processed / "merged.metadata.json"

    if loader_params_path.exists() and loader_meta_path.exists() and not force_rebuild:
        logger.info("检测到 DataLoader 参数缓存，尝试加载原始合并数据。")
        try:
            data_loader.load_params(loader_params_path)
            merged = data_loader.load_processed_dataset(paths.processed / "merged.parquet")
            logger.info("已从缓存加载合并数据与 DataLoader 配置")
        except Exception as exc:
            logger.warning(f"加载缓存的合并数据失败，将重新执行数据加载: {exc}")
            merged = None
    else:
        merged = None

    if merged is None:
        merge_method = config.get("data", {}).get("merge_method", "concat")
        selected_station = config.get("data", {}).get("selected_station")
        station_data = data_loader.load_multi_station(
            merge_method=merge_method,
            selected_station=selected_station,
        )
        if not station_data:
            raise FileNotFoundError(f"原始数据目录 {paths.raw} 中未找到任何 CSV 数据文件")

        if len(station_data) == 1:
            merged = next(iter(station_data.values()))
            logger.info("检测到单站点数据，跳过合并阶段")
        else:
            if merge_method == "single":
                merged = next(iter(station_data.values()))
                logger.info("merge_method=single，默认使用第一个站点")
            else:
                merged = data_loader.merge_stations(station_data, method=merge_method)
                logger.info(f"多站点数据合并完成，采用方法 {merge_method}")

        merged.sort_index(inplace=True)
        merged = data_loader.handle_missing_values(merged)
        passed, quality_report = data_loader.validate_data_quality(merged)
        if not passed:
            logger.warning(f"数据质量检查存在问题: {quality_report.get('issues', [])}")

        paths.processed.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(paths.processed / "merged.parquet")
        data_loader.save_params(loader_params_path)
        with open(loader_meta_path, "w", encoding="utf-8") as meta_fp:
            json.dump({"station_files": sorted(str(fp) for fp in paths.raw.glob("*.csv"))}, meta_fp, ensure_ascii=False, indent=2)
    else:
        passed, quality_report = data_loader.validate_data_quality(merged)
        if not passed:
            logger.warning(f"缓存数据的质量检查存在问题: {quality_report.get('issues', [])}")

    merged.sort_index(inplace=True)
    merged = data_loader.handle_missing_values(merged)
    passed, quality_report = data_loader.validate_data_quality(merged)
    if not passed:
        logger.warning(f"数据质量检查存在问题: {quality_report.get('issues', [])}")

    return merged, data_loader


def run_feature_pipeline(config: Dict, paths: PipelinePaths, logger, force_rebuild: bool) -> Dict[str, Dict[str, np.ndarray]]:
    # logger.info("开始执行数据预处理与特征工程流水线…")  # 已移除冗余日志

    merged, _ = load_or_prepare_merged_dataset(config, paths, logger, force_rebuild)

    splitter = DataSplitter(
        train_ratio=config.get("data", {}).get("train_ratio", 0.7),
        val_ratio=config.get("data", {}).get("val_ratio", 0.2),
        test_ratio=config.get("data", {}).get("test_ratio", 0.1),
    )
    split_meta_path = paths.splits / "split_info.json"
    if not force_rebuild and (paths.splits / "train.parquet").exists() and split_meta_path.exists():
        logger.info("检测到数据划分缓存，直接读取训练/验证/测试集")
        train_df = pd.read_parquet(paths.splits / "train.parquet")
        val_df = pd.read_parquet(paths.splits / "val.parquet")
        test_df = pd.read_parquet(paths.splits / "test.parquet")
        try:
            splitter.load_split_info(split_meta_path)
        except Exception as exc:
            logger.warning(f"加载划分元信息失败，将重新计算: {exc}")
    else:
        train_df, val_df, test_df = splitter.split_temporal(merged)
        splitter.save_splits(train_df, val_df, test_df, paths.splits, formats=["parquet"]) 

    norm_cfg = config.get("preprocessing", {}).get("normalization", {})
    preprocessor = Preprocessor(
        method=norm_cfg.get("method", "minmax"),
        feature_range=tuple(norm_cfg.get("feature_range", [0, 1])),
    )
    preprocessor.fit(train_df)
    train_proc = preprocessor.transform(train_df)
    val_proc = preprocessor.transform(val_df)
    test_proc = preprocessor.transform(test_df)
    preprocessor.save_params(paths.artifacts / "preprocessor.json")

    fe_cfg = config.get("feature_engineering", {})
    daytime_cfg = fe_cfg.get("daytime", {})
    weather_classifier = WeatherClassifier(
        ci_thresholds=fe_cfg.get("ci_thresholds", [0.2, 0.6]),
        wsi_thresholds=fe_cfg.get("wsi_thresholds", [0.3, 0.7]),
        fusion_weights=fe_cfg.get("fusion_weights", {"ci": 0.7, "wsi": 0.3}),
        daytime_ge_min=daytime_cfg.get("ge_min_wm2", 20.0),
        night_handling=daytime_cfg.get("night_handling", "exclude"),
    )
    train_weather = weather_classifier.classify(train_df)
    val_weather = weather_classifier.classify(val_df)
    test_weather = weather_classifier.classify(test_df)

    def _extract_day_mask(bundle: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        mask = bundle.get("day_mask")
        return np.asarray(mask, dtype=bool) if mask is not None else None

    train_day_mask = _extract_day_mask(train_weather)
    val_day_mask = _extract_day_mask(val_weather)
    test_day_mask = _extract_day_mask(test_weather)

    with open(paths.artifacts / "weather_classifier.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ci_thresholds": weather_classifier.ci_thresholds,
                "wsi_thresholds": weather_classifier.wsi_thresholds,
                "fusion_weights": weather_classifier.fusion_weights,
                "daytime_ge_min": weather_classifier.daytime_ge_min,
                "night_handling": weather_classifier.night_handling,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    vmd_cfg = config.get("preprocessing", {}).get("vmd", {})
    vmd = VMDDecomposer(
        n_modes=vmd_cfg.get("n_modes", 5),
        alpha=vmd_cfg.get("alpha", 2000),
        tau=vmd_cfg.get("tau", 2.0),
        DC=vmd_cfg.get("DC", 0),
        init=vmd_cfg.get("init", 1),
        tol=vmd_cfg.get("tolerance", 1e-6),
        max_iter=vmd_cfg.get("max_iter", 500),
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
    vmd.save_params(paths.artifacts / "vmd.pkl")

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
    train_dpsr, dpsr_weights = dpsr.fit_transform(
        train_vmd,
        day_mask=train_day_mask if apply_mask_dpsr else None,
    )
    # 记录各数据集样本量供自适应处理参考
    dpsr.last_split_sizes = {
        "train": len(train_vmd),
        "val": len(val_vmd),
        "test": len(test_vmd),
    }
    val_dpsr = dpsr.transform(
        val_vmd,
        day_mask=val_day_mask if apply_mask_dpsr else None,
    )
    test_dpsr = dpsr.transform(
        test_vmd,
        day_mask=test_day_mask if apply_mask_dpsr else None,
    )
    dpsr.save_weights(paths.artifacts / "dpsr_weights.pkl")

    # Clear GPU cache after DPSR to avoid OOM during DLFE.
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info("✓ DPSR后清理GPU - 已分配: %.2f GB, 已保留: %.2f GB", allocated, reserved)

    dlfe_cfg = fe_cfg.get("dlfe", {})
    dlfe = DLFE(
        target_dim=dlfe_cfg.get("target_dim", 30),
        sigma=dlfe_cfg.get("sigma", 1.0),
        alpha=dlfe_cfg.get("alpha", 2 ** -10),
        beta=dlfe_cfg.get("beta", 0.1),
        max_iter=dlfe_cfg.get("max_iter", 100),
        tol=dlfe_cfg.get("tol", 1e-5),
        use_float32_eigh=dlfe_cfg.get("use_float32_eigh", False),
        use_sparse_matrix=dlfe_cfg.get("use_sparse_matrix", True),
    )
    apply_mask_dlfe = daytime_cfg.get("apply_mask_in_dlfe", False)
    train_dlfe = dlfe.fit_transform(
        train_dpsr,
        dpsr_weights,
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
    dlfe.save_mapping(paths.artifacts / "dlfe_mapping.pkl")

    def build_feature_set(features: np.ndarray, processed: pd.DataFrame, weather_bundle: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feature_array = sanitize_feature_array(features)
        target_array = sanitize_feature_array(processed["power"].values.astype(np.float32))
        weather_array = np.asarray(weather_bundle["labels"], dtype=np.int64)
        day_mask_array = np.asarray(weather_bundle.get("day_mask", np.ones_like(weather_array, dtype=bool)), dtype=bool)
        feature_array, target_array, weather_array, day_mask_array = align_length(
            [feature_array, target_array, weather_array, day_mask_array]
        )
        return {
            "features": feature_array,
            "targets": sanitize_feature_array(target_array).reshape(-1, 1),
            "weather": weather_array,
            "day_mask": day_mask_array.astype(bool),
        }

    feature_sets = {
        "train": build_feature_set(train_dlfe, train_proc, train_weather),
        "val": build_feature_set(val_dlfe, val_proc, val_weather),
        "test": build_feature_set(test_dlfe, test_proc, test_weather),
    }

    logger.info("特征工程流水线执行完成：train=%d, val=%d, test=%d", *[feature_sets[split]["features"].shape[0] for split in ("train", "val", "test")])
    return feature_sets


def prepare_feature_sets(config: Dict, paths: PipelinePaths, logger, force_rebuild: bool) -> Dict[str, Dict[str, np.ndarray]]:
    splits = ["train", "val", "test"]
    cached = {} if force_rebuild else {split: load_cached_feature_split(paths.features, split) for split in splits}
    if not force_rebuild and all(cached.get(split) for split in splits):
        # Highlight reused cache in orange and log the event.
        print("\033[38;5;214m⚠️  检测到缓存的特征文件，直接加载（跳过DPSR和DLFE特征工程）\033[0m")
        logger.info("检测到缓存的特征文件，直接加载")
        return cached  # type: ignore

    feature_sets = run_feature_pipeline(config, paths, logger, force_rebuild)
    for split in splits:
        save_feature_split(
            paths.features,
            split,
            feature_sets[split]["features"],
            feature_sets[split]["targets"],
            feature_sets[split]["weather"],
            feature_sets[split].get("day_mask"),
        )
    return feature_sets



def build_sequence_sets(
    feature_sets: Dict[str, Dict[str, np.ndarray]],
    sequence_length: int,
    logger,
) -> Dict[str, Dict[str, np.ndarray]]:
    base_loader = PVDataLoader()  # 复用 create_sequence_data
    sequence_sets: Dict[str, Dict[str, np.ndarray]] = {}

    for split, data in feature_sets.items():
        features = data["features"]
        targets = data["targets"].flatten()
        weather = data["weather"]
        day_mask_points = (
            np.asarray(data.get("day_mask"), dtype=bool)
            if data.get("day_mask") is not None
            else np.ones(len(features), dtype=bool)
        )

        if len(features) <= sequence_length:
            raise ValueError(f"{split} 数据长度不足以构建序列 (len={len(features)}, sequence_length={sequence_length})")

        if len(day_mask_points) != len(features):
            raise ValueError(
                f"{split} 的 day_mask 长度({len(day_mask_points)}) 与特征长度({len(features)}) 不一致"
            )

        seq_features, seq_targets, seq_weather = base_loader.create_sequence_data(
            features,
            targets,
            sequence_length,
            weather_array=weather,
        )

        total_sequences = seq_features.shape[0]
        seq_day_mask = day_mask_points[sequence_length - 1 :]
        keep_mask = seq_day_mask.astype(bool)

        filtered_features = seq_features[keep_mask]
        filtered_targets = seq_targets[keep_mask]
        filtered_weather = seq_weather[keep_mask]
        filtered_day_mask = seq_day_mask[keep_mask]

        if filtered_features.size == 0:
            logger.warning("%s 剔除夜间后无有效序列，请检查白天阈值设置", split)

        valid_mask = filtered_weather >= 0
        if not np.all(valid_mask):
            filtered_features = filtered_features[valid_mask]
            filtered_targets = filtered_targets[valid_mask]
            filtered_weather = filtered_weather[valid_mask]
            filtered_day_mask = filtered_day_mask[valid_mask]

        kept_sequences = int(filtered_features.shape[0])
        day_ratio = float(kept_sequences / total_sequences) if total_sequences else 0.0
        weather_counts = {name: int(np.sum(filtered_weather == idx)) for idx, name in WEATHER_MAP.items()}

        logger.info(
            "%s 序列数据(白天末端)构建完成: 样本=%d/%d (保留率=%.1f%%), 序列长度=%d, 特征维度=%d",
            split,
            kept_sequences,
            total_sequences,
            day_ratio * 100,
            filtered_features.shape[1] if kept_sequences else sequence_length,
            filtered_features.shape[2] if kept_sequences else features.shape[1],
        )
        logger.info("%s 天气样本分布(白天末端): %s", split, weather_counts)

        sequence_sets[split] = {
            "features": filtered_features.astype(np.float32),
            "targets": filtered_targets.astype(np.float32),
            "weather": filtered_weather.astype(np.int64),
            "day_mask": filtered_day_mask.astype(bool),
            "meta": {
                "kept_sequences": kept_sequences,
                "total_sequences": int(total_sequences),
                "day_ratio": day_ratio,
                "weather_counts": weather_counts,
            },
        }

    return sequence_sets

def build_weather_dataloaders(
    sequence_set: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> Dict[str, DataLoader]:
    logger = logging.getLogger(__name__)
    features = sequence_set["features"]
    targets = sequence_set["targets"]
    weather = sequence_set["weather"]
    day_mask = sequence_set.get("day_mask")

    if day_mask is not None and day_mask.size and not np.all(day_mask):
        valid_mask = day_mask.astype(bool)
        features = features[valid_mask]
        targets = targets[valid_mask]
        weather = weather[valid_mask]

    valid_weather_mask = weather >= 0
    if not np.all(valid_weather_mask):
        features = features[valid_weather_mask]
        targets = targets[valid_weather_mask]
        weather = weather[valid_weather_mask]

    total_samples = features.shape[0]
    weather_counts = {name: int(np.sum(weather == idx)) for idx, name in WEATHER_MAP.items()}
    seq_meta = sequence_set.get("meta", {})
    day_ratio = seq_meta.get("day_ratio")
    if day_ratio is not None:
        logger.info("序列集白天占比: %.2f%% (样本=%d)", day_ratio * 100, total_samples)
    logger.info("天气样本数: %s", weather_counts)

    min_required = max(batch_size * 10, 1000)
    scarce = {name: count for name, count in weather_counts.items() if count < min_required}
    if scarce:
        logger.warning(
            "以下天气类别样本数低于推荐阈值 %d: %s。建议运行 scripts/diagnose_weather.py 进行阈值校准。",
            min_required,
            scarce,
        )

    dataloaders: Dict[str, DataLoader] = {}
    for weather_idx, weather_name in WEATHER_MAP.items():
        mask = weather == weather_idx
        if not np.any(mask):
            continue
        dataset = TensorDataset(
            torch.from_numpy(features[mask]).float(),
            torch.from_numpy(targets[mask]).float(),
        )
        dataloaders[weather_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return dataloaders


def evaluate_multi_weather_model(
    multi_model: MultiWeatherModel,
    sequence_set: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    logger,
) -> Tuple[PerformanceMetrics, Dict[str, PerformanceMetrics], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    metrics_tool = PerformanceMetrics(device=str(device))
    features = sequence_set["features"]
    targets = sequence_set["targets"]
    weather = sequence_set.get("weather")

    predictions = np.zeros_like(targets)
    per_weather_results: Dict[str, PerformanceMetrics] = {}
    per_weather_errors: Dict[str, np.ndarray] = {}

    for weather_idx, weather_name in WEATHER_MAP.items():
        if weather is None:
            mask = slice(None)
        else:
            mask = weather == weather_idx
            if not np.any(mask):
                logger.debug("天气 %s 无样本，跳过评估", weather_name)
                continue

        dataset = TensorDataset(
            torch.from_numpy(features[mask]).float(),
            torch.from_numpy(targets[mask]).float(),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = multi_model.models.get(weather_name)
        if model is None:
            logger.warning("未找到 %s 模型，跳过该天气的评估", weather_name)
            continue

        model.eval()
        pred_collector, target_collector = [], []
        with torch.no_grad():
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(device)
                outputs, _ = model(batch_features)
                pred_collector.append(outputs.cpu().numpy())
                target_collector.append(batch_targets.numpy())

        preds = np.concatenate(pred_collector, axis=0)
        tars = np.concatenate(target_collector, axis=0)
        predictions[mask] = preds

        metrics = metrics_tool.calculate_all_metrics(
            torch.from_numpy(preds),
            torch.from_numpy(tars),
        )
        per_weather_results[weather_name] = metrics
        per_weather_errors[weather_name] = preds - tars

    overall_metrics = metrics_tool.calculate_all_metrics(
        torch.from_numpy(predictions),
        torch.from_numpy(targets),
    )

    logger.info("测试集总体指标: %s", overall_metrics)
    for weather_name, metric_obj in per_weather_results.items():
        logger.info("天气 %s 指标: %s", weather_name, metric_obj)

    return overall_metrics, per_weather_results, predictions, targets, per_weather_errors


def build_model_builder(config: Dict, input_dim: int, sequence_length: int) -> ModelBuilder:
    builder = ModelBuilder()
    builder.config.update(
        {
            "input_dim": input_dim,
            "hidden_dims": config.get("model", {}).get("lstm", {}).get("hidden_sizes", [100, 50]),
            "dropout_rates": config.get("model", {}).get("lstm", {}).get("dropout_rates", [0.3, 0.2]),
            "output_dim": config.get("model", {}).get("output_dim", 1),
            "sequence_length": sequence_length,
            "batch_size": config.get("training", {}).get("batch_size", 64),
        }
    )
    return builder


def run_walk_forward(args: argparse.Namespace, config: Dict, paths: PipelinePaths, logger) -> None:
    walk_cfg = config.get("walk_forward", {})
    if not walk_cfg.get("enable", False):
        logger.error("配置中未启用 walk_forward.enable，无法执行 Walk-Forward 模式")
        return

    run_name = args.run_name or datetime.now().strftime("walk_%Y%m%d_%H%M%S")
    if paths.run_dir is None or paths.run_dir.name != run_name:
        paths = resolve_paths(config, run_name)

    logger.info("启动 Walk-Forward 任务，运行名称：%s", run_name)
    merged, _ = load_or_prepare_merged_dataset(config, paths, logger, args.force_rebuild)

    splitter = WalkForwardSplitter(config)
    folds = splitter.create_folds(merged)
    if not WalkForwardSplitter.validate_folds(folds):
        raise RuntimeError("Walk-Forward 划分验证失败，请检查配置的时间窗口是否重叠")

    wf_artifact_root = paths.artifacts / "walk_forward"
    wf_artifact_root.mkdir(parents=True, exist_ok=True)
    splitter.save_fold_info(folds, wf_artifact_root / "fold_info.json")

    wf_checkpoint_root = paths.checkpoints / "walk_forward"
    wf_result_root = paths.results / "walk_forward"
    wf_checkpoint_root.mkdir(parents=True, exist_ok=True)
    wf_result_root.mkdir(parents=True, exist_ok=True)

    online_override = False if getattr(args, "disable_online_learning", False) else None

    trainer = WalkForwardTrainer(
        config=config,
        base_artifact_dir=wf_artifact_root,
        base_checkpoint_dir=wf_checkpoint_root,
        base_result_dir=wf_result_root,
        logger=logger,
        weather_map=WEATHER_MAP,
        build_sequence_sets=build_sequence_sets,
        build_weather_dataloaders=build_weather_dataloaders,
        evaluate_fn=evaluate_multi_weather_model,
        build_model_builder=build_model_builder,
        online_learning_override=online_override,
    )

    summary = trainer.train_all_folds(folds)
    aggregate = summary.get("aggregate", {})
    if aggregate:
        logger.info("Walk-Forward 聚合指标: %s", json.dumps(aggregate, ensure_ascii=False))
    logger.info("Walk-Forward 任务完成，结果目录：%s", wf_result_root)


def run_train(args: argparse.Namespace, config: Dict, paths: PipelinePaths, logger) -> None:
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if paths.run_dir is None:
        paths = resolve_paths(config, run_name)
    logger.info("启动训练任务，运行名称：%s", run_name)

    if not getattr(args, "force_rebuild", False):
        args.force_rebuild = choose_feature_rebuild(args, paths, logger)

    feature_sets = prepare_feature_sets(config, paths, logger, args.force_rebuild)
    seq_length = config.get("data", {}).get("sequence_length", 24)
    sequence_sets = build_sequence_sets(feature_sets, seq_length, logger)

    batch_size = config.get("training", {}).get("batch_size", 64)
    num_workers = config.get("training", {}).get("num_workers", 4)

    train_loaders = build_weather_dataloaders(sequence_sets["train"], batch_size, num_workers, shuffle=True)
    val_loaders = build_weather_dataloaders(sequence_sets["val"], batch_size, num_workers, shuffle=False)

    input_dim = sequence_sets["train"]["features"].shape[2]
    model_builder = build_model_builder(config, input_dim, seq_length)
    multi_model = MultiWeatherModel(
        model_builder,
        use_model_parallel=config.get("model", {}).get("use_model_parallel", False),
    )

    training_cfg = dict(config.get("training", {}))
    if paths.checkpoints is None:
        raise RuntimeError("未能解析 checkpoint 保存路径")
    training_cfg["checkpoint_dir"] = str(paths.checkpoints)

    device_str = config.get("project", {}).get("device", "cuda")
    trainer_device = "cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu"
    trainer = GPUOptimizedTrainer(multi_model.models, training_cfg, device=trainer_device)

    epochs = training_cfg.get("epochs", 100)
    trainer.train_all_models(train_loaders, val_loaders, epochs=epochs)

    # 加载最佳权重
    for _, weather_name in WEATHER_MAP.items():
        best_path = paths.checkpoints / f"best_{weather_name}_model.pth"
        if best_path.exists():
            state = torch.load(best_path, map_location=trainer.device)
            multi_model.models[weather_name].load_state_dict(state["model_state_dict"])
            logger.info("已加载 %s 的最佳权重 (%s)", weather_name, best_path.name)

    overall_metrics, per_weather_metrics, predictions, targets, per_weather_errors = evaluate_multi_weather_model(
        multi_model,
        sequence_sets["test"],
        batch_size=batch_size,
        num_workers=num_workers,
        device=trainer.device,
        logger=logger,
    )

    if paths.results is None:
        raise RuntimeError("未能解析结果输出目录")
    weather_counts = export_weather_distribution(sequence_sets["test"].get("weather", np.array([])))

    evaluation_tool = PerformanceMetrics(device=str(trainer.device))
    multi_horizon_metrics = evaluation_tool.evaluate_multi_horizon(
        multi_model.models,
        sequence_sets["test"],
        horizons=config.get("evaluation", {}).get("horizons", [1, 3, 6]),
    )

    significance_results = evaluation_tool.compare_weather_significance(per_weather_errors)

    metrics_payload = export_metrics_bundle(overall_metrics, per_weather_metrics, multi_horizon_metrics)
    metrics_payload["significance"] = significance_results
    metrics_payload["weather_distribution"] = weather_counts

    with open(paths.results / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    visualizer = PerformanceVisualizer()
    visualizer.save_all_figures(
        {
            "overall": overall_metrics,
            "per_weather": per_weather_metrics,
            "multi_horizon": multi_horizon_metrics,
            "weather_distribution": weather_counts,
        },
        predictions.flatten(),
        targets.flatten(),
        output_dir=str(paths.results),
    )

    generate_markdown_report(
        paths.results / "evaluation_summary.md",
        overall_metrics,
        per_weather_metrics,
        multi_horizon_metrics,
        significance_results,
        weather_counts,
    )

    logger.info("训练与评估完成，结果已保存至 %s", paths.results)


def run_test(args: argparse.Namespace, config: Dict, paths: PipelinePaths, logger) -> None:
    logger.info("进入测试模式，运行名称：%s", args.run_name)
    feature_sets = prepare_feature_sets(config, paths, logger, args.force_rebuild)
    seq_length = config.get("data", {}).get("sequence_length", 24)
    sequence_sets = build_sequence_sets(feature_sets, seq_length, logger)

    if paths.checkpoints is None or paths.results is None:
        raise RuntimeError("测试模式需要提供合法的运行目录和检查点")

    checkpoint_dir = paths.checkpoints
    available_checkpoints = list(checkpoint_dir.glob("best_*_model.pth"))
    if not available_checkpoints:
        raise FileNotFoundError(f"在 {checkpoint_dir} 下未找到最佳模型检查点")

    batch_size = config.get("training", {}).get("batch_size", 64)
    num_workers = config.get("training", {}).get("num_workers", 4)
    input_dim = sequence_sets["test"]["features"].shape[2]

    model_builder = build_model_builder(config, input_dim, seq_length)
    multi_model = MultiWeatherModel(
        model_builder,
        use_model_parallel=config.get("model", {}).get("use_model_parallel", False),
    )

    device_str = config.get("project", {}).get("device", "cuda")
    eval_device = torch.device("cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu")

    for _, weather_name in WEATHER_MAP.items():
        best_path = checkpoint_dir / f"best_{weather_name}_model.pth"
        if not best_path.exists():
            logger.warning("未找到 %s 的最佳模型检查点: %s", weather_name, best_path.name)
            continue
        state = torch.load(best_path, map_location=eval_device)
        multi_model.models[weather_name].load_state_dict(state["model_state_dict"])
        logger.info("已加载 %s 的最佳权重 (%s)", weather_name, best_path.name)

    overall_metrics, per_weather_metrics, predictions, targets, per_weather_errors = evaluate_multi_weather_model(
        multi_model,
        sequence_sets["test"],
        batch_size=batch_size,
        num_workers=num_workers,
        device=eval_device,
        logger=logger,
    )

    weather_counts = export_weather_distribution(sequence_sets["test"].get("weather", np.array([])))

    evaluation_tool = PerformanceMetrics(device=str(eval_device))
    multi_horizon_metrics = evaluation_tool.evaluate_multi_horizon(
        multi_model.models,
        sequence_sets["test"],
        horizons=config.get("evaluation", {}).get("horizons", [1, 3, 6]),
    )

    significance_results = evaluation_tool.compare_weather_significance(per_weather_errors)

    metrics_payload = export_metrics_bundle(overall_metrics, per_weather_metrics, multi_horizon_metrics)
    metrics_payload["significance"] = significance_results
    metrics_payload["weather_distribution"] = weather_counts

    with open(paths.results / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    visualizer = PerformanceVisualizer()
    visualizer.save_all_figures(
        {
            "overall": overall_metrics,
            "per_weather": per_weather_metrics,
            "multi_horizon": multi_horizon_metrics,
            "weather_distribution": weather_counts,
        },
        predictions.flatten(),
        targets.flatten(),
        output_dir=str(paths.results),
    )

    generate_markdown_report(
        paths.results / "evaluation_summary.md",
        overall_metrics,
        per_weather_metrics,
        multi_horizon_metrics,
        significance_results,
        weather_counts,
    )

    logger.info("测试评估完成，结果保存至 %s", paths.results)




def choose_train_mode(args) -> str:
    """
    交互式选择训练模式，返回 'train' 或 'walk-forward'。

    优先读取环境变量 DLFE_TRAIN_MODE；若终端非交互则默认返回 'train'。
    输入 3 允许用户取消执行。
    """
    env_mode = os.getenv("DLFE_TRAIN_MODE", "").strip().lower()
    if env_mode in {"train", "walk-forward"}:
        return env_mode

    try:
        if not sys.stdin.isatty():
            return "train"
    except Exception:
        return "train"

    print()
    print("请选择训练模式：")
    print("1) 标准训练")
    print("2) Walk-Forward 验证")
    print("3) 取消")
    sel = input("请输入数字并回车 [默认1]: ").strip()

    if sel == "2":
        return "walk-forward"
    if sel == "3":
        print("已取消。")
        raise SystemExit(0)
    return "train"


def _feature_cache_exists(paths: PipelinePaths) -> bool:
    feats = paths.features
    return all((feats / name).exists() for name in ("train_features.npz", "val_features.npz", "test_features.npz"))


def choose_feature_rebuild(args, paths: PipelinePaths, logger) -> bool:
    """
    根据环境变量 / 交互输入决定是否重新运行特征流水线。
    返回 True 表示从头重算（等同 --force-rebuild），False 表示沿用缓存。
    """
    if not _feature_cache_exists(paths):
        logger.info("未发现特征缓存，自动从头重算（VMD/DPSR/DLFE）。")
        return True

    env = os.getenv("DLFE_FORCE_REBUILD", "").strip().lower()
    if env in {"1", "true", "yes", "y"}:
        logger.info("DLFE_FORCE_REBUILD=%s，按要求从头重算特征。", env)
        return True
    if env in {"0", "false", "no", "n"}:
        logger.info("DLFE_FORCE_REBUILD=%s，沿用现有特征缓存。", env)
        return False

    try:
        if not sys.stdin.isatty():
            logger.info("检测到非交互环境，默认使用特征缓存（可用 DLFE_FORCE_REBUILD 覆盖）。")
            return False
    except Exception:
        return False

    print()
    print("检测到特征缓存 (data/features/*.npz)")
    print("1) 使用缓存（更快）")
    print("2) 从头重算（执行 VMD/DPSR/DLFE 全流程）")
    print("3) 取消")
    sel = input("请选择 [默认1]: ").strip()
    if sel == "2":
        return True
    if sel == "3":
        print("已取消。")
        raise SystemExit(0)
    return False

def run_prepare(args: argparse.Namespace, config: Dict, paths: PipelinePaths, logger) -> None:
    if not getattr(args, "force_rebuild", False):
        args.force_rebuild = choose_feature_rebuild(args, paths, logger)
    prepare_feature_sets(config, paths, logger, args.force_rebuild)
    logger.info("特征缓存已准备完毕")


def main() -> None:
    args = parse_args()
    runtime = initialize_project(config_path=args.config)
    config = runtime["config"]
    logger = runtime["logger"]

    try:
        run_name = getattr(args, "run_name", None)
        paths = resolve_paths(config, run_name)

        if args.mode == "prepare":
            run_prepare(args, config, paths, logger)
        elif args.mode == "walk-forward":
            run_walk_forward(args, config, paths, logger)
        elif args.mode == "train":
            chosen_mode = choose_train_mode(args)
            if chosen_mode == "walk-forward":
                run_walk_forward(args, config, paths, logger)
            else:
                run_train(args, config, paths, logger)
        elif args.mode == "test":
            run_test(args, config, paths, logger)
        else:
            logger.error("模式 %s 暂未实现", args.mode)
    except Exception as exc:
        logger.error("执行过程中发生异常", exc_info=True)
        raise exc
    finally:
        logger.close()


if __name__ == "__main__":
    main()
