#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
璇婃柇/鏍″噯澶╂皵鍒嗙被锛圕I/WSI锛夎剼鏈?
- 缁熻鍚?split 鐨?sunny/cloudy/overcast 鏍锋湰涓庡簭鍒楁牱鏈垎甯?
- 鍒嗘瀽 CI-only 涓?WSI-only 鐨勫垎姝?
- 璇勪及鏄煎褰卞搷锛堝熀浜庡湴澶栬緪鐓у害 GE 鐨勭櫧澶╂帺鐮侊級
- 缃戞牸鎼滅储闃堝€硷紝杈撳嚭寤鸿闃堝€?JSON锛堝吋椤剧洰鏍囧垎甯?鍒嗘鐜?鏄煎鎯╃綒锛?

杈撳嚭鐩綍锛歟xperiments/runs/<run-name>/artifacts/weather_diagnostics/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yaml

# 澶嶇敤椤圭洰妯″潡
from src.data_processing.data_loader import DataLoader as PVDataLoader
from src.data_processing.preprocessor import Preprocessor
from src.feature_engineering.weather_classifier import WeatherClassifier
logging.getLogger("src.feature_engineering.weather_classifier").setLevel(logging.WARNING)

DEFAULT_TARGET_DIST = [0.33, 0.34, 0.33]  # sunny, cloudy, overcast


def load_yaml_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_run_paths(config: Dict, run_name: Optional[str]) -> Tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    runs_root = Path(config.get("project", {}).get("runs_dir", project_root / "experiments" / "runs"))
    run_dir = runs_root / (run_name or "diagnose")
    artifacts_dir = run_dir / "artifacts" / "weather_diagnostics"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, artifacts_dir


def read_splits_from_disk(root: Path) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    splits_dir = root / "data" / "splits"
    train_fp = splits_dir / "train.parquet"
    val_fp = splits_dir / "val.parquet"
    test_fp = splits_dir / "test.parquet"
    if train_fp.exists() and val_fp.exists() and test_fp.exists():
        return (
            pd.read_parquet(train_fp),
            pd.read_parquet(val_fp),
            pd.read_parquet(test_fp),
        )
    return None


def prepare_processed_splits(config: Dict, project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 澶嶇敤 DataLoader 鍚堝苟涓庡熀纭€缂哄け澶勭悊
    data_root = Path(config.get("data", {}).get("raw_dir", project_root / "data" / "raw"))
    loader = PVDataLoader(data_path=str(data_root))
    station_data = loader.load_multi_station(
        merge_method=config.get("data", {}).get("merge_method", "concat"),
        selected_station=config.get("data", {}).get("selected_station"),
    )
    if not station_data:
        raise FileNotFoundError(f"鏈湪 {data_root} 鎵惧埌浠讳綍鏁版嵁鏂囦欢")

    if len(station_data) == 1 or config.get("data", {}).get("merge_method") == "single":
        merged = next(iter(station_data.values()))
    else:
        merged = loader.merge_stations(station_data, method=config.get("data", {}).get("merge_method", "concat"))
    merged = loader.handle_missing_values(merged)

    # 绠€鍗曟寜 70/20/10 鏃堕棿椤哄簭鍒囧垎锛堜笌椤圭洰涓绘祦绋嬩竴鑷达級
    n = len(merged)
    i_train = int(n * config.get("data", {}).get("train_ratio", 0.7))
    i_val = int(n * (config.get("data", {}).get("train_ratio", 0.7) + config.get("data", {}).get("val_ratio", 0.2)))
    train_raw = merged.iloc[:i_train]
    val_raw = merged.iloc[i_train:i_val]
    test_raw = merged.iloc[i_val:]
    return train_raw, val_raw, test_raw


def preprocess_by_train(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    config: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    norm_cfg = config.get("preprocessing", {}).get("normalization", {})
    pre = Preprocessor(
        method=norm_cfg.get("method", "minmax"),
        feature_range=tuple(norm_cfg.get("feature_range", [0, 1])),
    )
    pre.fit(train_raw)
    return pre.transform(train_raw), pre.transform(val_raw), pre.transform(test_raw)



def compute_ci_wsi_and_labels(df_raw: pd.DataFrame, clf: WeatherClassifier) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # CI/WSI 璁＄畻
    ci, day_mask = clf.calculate_ci(df_raw["irradiance"].values, df_raw.index)
    wsi = clf.calculate_wsi(df_raw["pressure"].values, df_raw["humidity"].values, df_raw["temperature"].values)
    # 鍒嗙被锛堝崟璺緞锛?
    ci_cls = clf.classify_ci(ci)
    wsi_cls = clf.classify_wsi(wsi)
    # 铻嶅悎鍒嗙被
    fused_bundle = clf.classify(df_raw)
    fused = fused_bundle["labels"]
    day_mask = fused_bundle.get("day_mask", np.asarray(day_mask, dtype=bool))
    return ci, wsi, ci_cls, wsi_cls, fused, day_mask



def build_day_mask(df_raw: pd.DataFrame, clf: WeatherClassifier, ge_min: float = 20.0) -> np.ndarray:
    ge = clf._calculate_extraterrestrial_radiation(df_raw.index)  # type: ignore (鍐呴儴鏂规硶)
    return (ge >= ge_min).astype(bool)


def seq_label_from_pointwise(labels: np.ndarray, seq_len: int) -> np.ndarray:
    if labels.size <= seq_len:
        return np.empty((0,), dtype=int)
    # 鏈瀵归綈锛屽彇姣忎釜搴忓垪鏈€鍚庝竴涓椂闂存鐨勬爣绛?
    n = labels.size - seq_len + 1
    out = np.empty((n,), dtype=int)
    out[:] = labels[seq_len - 1 :]
    return out


def summarize_distribution(name: str, labels: np.ndarray) -> Dict[str, int]:
    mapping = {0: "sunny", 1: "cloudy", 2: "overcast"}
    counts = {"sunny": 0, "cloudy": 0, "overcast": 0}
    uniq, cnt = np.unique(labels, return_counts=True)
    for u, c in zip(uniq, cnt):
        counts[mapping.get(int(u), "?")] = int(c)
    total = int(labels.size)
    print(f"[{name}] 搴忓垪鏍锋湰鍒嗗竷: {counts} (total={total})")
    return counts


def grid_search_thresholds(
    train_raw: pd.DataFrame,
    seq_len: int,
    target_dist: List[float],
    ge_min: float,
    ci_grid: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.15, 0.30), (0.55, 0.70)),
    wsi_grid: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.25, 0.40), (0.65, 0.80)),
    step: float = 0.03,
    min_count: int = 500,
    log_best_only: bool = False,
) -> Dict:
    """在网格上搜索阈值组合并返回目标函数最优的候选列表。"""
    candidates: List[Dict] = []
    improvements: List[Dict] = []
    target = np.array(target_dist, dtype=float)
    target = target / max(target.sum(), 1e-9)
    best_obj = float("inf")

    for ci_low in np.arange(ci_grid[0][0], ci_grid[0][1] + 1e-9, step):
        for ci_high in np.arange(ci_grid[1][0], ci_grid[1][1] + 1e-9, step):
            if ci_low >= ci_high:
                continue
            for wsi_low in np.arange(wsi_grid[0][0], wsi_grid[0][1] + 1e-9, step):
                for wsi_high in np.arange(wsi_grid[1][0], wsi_grid[1][1] + 1e-9, step):
                    if wsi_low >= wsi_high:
                        continue
                    clf = WeatherClassifier(
                        ci_thresholds=[float(ci_low), float(ci_high)],
                        wsi_thresholds=[float(wsi_low), float(wsi_high)],
                        daytime_ge_min=ge_min,
                        daytime_mode=str(config.get("feature_engineering", {}).get("daytime", {}).get("mode", "ghi")).lower(),
                        daytime_ghi_min=float(config.get("feature_engineering", {}).get("daytime", {}).get("ghi_min_wm2", 5.0)),
                    )
                    ci, wsi, ci_cls, wsi_cls, fused, day_mask = compute_ci_wsi_and_labels(train_raw, clf)
                    day_points = day_mask.astype(bool)
                    disagree = float(np.mean(ci_cls[day_points] != wsi_cls[day_points])) if day_points.any() else 0.0
                    seq_labels = seq_label_from_pointwise(fused, seq_len)
                    seq_day = day_points[seq_len - 1 :]
                    if seq_labels.size and seq_day.size:
                        seq_labels = seq_labels[seq_day]
                    night_ratio = float(np.mean(~seq_day)) if seq_day.size else 1.0
                    counts = np.array(
                        [
                            int(np.sum(seq_labels == 0)),
                            int(np.sum(seq_labels == 1)),
                            int(np.sum(seq_labels == 2)),
                        ],
                        dtype=int,
                    )
                    if any(int(c) < min_count for c in counts):
                        continue
                    dist = counts / max(counts.sum(), 1e-9)
                    l1 = float(np.abs(dist - target).sum())
                    obj = 1.0 * l1 + 0.5 * disagree + 0.3 * night_ratio
                    entry = {
                        "ci_thresholds": [float(ci_low), float(ci_high)],
                        "wsi_thresholds": [float(wsi_low), float(wsi_high)],
                        "dist": dist.astype(float).tolist(),
                        "counts": counts.tolist(),
                        "disagree": disagree,
                        "night_ratio": night_ratio,
                        "objective": obj,
                    }
                    if obj < best_obj - 1e-9:
                        best_obj = obj
                        improvements.append(entry)
                    if not log_best_only:
                        candidates.append(entry)
    if log_best_only:
        top = improvements[-10:]
    else:
        ranked = sorted(candidates, key=lambda x: x["objective"])
        top = ranked[:10]
    return {"top": top, "improvements": improvements}






def main():
    parser = argparse.ArgumentParser(description="璇婃柇/鏍″噯 CI/WSI 澶╂皵鍒嗙被")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config" / "config.yaml"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--use-cache", action="store_true", help="浼樺厛璇诲彇 data/splits 缂撳瓨")
    parser.add_argument("--calibrate", action="store_true", help="鎵ц闃堝€肩綉鏍兼悳绱㈠苟杈撳嚭寤鸿")
    parser.add_argument("--target-dist", type=float, nargs=3, default=DEFAULT_TARGET_DIST)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--ge-min", type=float, default=20.0, help="GE (W/m^2) threshold for daytime")
    parser.add_argument("--log-best-only", action="store_true", help="Only print candidates that improve objective")
    parser.add_argument("--min-count", type=int, default=500, help="Minimum sequences per class for calibration candidates")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    project_root = Path(__file__).resolve().parents[1]
    # 浠呯粓绔緭鍑猴紝鏃犻渶杩愯鐩綍涓庤惤鐩?

    # 璇诲彇鎴栨瀯寤?splits
    raw_triplet = read_splits_from_disk(project_root) if args.use_cache else None
    if raw_triplet is None:
        raw_triplet = prepare_processed_splits(config, project_root)

    train_raw, val_raw, test_raw = raw_triplet
    train_proc, val_proc, test_proc = preprocess_by_train(train_raw, val_raw, test_raw, config)

    # 浣跨敤鍘熼厤缃槇鍊煎厛鍋氫竴娆¤瘖鏂?
    fe_cfg = config.get("feature_engineering", {})
    daytime_cfg = fe_cfg.get("daytime", {})
    loc_cfg = config.get("data", {}).get("location", {}) or {}
    clf = WeatherClassifier(
        location_lat=float(loc_cfg.get("lat", 38.5)),
        location_lon=float(loc_cfg.get("lon", 105.0)),
        elevation=float(loc_cfg.get("elevation", 1500.0)),
        time_zone_hours=float(loc_cfg.get("time_zone_hours", 8.0)),
        ci_thresholds=fe_cfg.get("ci_thresholds", [0.2, 0.6]),
        wsi_thresholds=fe_cfg.get("wsi_thresholds", [0.3, 0.7]),
        fusion_weights=fe_cfg.get("fusion_weights", {"ci": 0.7, "wsi": 0.3}),
        daytime_ge_min=daytime_cfg.get("ge_min_wm2", args.ge_min),
        daytime_mode=str(daytime_cfg.get("mode", "ghi")).lower(),
        daytime_ghi_min=float(daytime_cfg.get("ghi_min_wm2", 5.0)),
        night_handling=daytime_cfg.get("night_handling", "exclude"),
    )

    results: Dict[str, Dict] = {"splits": {}}

    for name, df in ("train", train_raw), ("val", val_raw), ("test", test_raw):
        ci, wsi, ci_cls, wsi_cls, fused, day_mask = compute_ci_wsi_and_labels(df, clf)
        day_points = day_mask.astype(bool)
        disagree_rate = float(np.mean(ci_cls[day_points] != wsi_cls[day_points])) if day_points.any() else 0.0
        seq_labels = seq_label_from_pointwise(fused, args.sequence_length)
        seq_day = day_points[args.sequence_length - 1 :]
        if seq_labels.size and seq_day.size:
            seq_labels = seq_labels[seq_day]
        counts = summarize_distribution(name, seq_labels)
        results["splits"][name] = {
            "disagree_rate": disagree_rate,
            "sequence_counts": counts,
            "day_ratio": float(np.mean(day_points)) if day_points.size else 0.0,
        }


    # 鍙€夛細缃戞牸鎼滅储寤鸿闃堝€硷紙鍩轰簬璁粌闆嗭級
    if args.calibrate:
        suggestions = grid_search_thresholds(
            train_raw=train_raw,
            seq_len=args.sequence_length,
            target_dist=args.target_dist,
            ge_min=args.ge_min,
            step=0.03,
            min_count=args.min_count,
            log_best_only=args.log_best_only,
        )
        results["suggestions"] = suggestions
        improvements = suggestions.get("improvements", [])
        if improvements:
            for entry in improvements:
                print(
                    f"[improved] CI={entry['ci_thresholds']} WSI={entry['wsi_thresholds']} "
                    f"objective={entry['objective']:.4f} counts={entry['counts']}"
                )
        elif not suggestions.get("top"):
            print("No calibration candidate satisfied the constraints; nothing to log.")

    # Console output only (no files written)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("Diagnostics complete (not written to file)")


if __name__ == "__main__":
    main()
