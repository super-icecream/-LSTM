#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Init Thresholds (CI/WSI) Script with GPU Acceleration

Strictly follows 草稿纸.md requirements, and accelerates where possible:
- Data load consistent with main.py (config/config.yaml, DataLoader/DataSplitter)
- Feature engineering on GPU (PyTorch CUDA) for rolling mean/std/diff when available
- KMeans clustering accelerated on GPU (custom torch CUDA); fallbacks: scikit-learn, then NumPy
- Threshold inference: gaussian|kde|quantile with bootstrap; CPU is fine (cheap)
- Diagnostics and artifacts output
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import sys
import time
import pickle

# Optional deps
try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.metrics import silhouette_score  # type: ignore
    from sklearn.preprocessing import RobustScaler, StandardScaler  # type: ignore
    from sklearn.mixture import GaussianMixture  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False
try:
    from scipy.stats import gaussian_kde  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
try:
    import torch  # type: ignore
    _HAVE_TORCH = True
    _CUDA_OK = torch.cuda.is_available()
except Exception:
    _HAVE_TORCH = False
    _CUDA_OK = False

try:
    import lightgbm as lgb  # type: ignore
    _HAVE_LIGHTGBM = True
except Exception:
    _HAVE_LIGHTGBM = False

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Project modules
from src.data_processing.data_loader import DataLoader as PVDataLoader
from src.data_processing.data_splitter import DataSplitter
from src.feature_engineering.weather_classifier import WeatherClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "diagnostics" / "init_thresholds"
WEATHER_NAME = {0: "sunny", 1: "cloudy", 2: "overcast"}

logger = logging.getLogger("init_thresholds")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_ch)

class OneLineProgress:
    """
    Single-line progress printer to avoid noisy terminal refresh.
    Use start(name) -> update(percent, msg) -> done(ok, summary)
    """
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.stage = None
        self._last = ""
        self._last_pct = -1
        self._last_msg = ""
        self._last_ts = 0.0
        self._newlined = True

    def _render(self, s: str):
        if s == self._last:
            return
        self.stream.write("\r" + s)
        pad = max(0, len(self._last) - len(s))
        if pad:
            self.stream.write(" " * pad)
        self.stream.flush()
        self._last = s
        self._newlined = False

    def start(self, name: str):
        self.stage = name
        self._last_pct = -1
        self._last_msg = ""
        self._render(f"[{name}] 0% [--------------------]")

    def update(self, percent: int, msg: str = ""):
        if self.stage is None:
            return
        percent = int(max(0, min(100, percent)))
        now = time.time()
        # throttle to ~30ms and skip duplicates
        if percent == self._last_pct and msg == self._last_msg:
            return
        if now - self._last_ts < 0.03 and percent != 100:
            return
        self._last_ts = now
        self._last_pct = percent
        self._last_msg = msg
        bar_len = 20
        filled = int(round(percent / 100 * bar_len))
        bar = "#" * filled + "-" * (bar_len - filled)
        detail = f" {msg}" if msg else ""
        self._render(f"[{self.stage}] {percent:3d}% [{bar}]{detail}")

    def done(self, ok: bool = True, summary: str = ""):
        if self.stage is None:
            return
        mark = "✓" if ok else "✗"
        detail = f" {summary}" if summary else ""
        self._render(f"[{self.stage}] 100% [####################] {mark}{detail}")
        if not self._newlined:
            self.stream.write("\n")
            self.stream.flush()
            self._newlined = True
        self.stage = None
        self._last = ""
        self._last_pct = -1
        self._last_msg = ""
        self._last_ts = 0.0


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def dump_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_data_paths(config: Dict) -> Tuple[Path, Path, Path]:
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    # Align with main.py convention:
    # root_dir defaults to <project>/data, with subdirs raw/processed/splits
    root_dir = Path(data_cfg.get("root_dir", PROJECT_ROOT / "data"))
    raw_dir = Path(data_cfg.get("raw_dir", root_dir / "raw"))
    processed_dir = Path(data_cfg.get("processed_dir", root_dir / "processed"))
    splits_dir = Path(data_cfg.get("splits_dir", root_dir / "splits"))
    return raw_dir, processed_dir, splits_dir


def read_or_prepare_splits(config: Dict,
                           raw_dir: Path,
                           processed_dir: Path,
                           splits_dir: Path,
                           *,
                           config_path: Path = DEFAULT_CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_fp = processed_dir / "merged.parquet"
    train_fp = splits_dir / "train.parquet"
    val_fp = splits_dir / "val.parquet"
    test_fp = splits_dir / "test.parquet"
    if merged_fp.exists() and train_fp.exists() and val_fp.exists() and test_fp.exists():
        logger.info("Reading existing processed/splits from disk")
        return pd.read_parquet(train_fp), pd.read_parquet(val_fp), pd.read_parquet(test_fp)

    logger.info("No existing processed/splits. Building via DataLoader/DataSplitter ...")
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    merge_method = config.get("data", {}).get("merge_method", "single")
    selected_station = config.get("data", {}).get("selected_station", None)
    loader = PVDataLoader(data_path=str(raw_dir), config_path=str(config_path))
    stations = loader.load_multi_station(merge_method=merge_method, selected_station=selected_station)
    if not stations:
        raise FileNotFoundError(f"No CSV/Excel under {raw_dir}")
    merged = next(iter(stations.values())) if (merge_method == "single" or len(stations) == 1) else loader.merge_stations(stations, method=merge_method or "concat")
    merged = loader.handle_missing_values(merged)
    ok, quality = loader.validate_data_quality(merged)
    if not ok:
        logger.warning("Data quality issues detected: %s", quality)
    merged.to_parquet(merged_fp)

    splitter = DataSplitter(
        train_ratio=float(config.get("data", {}).get("train_ratio", 0.7)),
        val_ratio=float(config.get("data", {}).get("val_ratio", 0.2)),
        test_ratio=float(config.get("data", {}).get("test_ratio", 0.1)),
        seed=int(config.get("project", {}).get("seed", 42)),
    )
    train_df, val_df, test_df = splitter.split_temporal(merged, keep_continuity=True)
    splitter.save_splits(train_df, val_df, test_df, output_dir=splits_dir, formats=["parquet"])
    return train_df, val_df, test_df


def slice_split(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df


def engineer_features(df: pd.DataFrame,
                      window_mins: int,
                      min_samples_per_window: int,
                      scaler_name: str,
                      use_pca: bool,
                      pca_n: int,
                      freq_minutes: int,
                      *,
                      clf: Optional[WeatherClassifier] = None,
                      feat_weight_tod: float = 2.0,
                      feat_weight_diffuse: float = 2.0,
                      feat_weight_z: float = 1.5,
                      features_basic_only: bool = False,
                      return_transforms: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Build clustering features excluding CI/WSI.
    GPU acceleration:
      - If torch.cuda is available, rolling mean/std/diff computed on GPU using cumsum
      - Otherwise fall back to pandas.rolling
    
    Args:
        return_transforms: If True, meta will include 'scaler' and 'pca' objects for later reuse.
    """
    df = df.copy()
    required = ["power", "irradiance", "temperature", "pressure", "humidity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    # 时间编码（Time-of-day 正余弦），用于分离清晨/傍晚与正午的相似功率段
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("索引需为 DatetimeIndex 以计算时间编码")
    _mins = (df.index.hour * 60 + df.index.minute + (df.index.second / 60.0)).astype(float)
    _t = _mins / 1440.0  # [0,1) 日内相位
    df["tod_sin"] = np.sin(2 * np.pi * _t)
    df["tod_cos"] = np.cos(2 * np.pi * _t)
    # 日内分钟桶（用于稳健分组统计，避免把同一时刻的“相对异常”混在一起）
    df["tod_bin"] = (df.index.hour * 60 + df.index.minute).astype(int)

    # 直射/扩散与相对异常度特征（软引导）
    # - dni_over_ghi: 直射占比，晴正午高，多云正午低；若无 dni 列则跳过
    # - 按分钟-of-day分组的稳健 z-score：z_power/z_ghi/z_hum/z_temp/z_pres（及 z_dni 若可用）
    def _robust_group_z(series: pd.Series, group: pd.Series, clip: float = 5.0) -> np.ndarray:
        g = series.groupby(group)
        med = g.transform("median")
        # 使用组内中位数填充缺失，避免后续 dropna
        s = series.copy()
        s[pd.isna(s)] = med[pd.isna(s)]
        q1 = s.groupby(group).transform(lambda x: x.quantile(0.25))
        q3 = s.groupby(group).transform(lambda x: x.quantile(0.75))
        iqr = (q3 - q1).astype(float)
        iqr = iqr.replace(0.0, np.nan).fillna(1.0)
        z = (s - med) / iqr
        if clip is not None:
            z = z.clip(lower=-clip, upper=clip)
        return z.values.astype(float)

    if "dni" in df.columns:
        ghi = df["irradiance"].astype(float).values
        dni = df["dni"].astype(float).values
        dni_filled = np.nan_to_num(dni, nan=0.0)
        # 计算 cosZ = sin(elevation) ≈ GE / (I0*E0)
        cosZ = None
        try:
            if clf is not None:
                ge_vec = np.asarray(clf._calculate_extraterrestrial_radiation(df.index), dtype=float)
                # 计算 E0（日地距离修正）
                doy = df.index.dayofyear.values.astype(float)
                gamma = 2 * np.pi * (doy - 1.0) / 365.0
                e0 = (1.000110 +
                      0.034221 * np.cos(gamma) +
                      0.001280 * np.sin(gamma) +
                      0.000719 * np.cos(2 * gamma) +
                      0.000077 * np.sin(2 * gamma))
                i0 = float(getattr(clf, "solar_constant", 1367.0))
                cosZ = np.clip(ge_vec / (i0 * e0), 0.0, 1.0)
        except Exception:
            cosZ = None
        if cosZ is None:
            # 回退：不做几何修正（弱化效果），尽量不断流程
            cosZ = np.ones_like(dni_filled)
        # 水平直射占比 BF = (DNI*cosZ)/GHI，散射比 DF = 1 - BF
        bf = (dni_filled * cosZ) / np.maximum(ghi, 1e-6)
        bf = np.clip(bf, 0.0, 1.5)
        df["dni_over_ghi"] = bf  # 复用列名，实际存放 BF
        # z-score for DNI itself（仍按分钟桶）
        df["z_dni"] = _robust_group_z(pd.Series(dni, index=df.index), df["tod_bin"])
    # 核心观测的稳健 z-score（按日内分钟分组） - 基础字段模式下跳过
    if not features_basic_only:
        df["z_power"] = _robust_group_z(df["power"].astype(float), df["tod_bin"])
        df["z_ghi"] = _robust_group_z(df["irradiance"].astype(float), df["tod_bin"])
        df["z_hum"] = _robust_group_z(df["humidity"].astype(float), df["tod_bin"])
        df["z_temp"] = _robust_group_z(df["temperature"].astype(float), df["tod_bin"])
        df["z_pres"] = _robust_group_z(df["pressure"].astype(float), df["tod_bin"])

    steps = max(int(round(window_mins / max(freq_minutes, 1))), 1)
    # Consistency guard: ensure required samples per window does not exceed window steps.
    # Otherwise pandas.rolling(min_periods=...) would yield all-NaN and dropna() empties the frame.
    if min_samples_per_window > steps:
        logger.warning(
            "min-samples-per-window (%d) 大于窗口步数 (%d)，已自动调整为 %d",
            int(min_samples_per_window), int(steps), int(steps)
        )
        min_samples_per_window = int(steps)
    roll_cols = ["power", "irradiance", "humidity", "pressure"]

    if features_basic_only:
        # 跳过滚动统计
        pass
    elif _HAVE_TORCH and _CUDA_OK:
        device = torch.device("cuda")
        arr = torch.tensor(df[roll_cols].values, dtype=torch.float32, device=device)
        n, c = arr.shape
        # handle NaNs in GPU path via mask + filled arrays
        finite_mask = torch.isfinite(arr)
        arr_filled = torch.where(finite_mask, arr, torch.zeros_like(arr))
        # diff1 (set NaN where neighbors invalid)
        diff = torch.empty_like(arr_filled)
        diff[0, :] = float("nan")
        dcore = arr_filled[1:, :] - arr_filled[:-1, :]
        valid_edges = finite_mask[1:, :] & finite_mask[:-1, :]
        dcore = torch.where(valid_edges, dcore, torch.full_like(dcore, float("nan")))
        diff[1:, :] = dcore
        # rolling mean/std with fixed window = steps
        def rolling_mean_std(x: torch.Tensor, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
            # Use prefix-padding trick with valid-counts to support NaNs and min_samples_per_window.
            # x: [n, c]; returns mean/std aligned to original length with first windows possibly NaN.
            n, c2 = x.shape
            valid = torch.isfinite(x).to(x.dtype)
            xf = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
            cs = torch.cumsum(xf, dim=0)          # [n, c]
            cs2 = torch.cumsum(xf * xf, dim=0)    # [n, c]
            cc = torch.cumsum(valid, dim=0)       # [n, c]
            zero = torch.zeros((1, c2), device=x.device, dtype=x.dtype)
            cs_pad = torch.vstack([zero, cs])    # [n+1, c]
            cs2_pad = torch.vstack([zero, cs2])  # [n+1, c]
            cc_pad = torch.vstack([zero, cc])    # [n+1, c]
            # window sums over last w samples
            win_sum = cs_pad[w:] - cs_pad[:-w]       # [n+1-w, c]
            win_sum2 = cs2_pad[w:] - cs2_pad[:-w]    # [n+1-w, c]
            win_cnt = cc_pad[w:] - cc_pad[:-w]       # [n+1-w, c]
            # avoid division by zero
            mean_core = win_sum / torch.clamp(win_cnt, min=1.0)
            var_core = torch.clamp(win_sum2 / torch.clamp(win_cnt, min=1.0) - mean_core * mean_core, min=0.0)
            std_core = torch.sqrt(var_core)
            # mask out insufficient samples (< min_samples_per_window)
            need = float(max(min_samples_per_window, 1))
            valid_core = win_cnt >= need
            mean_core = torch.where(valid_core, mean_core, torch.full_like(mean_core, float("nan")))
            std_core = torch.where(valid_core, std_core, torch.full_like(std_core, float("nan")))
            # align back to length n
            if w > 1:
                nan_pad = torch.full((w - 1, c2), float("nan"), device=x.device, dtype=x.dtype)
                mean = torch.vstack([nan_pad, mean_core])  # [n, c]
                std = torch.vstack([nan_pad, std_core])    # [n, c]
            else:
                mean, std = mean_core, std_core
            return mean, std

        rmean, rstd = rolling_mean_std(arr, steps)
        rmean_np = rmean.detach().cpu().numpy()
        rstd_np = rstd.detach().cpu().numpy()
        diff_np = diff.detach().cpu().numpy()
        for j, col in enumerate(roll_cols):
            df[f"{col}_rmean"] = rmean_np[:, j]
            df[f"{col}_rstd"] = rstd_np[:, j]
            df[f"{col}_diff1"] = diff_np[:, j]
    elif not features_basic_only:
        for col in roll_cols:
            r = df[col].rolling(window=steps, min_periods=min_samples_per_window)
            df[f"{col}_rmean"] = r.mean()
            df[f"{col}_rstd"] = r.std()
            df[f"{col}_diff1"] = df[col].diff(1)

    # 组装特征矩阵
    if features_basic_only:
        base_cols = ["power", "irradiance", "temperature", "pressure", "humidity", "tod_sin", "tod_cos"]
        if "dni" in df.columns:
            base_cols.append("dni")
        if "irradiance_total" in df.columns:
            base_cols.append("irradiance_total")
        df_feat = df[base_cols].dropna().copy()
        feature_cols = base_cols
    else:
        df_feat = df.dropna().copy()
        feature_cols = [
            "power", "irradiance", "temperature", "pressure", "humidity",
            "power_rmean", "power_rstd", "power_diff1",
            "irradiance_rmean", "irradiance_rstd", "irradiance_diff1",
            "humidity_rmean", "humidity_rstd", "humidity_diff1",
            "pressure_rmean", "pressure_rstd", "pressure_diff1",
            "tod_sin", "tod_cos",
        ]
        # 可用时追加直射占比与 z-score 特征（dni 可选）
        if "dni_over_ghi" in df.columns:
            feature_cols.append("dni_over_ghi")
            if "z_dni" in df.columns:
                feature_cols.append("z_dni")
        feature_cols.extend([c for c in ["z_power", "z_ghi", "z_hum", "z_temp", "z_pres"] if c in df.columns])

    if df_feat.empty:
        raise ValueError("窗口统计后无有效样本，请调整 --window-mins 或 --min-samples-per-window")
    X = df_feat[feature_cols].values.astype(float)

    scaler_obj = None
    if _HAVE_SKLEARN:
        scaler_obj = StandardScaler() if scaler_name == "standard" else RobustScaler()
        Xs = scaler_obj.fit_transform(X)
    else:
        med = np.nanmedian(X, axis=0)
        iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
        iqr[iqr == 0] = 1.0
        Xs = (X - med) / iqr
        # Store NumPy fallback params for later use
        scaler_obj = {"type": "numpy_fallback", "median": med.tolist(), "iqr": iqr.tolist()}

    # Post-scaling feature weighting: emphasize time-of-day, diffuse ratio, and robust z-scores
    weights = np.ones(len(feature_cols), dtype=float)
    name_to_idx = {n: i for i, n in enumerate(feature_cols)}
    for n in ("tod_sin", "tod_cos"):
        if n in name_to_idx:
            weights[name_to_idx[n]] = float(feat_weight_tod)
    if "dni_over_ghi" in name_to_idx:
        weights[name_to_idx["dni_over_ghi"]] = float(feat_weight_diffuse)
    for n in list(name_to_idx.keys()):
        if n.startswith("z_"):
            weights[name_to_idx[n]] = float(feat_weight_z)
    Xs = Xs * weights[np.newaxis, :]

    meta = {"feature_cols": feature_cols, "steps": steps, "index": df_feat.index, "weights": weights.tolist()}
    
    # Store scaler if requested
    if return_transforms:
        meta["scaler"] = scaler_obj
    
    # PCA: prefer GPU (torch) if available; else sklearn; else NumPy SVD
    pca_obj = None
    if use_pca and pca_n > 0:
        n_comp = int(max(1, min(pca_n, Xs.shape[1] - 1)))
        if _HAVE_TORCH and _CUDA_OK:
            # GPU PCA via torch.pca_lowrank / SVD
            device = torch.device("cuda")
            Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
            # center
            mean = Xt.mean(dim=0, keepdim=True)
            Xc = Xt - mean
            try:
                # Efficient for tall matrices
                U, S, V = torch.pca_lowrank(Xc, q=n_comp)
            except Exception:
                # Fallback to full SVD on GPU
                U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
                V = Vh.t()
                # Trim to n_comp
                U = U[:, :n_comp]
                S = S[:n_comp]
                V = V[:, :n_comp]
            # Project to first n_comp
            comps = V[:, :n_comp]
            X_proj = (Xc @ comps).detach().cpu().numpy()
            # Explained variance ratio (sum)
            n_samples = Xs.shape[0]
            var_explained = (S[:n_comp] ** 2) / max(n_samples - 1, 1)
            total_var = torch.var(Xc, dim=0, unbiased=True).sum()
            evr_sum = (var_explained.sum() / torch.clamp(total_var, min=1e-9)).item()
            Xs = X_proj
            meta["pca_explained_var"] = float(evr_sum)
            # Store GPU PCA params for later use (as numpy arrays)
            if return_transforms:
                pca_obj = {
                    "type": "gpu_pca",
                    "mean": mean.detach().cpu().numpy().flatten().tolist(),
                    "components": comps.detach().cpu().numpy().tolist(),
                    "n_components": n_comp,
                }
            # Release GPU memory
            del Xt, Xc, U, S, V, comps
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        elif _HAVE_SKLEARN:
            pca_obj = PCA(n_components=n_comp, random_state=42)
            Xs = pca_obj.fit_transform(Xs)
            meta["pca_explained_var"] = float(np.sum(pca_obj.explained_variance_ratio_))
        else:
            # NumPy SVD fallback on CPU
            Xc = Xs - Xs.mean(axis=0, keepdims=True)
            pca_mean = Xs.mean(axis=0)
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            comps = Vt[:n_comp].T  # (d, n_comp)
            Xs = Xc @ comps
            n_samples = Xs.shape[0]
            var_explained = (s[:n_comp] ** 2) / max(n_samples - 1, 1)
            total_var = Xc.var(axis=0, ddof=1).sum()
            meta["pca_explained_var"] = float((var_explained.sum() / max(total_var, 1e-9)))
            if return_transforms:
                pca_obj = {
                    "type": "numpy_pca",
                    "mean": pca_mean.tolist(),
                    "components": comps.tolist(),
                    "n_components": n_comp,
                }
        if return_transforms and pca_obj is not None:
            meta["pca"] = pca_obj
    
    return Xs, meta


def kmeans_cluster(X: np.ndarray,
                   power_series: pd.Series,
                   index: pd.DatetimeIndex,
                   seed: int,
                   n_init: int,
                   max_iter: int,
                   init_centers: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    def _numpy_kmeans(X_: np.ndarray, k: int, n_init_: int, max_iter_: int, seed_: int,
                      init_centers_: Optional[np.ndarray] = None) -> np.ndarray:
        rng = np.random.default_rng(seed_)
        best_labels = None
        best_inertia = np.inf
        n = X_.shape[0]
        trials = 1 if init_centers_ is not None else max(1, n_init_)
        for _ in range(trials):
            if init_centers_ is not None and init_centers_.shape == (k, X_.shape[1]):
                centers = init_centers_.copy()
            else:
                idxs = rng.choice(n, size=min(k, n), replace=False)
                centers = X_[idxs].copy()
            labels = np.zeros(n, dtype=int)
            for _it in range(max(1, max_iter_)):
                d2 = ((X_[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                new_labels = np.argmin(d2, axis=1)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                for j in range(k):
                    pts = X_[labels == j]
                    centers[j] = pts.mean(axis=0) if len(pts) else X_[rng.integers(0, n)]
            inertia = 0.0
            for j in range(k):
                pts = X_[labels == j]
                if len(pts):
                    inertia += float(((pts - centers[j]) ** 2).sum())
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
        return best_labels if best_labels is not None else np.zeros(n, dtype=int)

    def _torch_kmeans_gpu(X_: np.ndarray, k: int, n_init_: int, max_iter_: int, seed_: int,
                          init_centers_: Optional[np.ndarray] = None) -> np.ndarray:
        if not (_HAVE_TORCH and _CUDA_OK):
            raise RuntimeError("Torch/CUDA not available")
        device = torch.device("cuda")
        xt = torch.tensor(X_, dtype=torch.float32, device=device)
        n = xt.shape[0]
        best_labels_t = None
        best_inertia = float("inf")
        g = torch.Generator(device=device)
        g.manual_seed(seed_)
        trials = 1 if init_centers_ is not None else max(1, n_init_)
        for _ in range(trials):
            if init_centers_ is not None and init_centers_.shape == (k, X_.shape[1]):
                centers = torch.tensor(init_centers_, dtype=torch.float32, device=device).clone()
            else:
                if n >= k:
                    idx = torch.randperm(n, generator=g, device=device)[:k]
                else:
                    idx = torch.randint(low=0, high=n, size=(k,), generator=g, device=device)
                centers = xt[idx].clone()
            labels = torch.zeros((n,), dtype=torch.long, device=device)
            for _it in range(max(1, max_iter_)):
                x2 = (xt * xt).sum(dim=1, keepdim=True)  # (n,1)
                c2 = (centers * centers).sum(dim=1).unsqueeze(0)  # (1,k)
                d2 = x2 + c2 - 2.0 * (xt @ centers.t())  # (n,k)
                new_labels = torch.argmin(d2, dim=1)
                if torch.equal(new_labels, labels):
                    break
                labels = new_labels
                for j in range(k):
                    mask = labels == j
                    if mask.any():
                        centers[j] = xt[mask].mean(dim=0)
                    else:
                        ridx = torch.randint(low=0, high=n, size=(1,), generator=g, device=device).item()
                        centers[j] = xt[ridx]
            x2 = (xt * xt).sum(dim=1, keepdim=True)
            c2 = (centers * centers).sum(dim=1).unsqueeze(0)
            d2 = x2 + c2 - 2.0 * (xt @ centers.t())
            inertia = float(d2.gather(1, labels.view(-1, 1)).sum().item())
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels_t = labels.clone()
        return best_labels_t.detach().cpu().numpy() if best_labels_t is not None else np.zeros((n,), dtype=int)

    # Prefer GPU torch KMeans; else sklearn; else numpy
    if _HAVE_TORCH and _CUDA_OK:
        labels = _torch_kmeans_gpu(X, k=3, n_init_=n_init, max_iter_=max_iter, seed_=seed, init_centers_=init_centers)
    elif _HAVE_SKLEARN:
        if init_centers is not None and init_centers.shape == (3, X.shape[1]):
            labels = KMeans(n_clusters=3, init=init_centers, n_init=1, max_iter=max_iter, tol=1e-4, random_state=seed).fit_predict(X)
        else:
            labels = KMeans(n_clusters=3, init="k-means++", n_init=n_init, max_iter=max_iter, tol=1e-4, random_state=seed).fit_predict(X)
    else:
        labels = _numpy_kmeans(X, k=3, n_init_=n_init, max_iter_=max_iter, seed_=seed, init_centers_=init_centers)

    # Cluster -> weather by median power
    medians = []
    for k in range(3):
        mask = labels == k
        med = float(np.nanmedian(power_series.loc[index[mask]].values)) if np.any(mask) else -np.inf
        medians.append((k, med))
    ordered = sorted(medians, key=lambda t: t[1], reverse=True)
    mapping = {ordered[0][0]: 0, ordered[1][0]: 1, ordered[2][0]: 2}
    weather_labels = np.array([mapping[int(c)] for c in labels], dtype=int)

    # silhouette (approx if sklearn missing)
    if _HAVE_SKLEARN and len(X) > 10:
        sil = float(silhouette_score(X, labels))
    else:
        overall_mean = X.mean(axis=0, keepdims=True)
        tss = float(((X - overall_mean) ** 2).sum())
        wcss = 0.0
        for j in range(3):
            pts = X[labels == j]
            if len(pts):
                wcss += float(((pts - pts.mean(axis=0, keepdims=True)) ** 2).sum())
        sil = float(max(0.0, 1.0 - wcss / max(tss, 1e-9)))

    counts = {"sunny": int(np.sum(weather_labels == 0)),
              "cloudy": int(np.sum(weather_labels == 1)),
              "overcast": int(np.sum(weather_labels == 2))}
    return weather_labels, {"cluster_to_weather": mapping, "counts": counts, "silhouette": sil}


def _gaussian_intersections(m1: float, s1: float, m2: float, s2: float) -> List[float]:
    s1 = float(max(s1, 1e-9))
    s2 = float(max(s2, 1e-9))
    a = (1.0 / (2 * s2 * s2)) - (1.0 / (2 * s1 * s1))
    b = (m1 / (s1 * s1)) - (m2 / (s2 * s2))
    c = (m2 * m2) / (2 * s2 * s2) - (m1 * m1) / (2 * s1 * s1) + np.log(s2 / s1)
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        return [(-c) / b]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    sqrt_disc = np.sqrt(disc)
    x1 = (-b + sqrt_disc) / (2 * a)
    x2 = (-b - sqrt_disc) / (2 * a)
    return [x1, x2]


def _kde_intersection(xa: np.ndarray, xb: np.ndarray, grid_count: int = 512) -> Optional[float]:
    if not _HAVE_SCIPY:
        return None
    if xa.size < 2 or xb.size < 2:
        return None
    lo = float(min(np.min(xa), np.min(xb)))
    hi = float(max(np.max(xa), np.max(xb)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    grid = np.linspace(lo, hi, grid_count)
    try:
        kde_a = gaussian_kde(xa)
        kde_b = gaussian_kde(xb)
        da = kde_a(grid)
        db = kde_b(grid)
        diff = da - db
        sign = np.sign(diff)
        zc = np.where(np.diff(sign) != 0)[0]
        if zc.size == 0:
            return None
        ma = float(np.mean(xa))
        mb = float(np.mean(xb))
        mid = 0.5 * (ma + mb)
        cand = [grid[i] for i in zc]
        best = min(cand, key=lambda x: abs(x - mid))
        return float(best)
    except Exception:
        return None


def _quantile_midpoint(xa: np.ndarray, xb: np.ndarray, lower_q: float, upper_q: float) -> float:
    if np.mean(xa) <= np.mean(xb):
        low_side = np.quantile(xa, upper_q)
        high_side = np.quantile(xb, lower_q)
    else:
        low_side = np.quantile(xb, upper_q)
        high_side = np.quantile(xa, lower_q)
    return float(0.5 * (low_side + high_side))


@dataclass
class ThresholdEstimate:
    value: float
    method: str
    band_p10: Optional[float] = None
    band_p90: Optional[float] = None


def estimate_pair_threshold(x_low: np.ndarray, x_high: np.ndarray, q_low: float, q_high: float) -> ThresholdEstimate:
    xa = np.asarray(x_low).astype(float)
    xb = np.asarray(x_high).astype(float)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if xa.size < 10 or xb.size < 10:
        t = _quantile_midpoint(xa, xb, q_low, q_high)
        return ThresholdEstimate(value=t, method="quantile")
    m1, s1 = float(np.mean(xa)), float(np.std(xa, ddof=1))
    m2, s2 = float(np.mean(xb)), float(np.std(xb, ddof=1))
    roots = _gaussian_intersections(m1, s1, m2, s2)
    if roots:
        mid = 0.5 * (m1 + m2)
        t = sorted(roots, key=lambda r: abs(r - mid))[0]
        if np.isfinite(t):
            return ThresholdEstimate(value=float(t), method="gaussian")
    t_kde = _kde_intersection(xa, xb, grid_count=1024)
    if t_kde is not None and np.isfinite(t_kde):
        return ThresholdEstimate(value=float(t_kde), method="kde")
    return ThresholdEstimate(value=_quantile_midpoint(xa, xb, q_low, q_high), method="quantile")


def bootstrap_band(
    x_low: np.ndarray,
    x_high: np.ndarray,
    base_method: str,
    n_bootstrap: int = 200,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> Tuple[float, float]:
    if n_bootstrap <= 0 or min(len(x_low), len(x_high)) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    ests: List[float] = []
    for _ in range(n_bootstrap):
        xa = rng.choice(x_low, size=len(x_low), replace=True)
        xb = rng.choice(x_high, size=len(x_high), replace=True)
        if base_method == "gaussian":
            m1, s1 = float(np.mean(xa)), float(np.std(xa, ddof=1))
            m2, s2 = float(np.mean(xb)), float(np.std(xb, ddof=1))
            roots = _gaussian_intersections(m1, s1, m2, s2)
            if roots:
                mid = 0.5 * (m1 + m2)
                ests.append(float(sorted(roots, key=lambda r: abs(r - mid))[0]))
                continue
        if base_method == "kde" and _HAVE_SCIPY:
            t = _kde_intersection(xa, xb, grid_count=512)
            if t is not None:
                ests.append(float(t))
                continue
        ests.append(_quantile_midpoint(xa, xb, q_low, q_high))
    if not ests:
        return (float("nan"), float("nan"))
    return float(np.quantile(ests, q_low)), float(np.quantile(ests, q_high))


def density_overlap(xa: np.ndarray, xb: np.ndarray) -> float:
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if xa.size < 2 or xb.size < 2:
        return float("nan")
    lo = float(min(np.min(xa), np.min(xb)))
    hi = float(max(np.max(xa), np.max(xb)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return float("nan")
    grid = np.linspace(lo, hi, 1024)
    if _HAVE_SCIPY:
        try:
            da = gaussian_kde(xa)(grid)
            db = gaussian_kde(xb)(grid)
        except Exception:
            # Fallback to Gaussian approx to avoid LinAlgError on tiny/degenerate samples
            def _g(x, m, s):
                s = max(s, 1e-9)
                return np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
            da = _g(grid, float(np.mean(xa)), float(np.std(xa, ddof=1)))
            db = _g(grid, float(np.mean(xb)), float(np.std(xb, ddof=1)))
    else:
        def _g(x, m, s):
            s = max(s, 1e-9)
            return np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
        da = _g(grid, float(np.mean(xa)), float(np.std(xa, ddof=1)))
        db = _g(grid, float(np.mean(xb)), float(np.std(xb, ddof=1)))
    # Overlap area under min(pdf_a, pdf_b); should fall in [0,1] when KDE ~normalized.
    area = np.trapz(np.minimum(da, db), grid)
    return float(np.clip(area, 0.0, 1.0))


def density_overlap_maxnorm(xa: np.ndarray, xb: np.ndarray) -> float:
    """
    Overlap normalized by max(total mass), i.e., area / max(int pdf_a, int pdf_b).
    Keeps result in [0,1] and is less sensitive to KDE mass drift.
    """
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if xa.size < 2 or xb.size < 2:
        return float("nan")
    lo = float(min(np.min(xa), np.min(xb)))
    hi = float(max(np.max(xa), np.max(xb)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return float("nan")
    grid = np.linspace(lo, hi, 1024)
    if _HAVE_SCIPY:
        try:
            da = gaussian_kde(xa)(grid)
            db = gaussian_kde(xb)(grid)
        except Exception:
            def _g(x, m, s):
                s = max(s, 1e-9)
                return np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
            da = _g(grid, float(np.mean(xa)), float(np.std(xa, ddof=1)))
            db = _g(grid, float(np.mean(xb)), float(np.std(xb, ddof=1)))
    else:
        def _g(x, m, s):
            s = max(s, 1e-9)
            return np.exp(-0.5 * ((x - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
        da = _g(grid, float(np.mean(xa)), float(np.std(xa, ddof=1)))
        db = _g(grid, float(np.mean(xb)), float(np.std(xb, ddof=1)))
    area = np.trapz(np.minimum(da, db), grid)
    norm = max(np.trapz(da, grid), np.trapz(db, grid))
    return float(np.clip(area / max(norm, 1e-9), 0.0, 1.0))

def classify_with_thresholds_ci(ci: np.ndarray, th_low: float, th_high: float) -> np.ndarray:
    labels = np.ones_like(ci, dtype=int)
    labels[ci >= th_high] = 0
    labels[ci <= th_low] = 2
    return labels


def classify_with_thresholds_wsi(wsi: np.ndarray, th_low: float, th_high: float) -> np.ndarray:
    labels = np.ones_like(wsi, dtype=int)
    labels[wsi < th_low] = 0
    labels[wsi > th_high] = 2
    return labels


def plot_diagnostics(output_dir: Path,
                     ci_vals: np.ndarray,
                     wsi_vals: np.ndarray,
                     weather_labels: np.ndarray,
                     power_vals: np.ndarray,
                     irradiance_vals: np.ndarray) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        logger.warning("matplotlib not available; skip plots.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {0: "#fdae61", 1: "#abd9e9", 2: "#2c7bb6"}
    # CI hist
    plt.figure(figsize=(8, 5))
    for c in [0, 1, 2]:
        vals = ci_vals[weather_labels == c]
        if vals.size:
            plt.hist(vals, bins=40, alpha=0.35, color=colors[c], label=WEATHER_NAME[c], density=True)
    plt.xlabel("CI")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ci_hist.png", dpi=160)
    plt.close()
    # WSI hist
    plt.figure(figsize=(8, 5))
    for c in [0, 1, 2]:
        vals = wsi_vals[weather_labels == c]
        if vals.size:
            plt.hist(vals, bins=40, alpha=0.35, color=colors[c], label=WEATHER_NAME[c], density=True)
    plt.xlabel("WSI")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wsi_hist.png", dpi=160)
    plt.close()
    # Scatter P vs I
    plt.figure(figsize=(6, 6))
    for c in [0, 1, 2]:
        mask = weather_labels == c
        if mask.any():
            plt.scatter(irradiance_vals[mask], power_vals[mask], s=6, alpha=0.4, c=colors[c], label=WEATHER_NAME[c])
    plt.xlabel("Irradiance (I*)")
    plt.ylabel("Power (P*)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_p_vs_i.png", dpi=160)
    plt.close()


def main():
    prog = OneLineProgress()
    
    # First pass: parse only --config to load defaults from config file
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    pre_args, _ = pre_parser.parse_known_args()
    
    # Load config to get clustering defaults
    try:
        config_for_defaults = load_yaml(Path(pre_args.config))
        fe_cfg = config_for_defaults.get("feature_engineering", {})
        cluster_cfg = fe_cfg.get("clustering", {})
    except Exception:
        cluster_cfg = {}
    
    # Extract defaults from config
    def_daytime_mode = cluster_cfg.get("daytime_mode", "ghi")
    def_daytime_ghi_min = float(cluster_cfg.get("daytime_ghi_min", 5.0))
    def_daytime_ge_min = float(cluster_cfg.get("daytime_ge_min", 20.0))
    def_features_basic_only = bool(cluster_cfg.get("features_basic_only", False))
    def_window_mins = int(cluster_cfg.get("window_mins", 60))
    def_min_samples_per_window = int(cluster_cfg.get("min_samples_per_window", 3))
    def_scaler = str(cluster_cfg.get("scaler", "robust"))
    def_use_pca = bool(cluster_cfg.get("use_pca", False))
    def_pca_n = int(cluster_cfg.get("pca_n", 10))
    def_feat_weight_tod = float(cluster_cfg.get("feat_weight_tod", 2.0))
    def_feat_weight_diffuse = float(cluster_cfg.get("feat_weight_diffuse", 2.0))
    def_feat_weight_z = float(cluster_cfg.get("feat_weight_z", 1.5))
    def_learn_prototypes = bool(cluster_cfg.get("learn_prototypes", False))
    def_proto_sunny = cluster_cfg.get("proto_sunny")
    def_proto_cloudy = cluster_cfg.get("proto_cloudy")
    def_proto_overcast = cluster_cfg.get("proto_overcast")
    def_prototypes_min_count = int(cluster_cfg.get("prototypes_min_count", 30))
    def_proto_tol_mins = int(cluster_cfg.get("prototypes_tol_mins", 8))
    
    # Main parser with config-based defaults
    parser = argparse.ArgumentParser(description="Initialize CI/WSI thresholds (GPU-accelerated where possible)")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    # Features/Clustering (defaults from config.feature_engineering.clustering)
    parser.add_argument("--window-mins", type=int, default=def_window_mins)
    parser.add_argument("--min-samples-per-window", type=int, default=def_min_samples_per_window)
    parser.add_argument("--scaler", type=str, default=def_scaler, choices=["robust", "standard"])
    parser.add_argument("--use-pca", action="store_true", default=def_use_pca)
    parser.add_argument("--pca-n", type=int, default=def_pca_n)
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--kmeans-anchors", action="store_true",
                        help="Enable semi-supervised KMeans initialization from physically-motivated anchors")
    parser.add_argument("--anchor-midday-half-min", type=int, default=90,
                        help="Half window (minutes) around 12:00 to select midday anchors")
    parser.add_argument("--anchor-q-high", type=float, default=0.70,
                        help="High quantile for sunny-like thresholds (0-1)")
    parser.add_argument("--anchor-q-low", type=float, default=0.30,
                        help="Low quantile for overcast-like thresholds (0-1)")
    # Prototype learning from user-provided typical days (defaults from config)
    parser.add_argument("--learn-prototypes", action="store_true", default=def_learn_prototypes,
                        help="Learn KMeans initial centers (prototypes) from user-provided typical-day files")
    parser.add_argument("--proto-sunny", type=str, default=def_proto_sunny, help="Excel/CSV file containing timestamps for SUNNY samples")
    parser.add_argument("--proto-cloudy", type=str, default=def_proto_cloudy, help="Excel/CSV file containing timestamps for CLOUDY samples")
    parser.add_argument("--proto-overcast", type=str, default=def_proto_overcast, help="Excel/CSV file containing timestamps for OVERCAST samples")
    parser.add_argument("--prototypes-out", type=str, default=None, help="Path to save learned prototypes profile (YAML)")
    parser.add_argument("--use-prototypes", type=str, default=None, help="Path to load prototypes profile (YAML) and use as KMeans init")
    parser.add_argument("--proto-tolerance-mins", type=int, default=def_proto_tol_mins,
                        help="Tolerance (minutes) for aligning typical timestamps to dataset index")
    parser.add_argument("--prototypes-min-count", type=int, default=def_prototypes_min_count,
                        help="Minimum samples required per class to compute a reliable prototype")
    # Teacher days: user-provided typical days to derive initial centers directly
    parser.add_argument("--teach-sunny", type=str, default=None,
                        help="Comma-separated dates or ranges for typical sunny days, e.g. 2019-06-01,2019-06-15:2019-06-20")
    parser.add_argument("--teach-cloudy", type=str, default=None,
                        help="Comma-separated dates/ranges for typical cloudy days")
    parser.add_argument("--teach-over", type=str, default=None,
                        help="Comma-separated dates/ranges for typical overcast days")
    parser.add_argument("--teach-min-samples", type=int, default=50,
                        help="Minimum samples per taught class to build centers; else fallback to anchors/vanilla")
    # Feature weighting (defaults from config)
    parser.add_argument("--features-basic-only", action="store_true", default=def_features_basic_only,
                        help="Use only basic fields for clustering (Time-of-day, power, irradiance, dni(if any), irradiance_total(if any), temperature, pressure, humidity). Disable rolling/z-scores.")
    parser.add_argument("--feat-weight-tod", type=float, default=def_feat_weight_tod,
                        help="Weight for time-of-day features (tod_sin/tod_cos) after scaling")
    parser.add_argument("--feat-weight-diffuse", type=float, default=def_feat_weight_diffuse,
                        help="Weight for diffuse-related feature (dni_over_ghi) after scaling")
    parser.add_argument("--feat-weight-z", type=float, default=def_feat_weight_z,
                        help="Weight for robust z-score features (z_*) after scaling")
    # GMM diagnostics (optional)
    parser.add_argument("--gmm-enable", action="store_true")
    parser.add_argument("--gmm-cov-type", type=str, default="auto", choices=["auto", "full", "diag", "tied", "spherical"])
    parser.add_argument("--gmm-max-iter", type=int, default=500)
    # Daytime mask override (defaults from config)
    parser.add_argument("--daytime-ge-min", type=float, default=def_daytime_ge_min,
                        help="Override feature_engineering.daytime.ge_min_wm2; e.g., 0 keeps all as daytime")
    parser.add_argument("--daytime-mode", type=str, default=def_daytime_mode, choices=["ge", "ghi", "or", "and"],
                        help="Daytime mask mode: ge|ghi|or|and; default from config.clustering.daytime_mode (recommend 'ghi')")
    parser.add_argument("--daytime-ghi-min", type=float, default=def_daytime_ghi_min,
                        help="Override daytime GHI threshold (W/m^2); default from config.clustering.daytime_ghi_min")
    # Thresholds/Bands
    parser.add_argument("--gray-width-ci", type=float, default=0.02)
    parser.add_argument("--gray-width-wsi", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--agreement-min", type=float, default=0.85, help="Minimum CI/WSI agreement required to mark status=ok")
    parser.add_argument("--overlap-mode", type=str, default="raw", choices=["raw", "maxnorm"],
                        help="Overlap metric mode: raw=area under min(pdf), maxnorm=normalize by max integral")
    parser.add_argument("--write-config", action="store_true",
                        help="Update feature_engineering.{ci, wsi}_thresholds in the provided --config")
    parser.add_argument("--band-q-low", type=float, default=0.10,
                        help="Lower quantile used for band estimation (0 < q < 0.5)")
    parser.add_argument("--band-q-high", type=float, default=0.90,
                        help="Upper quantile used for band estimation (0.5 < q < 1)")
    parser.add_argument("--min-class-ratio", type=float, default=0.05,
                        help="Minimum required ratio for each weather class; below this will mark result as imbalanced")
    args = parser.parse_args()

    np.random.seed(args.seed)

    prog.start("Load config")
    config = load_yaml(Path(args.config))
    raw_dir, processed_dir, splits_dir = resolve_data_paths(config)
    prog.update(100, "paths")
    prog.done(True)
    prog.start("Prepare splits")
    train_df, val_df, test_df = read_or_prepare_splits(
        config,
        raw_dir,
        processed_dir,
        splits_dir,
        config_path=Path(args.config),
    )
    prog.update(100, "ok")
    prog.done(True)
    split_df = {"train": train_df, "val": val_df, "test": test_df}[args.split]
    prog.start("Slice")
    split_df = slice_split(split_df, args.start_date, args.end_date)
    if split_df.empty:
        raise ValueError("所选切片为空，请检查 --split 与日期范围")

    prog.update(100, "ok")
    prog.done(True)
    fe_cfg = config.get("feature_engineering", {})
    day_cfg = fe_cfg.get("daytime", {})
    # Use CLI args directly (defaults already loaded from config in argparse)
    used_ge_min = max(0.0, float(args.daytime_ge_min))
    used_day_mode = str(args.daytime_mode).lower()
    used_ghi_min = float(args.daytime_ghi_min)
    if used_day_mode not in {"ge", "ghi", "or", "and"}:
        used_day_mode = "ghi"
    # Reflect used values for logs
    try:
        day_cfg["ge_min_wm2"] = used_ge_min
        day_cfg["mode"] = used_day_mode
        day_cfg["ghi_min_wm2"] = used_ghi_min
    except Exception:
        pass
    # Location settings: sync with main.py defaults unless config provides overrides.
    # main.py does not pass location_* explicitly, so WeatherClassifier defaults are:
    # lat=38.5, lon=105.0, elevation=1500. To keep behavior in sync, we read optional
    # overrides from config.data.location and fall back to those defaults.
    loc_cfg = config.get("data", {}).get("location", {}) or {}
    loc_lat = float(loc_cfg.get("lat", 38.5))
    loc_lon = float(loc_cfg.get("lon", 105.0))
    loc_elev = float(loc_cfg.get("elevation", 1500.0))
    loc_tz = float(loc_cfg.get("time_zone_hours", 8.0))  # 中国标准时间默认 UTC+8
    clf = WeatherClassifier(
        location_lat=loc_lat,
        location_lon=loc_lon,
        elevation=loc_elev,
        time_zone_hours=loc_tz,
        ci_thresholds=fe_cfg.get("ci_thresholds", [0.2, 0.6]),
        wsi_thresholds=fe_cfg.get("wsi_thresholds", [0.3, 0.7]),
        fusion_weights=fe_cfg.get("fusion_weights", {"ci": 0.7, "wsi": 0.3}),
        daytime_ge_min=used_ge_min,
        daytime_mode=used_day_mode,
        daytime_ghi_min=used_ghi_min,
        night_handling=day_cfg.get("night_handling", "exclude"),
    )
    prog.start("Day mask")
    ge_mask = clf.calculate_ci(split_df["irradiance"].values, split_df.index)[1]
    day_mask = np.asarray(ge_mask, dtype=bool)
    df_day = split_df.loc[day_mask]
    if df_day.empty:
        raise ValueError("白天样本为空，请检查 feature_engineering.daytime 配置（mode/ge_min_wm2/ghi_min_wm2）与 --daytime-* 覆写参数")

    prog.update(100, "ok")
    prog.done(True)
    try:
        if used_day_mode == "ge":
            logger.info("白天样本筛选: split=%d, day=%d (mask=GE≥%.2f W/m²)", int(len(split_df)), int(len(df_day)), float(used_ge_min))
        elif used_day_mode == "ghi":
            logger.info("白天样本筛选: split=%d, day=%d (mask=GHI≥%.2f W/m²)", int(len(split_df)), int(len(df_day)), float(used_ghi_min))
        elif used_day_mode == "or":
            logger.info("白天样本筛选: split=%d, day=%d (mask=GE≥%.2f OR GHI≥%.2f W/m²)", int(len(split_df)), int(len(df_day)), float(used_ge_min), float(used_ghi_min))
        else:
            logger.info("白天样本筛选: split=%d, day=%d (mask=GE≥%.2f AND GHI≥%.2f W/m²)", int(len(split_df)), int(len(df_day)), float(used_ge_min), float(used_ghi_min))
    except Exception:
        pass

    prog.start("Features")
    freq_minutes = int(config.get("data", {}).get("frequency_minutes", 15))
    X, meta = engineer_features(
        df_day,
        window_mins=int(args.window_mins),
        min_samples_per_window=int(args.min_samples_per_window),
        scaler_name=args.scaler,
        use_pca=bool(args.use_pca),
        pca_n=int(max(args.pca_n, 1)),
        freq_minutes=freq_minutes,
        clf=clf,
        feat_weight_tod=float(args.feat_weight_tod),
        feat_weight_diffuse=float(args.feat_weight_diffuse),
        feat_weight_z=float(args.feat_weight_z),
        features_basic_only=bool(args.features_basic_only),
        return_transforms=True,
    )
    prog.update(100, "ok")
    prog.done(True)
    idx = meta["index"]
    prog.start("Clustering")
    # Build KMeans initial centers
    init_centers = None

    def _parse_date_spec(spec: Optional[str]) -> Optional[set]:
        if not spec:
            return None
        parts = [p.strip() for p in str(spec).split(",") if p.strip()]
        dates = set()
        for p in parts:
            if ":" in p:
                a, b = p.split(":", 1)
                try:
                    rng = pd.date_range(pd.to_datetime(a).normalize(), pd.to_datetime(b).normalize(), freq="D")
                    dates.update({d.date() for d in rng})
                except Exception:
                    continue
            else:
                try:
                    dates.add(pd.to_datetime(p).normalize().date())
                except Exception:
                    continue
        return dates or None

    # 1) Teacher centers (highest priority): user-provided typical days
    teach_sets = {
        "sunny": _parse_date_spec(getattr(args, "teach_sunny", None)),
        "cloudy": _parse_date_spec(getattr(args, "teach_cloudy", None)),
        "over": _parse_date_spec(getattr(args, "teach_over", None)),
    }
    if any(v for v in teach_sets.values()):
        try:
            idx_dates = pd.to_datetime(idx).date
            def pick(mask_dates: Optional[set]) -> np.ndarray:
                if not mask_dates:
                    return np.empty((0, X.shape[1]), dtype=float)
                sel = np.array([d in mask_dates for d in idx_dates], dtype=bool)
                return X[sel]
            xs = pick(teach_sets["sunny"])
            xc = pick(teach_sets["cloudy"])
            xo = pick(teach_sets["over"])
            cs = xs.mean(axis=0) if xs.shape[0] >= int(args.teach_min_samples) else None
            cc = xc.mean(axis=0) if xc.shape[0] >= int(args.teach_min_samples) else None
            co = xo.mean(axis=0) if xo.shape[0] >= int(args.teach_min_samples) else None
            if all(c is not None for c in (cs, cc, co)):
                init_centers = np.vstack([cs, cc, co]).astype(float)
                logger.info("KMeans teacher centers from typical days: sunny=%d, cloudy=%d, overcast=%d",
                            xs.shape[0], xc.shape[0], xo.shape[0])
            else:
                logger.info("KMeans teacher centers skipped (counts: sunny=%d, cloudy=%d, over=%d; need >= %d each)",
                            xs.shape[0], xc.shape[0], xo.shape[0], int(args.teach_min_samples))
        except Exception as e:
            logger.warning("Build teacher centers failed: %s", e)
            init_centers = None

    # 2) Prototypes learned from timestamp lists in external files（次优先级）
    if init_centers is None and (getattr(args, "use_prototypes", None) or getattr(args, "learn_prototypes", False)):
        try:
            centers = None
            profile = None
            # Try loading first
            if getattr(args, "use_prototypes", None):
                try:
                    profile = yaml.safe_load(Path(args.use_prototypes).read_text(encoding="utf-8")) or {}
                    centers = np.asarray(profile.get("prototypes", []), dtype=float)
                    if centers.shape != (3, X.shape[1]):
                        logger.warning("use-prototypes: dimension mismatch %s vs (3,%d)", centers.shape, X.shape[1])
                        centers = None
                except Exception as e:
                    logger.warning("use-prototypes failed: %s", e)
            # Learn from Excel/CSV timestamp lists if requested or loading failed
            if centers is None and bool(getattr(args, "learn_prototypes", False)):
                def _read_times_any(path: Optional[str]) -> Optional[pd.DatetimeIndex]:
                    if not path:
                        return None
                    p = Path(path)
                    if not p.exists():
                        logger.warning("Typical file not found: %s", path)
                        return None
                    try:
                        dfp = pd.read_excel(p) if p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(p)
                        # find a time-like column
                        target = None
                        for name in ["timestamp", "time", "datetime", "Time(year-month-day h:m:s)", "Time (year-month-day h:m:s)"]:
                            if name in dfp.columns:
                                target = name
                                break
                        if target is None:
                            for c in dfp.columns:
                                if isinstance(c, str) and "time" in c.lower():
                                    target = c; break
                        if target is None:
                            logger.warning("No time column in %s", str(p))
                            return None
                        ts = pd.to_datetime(dfp[target], errors="coerce")
                        ts = ts[pd.notna(ts)]
                        return pd.DatetimeIndex(ts.sort_values().unique())
                    except Exception as e:
                        logger.warning("Read %s failed: %s", str(p), e)
                        return None
                def _align_to_index(tidx: Optional[pd.DatetimeIndex], ref: pd.DatetimeIndex, tol_min: int) -> np.ndarray:
                    if tidx is None or len(tidx) == 0:
                        return np.zeros(len(ref), dtype=bool)
                    df_ref = pd.DataFrame({"ref": ref.values})
                    df_ref["ref"] = pd.to_datetime(df_ref["ref"])
                    df_t = pd.DataFrame({"ts": tidx.values})
                    df_t["ts"] = pd.to_datetime(df_t["ts"])
                    df_ref = df_ref.sort_values("ref")
                    df_t = df_t.sort_values("ts")
                    m = pd.merge_asof(df_t, df_ref, left_on="ts", right_on="ref", direction="nearest",
                                      tolerance=pd.Timedelta(minutes=int(args.proto_tolerance_mins)))
                    matched = m["ref"].dropna().unique()
                    if matched.size == 0:
                        return np.zeros(len(ref), dtype=bool)
                    pos = pd.Index(ref).get_indexer(matched)
                    mask = np.zeros(len(ref), dtype=bool)
                    mask[pos[pos >= 0]] = True
                    return mask
                ts_s = _read_times_any(getattr(args, "proto_sunny", None))
                ts_c = _read_times_any(getattr(args, "proto_cloudy", None))
                ts_o = _read_times_any(getattr(args, "proto_overcast", None))
                idx_ref = meta["index"]
                ms = _align_to_index(ts_s, idx_ref, int(args.proto_tolerance_mins))
                mc = _align_to_index(ts_c, idx_ref, int(args.proto_tolerance_mins))
                mo = _align_to_index(ts_o, idx_ref, int(args.proto_tolerance_mins))
                ns, nc, no = int(ms.sum()), int(mc.sum()), int(mo.sum())
                logger.info("Prototypes (files) matches: sunny=%d, cloudy=%d, overcast=%d (tol=%dmin)",
                            ns, nc, no, int(args.proto_tolerance_mins))
                def _robust_center(A: np.ndarray) -> Optional[np.ndarray]:
                    if A.shape[0] == 0:
                        return None
                    med = np.nanmedian(A, axis=0)
                    if np.any(np.isnan(med)):
                        med = np.nanmean(A, axis=0)
                    return med.astype(float)
                cs = _robust_center(X[ms]) if ns >= int(args.prototypes_min_count) else None
                cc = _robust_center(X[mc]) if nc >= int(args.prototypes_min_count) else None
                co = _robust_center(X[mo]) if no >= int(args.prototypes_min_count) else None
                if all(v is not None for v in (cs, cc, co)):
                    centers = np.vstack([cs, cc, co])
                    profile = {
                        "prototypes": centers.tolist(),
                        "feature_cols": meta.get("feature_cols", []),
                        "weights": meta.get("weights", None),
                        "pca_used": bool(args.use_pca),
                        "scaler": args.scaler,
                        "counts": {"sunny": ns, "cloudy": nc, "overcast": no},
                        "tolerance_mins": int(args.proto_tolerance_mins),
                    }
                    if getattr(args, "prototypes_out", None):
                        try:
                            outp = Path(args.prototypes_out)
                            outp.parent.mkdir(parents=True, exist_ok=True)
                            dump_yaml(profile, outp)
                            logger.info("Saved prototypes to %s", str(outp))
                        except Exception as e:
                            logger.warning("Save prototypes failed: %s", e)
                else:
                    logger.info("Learning prototypes skipped (insufficient per-class matches; need >= %d)", int(args.prototypes_min_count))
            if centers is not None:
                init_centers = centers
        except Exception as e:
            logger.warning("Prototypes processing failed: %s", e)

    # 3) Optional semi-supervised anchors for KMeans init（再次回退）
    if bool(getattr(args, "kmeans_anchors", False)):
        try:
            sub_for_anchors = df_day.loc[meta["index"]]
            # Require DNI for diffuse separation; else skip anchors
            if "dni" in sub_for_anchors.columns and sub_for_anchors["dni"].notna().any():
                # Midday band
                mins = (sub_for_anchors.index.hour * 60 + sub_for_anchors.index.minute + (sub_for_anchors.index.second / 60.0)).astype(float)
                midday_mask = np.abs(mins - 12 * 60) <= float(args.anchor_midday_half_min)
                mid = sub_for_anchors.loc[midday_mask]
                if not mid.empty:
                    ghi = mid["irradiance"].astype(float).values
                    dni = mid["dni"].astype(float).fillna(0.0).values
                    hum = mid["humidity"].astype(float).values
                    tmp = mid["temperature"].astype(float).values
                    prs = mid["pressure"].astype(float).values
                    # 计算 BF = (DNI*cosZ)/GHI 作为直射占比
                    ge_mid = np.asarray(clf._calculate_extraterrestrial_radiation(mid.index), dtype=float)
                    doy_mid = mid.index.dayofyear.values.astype(float)
                    gamma_mid = 2 * np.pi * (doy_mid - 1.0) / 365.0
                    e0_mid = (1.000110 +
                              0.034221 * np.cos(gamma_mid) +
                              0.001280 * np.sin(gamma_mid) +
                              0.000719 * np.cos(2 * gamma_mid) +
                              0.000077 * np.sin(2 * gamma_mid))
                    i0 = float(getattr(clf, "solar_constant", 1367.0))
                    cosZ_mid = np.clip(ge_mid / (i0 * e0_mid), 0.0, 1.0)
                    ratio = ((dni * cosZ_mid) / np.maximum(ghi, 1e-6)).clip(0.0, 1.5)
                    qh = float(args.anchor_q_high)
                    ql = float(args.anchor_q_low)
                    # thresholds
                    t_ghi_hi = np.nanquantile(ghi, qh)
                    t_ghi_lo = np.nanquantile(ghi, ql)
                    t_ratio_hi = np.nanquantile(ratio, qh)
                    t_ratio_lo = np.nanquantile(ratio, ql)
                    t_hum_lo = np.nanquantile(hum, ql)
                    t_tmp_hi = np.nanquantile(tmp, qh)
                    t_prs_hi = np.nanquantile(prs, qh)
                    # boolean masks within 'mid'
                    sunny_m = (ghi >= t_ghi_hi) & (ratio >= t_ratio_hi) & (hum <= t_hum_lo) & (tmp >= t_tmp_hi) & (prs >= t_prs_hi)
                    cloudy_m = (ghi >= t_ghi_hi) & (ratio <= t_ratio_lo)
                    over_m = (ghi <= t_ghi_lo) & (ratio <= t_ratio_lo) & (hum >= np.nanquantile(hum, 1 - ql))
                    # align to X rows
                    mid_index = mid.index
                    m_sunny = np.isin(meta["index"], mid_index[sunny_m])
                    m_cloudy = np.isin(meta["index"], mid_index[cloudy_m])
                    m_over = np.isin(meta["index"], mid_index[over_m])
                    # require minimal counts
                    def _center(m):
                        arr = X[m]
                        return arr.mean(axis=0) if arr.shape[0] > 10 else None
                    c_s, c_c, c_o = _center(m_sunny), _center(m_cloudy), _center(m_over)
                    if c_s is not None and c_c is not None and c_o is not None and init_centers is None:
                        init_centers = np.vstack([c_s, c_c, c_o]).astype(float)
                        logger.info("KMeans anchors: sunny=%d, cloudy=%d, overcast=%d",
                                    int(m_sunny.sum()), int(m_cloudy.sum()), int(m_over.sum()))
                    else:
                        logger.info("KMeans anchors skipped (insufficient anchor samples)")
            else:
                logger.info("KMeans anchors skipped (DNI not available)")
        except Exception as e:
            logger.warning("KMeans anchors build failed: %s", e)
            init_centers = None

    labels, cluster_info = kmeans_cluster(
        X,
        power_series=df_day["power"],
        index=idx,
        seed=int(args.seed),
        n_init=1 if init_centers is not None else int(args.kmeans_n_init),
        max_iter=int(args.kmeans_max_iter),
        init_centers=init_centers,
    )
    prog.update(100, "ok")
    prog.done(True)

    # 中文聚类统计打印（仅白天样本）
    try:
        cn = {0: "晴", 1: "多云", 2: "阴"}
        total_day = int(len(df_day))
        effective = int(len(idx))  # 进入聚类的有效样本（窗口对齐后）
        cnt = cluster_info.get("counts", {}) if isinstance(cluster_info, dict) else {}
        sunny_n = int(cnt.get("sunny", 0))
        cloudy_n = int(cnt.get("cloudy", 0))
        over_n = int(cnt.get("overcast", 0))
        den = max(effective, 1)
        logger.info("聚类结果统计（仅白天）：")
        logger.info(" - 白天样本总数: %d", total_day)
        logger.info(" - 进入聚类的有效样本数: %d", effective)
        logger.info(
            " - 各天气样本数: 晴=%d (%.1f%%), 多云=%d (%.1f%%), 阴=%d (%.1f%%)",
            sunny_n, 100.0 * sunny_n / den,
            cloudy_n, 100.0 * cloudy_n / den,
            over_n,  100.0 * over_n  / den,
        )
        sil = cluster_info.get("silhouette", float("nan"))
        logger.info(" - 轮廓系数(silhouette): %s", f"{sil:.3f}" if np.isfinite(sil) else "NaN")
        mp = cluster_info.get("cluster_to_weather", {})
        for k, w in mp.items():
            logger.info(" - KMeans簇 %d → 天气: %s", int(k), cn.get(int(w), str(w)))
    except Exception:
        # 打印统计不影响主流程
        pass

    # Optional: GMM diagnostics (soft validation)
    gmm_agreement = None
    gmm_block = None
    if args.gmm_enable:
        prog.start("GMM")
        if not _HAVE_SKLEARN:
            logger.warning("scikit-learn not available; skip GMM diagnostics.")
            prog.done(False, "skipped")
        else:
            # Initialize means from KMeans partition
            means_init = []
            for k in range(3):
                pts = X[labels == k]
                means_init.append(np.mean(pts, axis=0) if len(pts) else np.mean(X, axis=0))
            means_init = np.vstack(means_init)
            rng_seed = int(args.seed)
            cov_choice = args.gmm_cov_type
            bic_scores = None
            if cov_choice == "auto":
                best_bic = np.inf
                best_ct = "full"
                scores = {}
                for ct in ["full", "diag"]:
                    gm_try = GaussianMixture(n_components=3, covariance_type=ct, max_iter=int(args.gmm_max_iter),
                                             random_state=rng_seed, means_init=means_init)
                    gm_try.fit(X)
                    scores[ct] = float(gm_try.bic(X))
                    if scores[ct] < best_bic:
                        best_bic = scores[ct]
                        best_ct = ct
                cov_choice = best_ct
                bic_scores = scores
            gm = GaussianMixture(n_components=3, covariance_type=cov_choice, max_iter=int(args.gmm_max_iter),
                                 random_state=rng_seed, means_init=means_init)
            gm.fit(X)
            comp = gm.predict(X)
            # Map components -> weather by median power
            pairs = []
            for c in range(3):
                mask = comp == c
                med = float(np.nanmedian(df_day.loc[idx[mask], "power"].values)) if np.any(mask) else -np.inf
                pairs.append((c, med))
            ordered = sorted(pairs, key=lambda t: t[1], reverse=True)
            mapping = {ordered[0][0]: 0, ordered[1][0]: 1, ordered[2][0]: 2}
            gmm_weather = np.array([mapping[int(c)] for c in comp], dtype=int)
            gmm_agreement = float(np.mean(gmm_weather == labels))
            gmm_block = {
                "covariance_type": cov_choice,
                "bic": bic_scores,
                "converged": bool(gm.converged_),
                "weights": gm.weights_.tolist(),
                "means_shape": list(gm.means_.shape),
                "agreement_with_kmeans": gmm_agreement,
            }
            prog.update(100, f"agree={gmm_agreement:.3f}")
            prog.done(True)

    prog.start("CI/WSI")
    sub = df_day.loc[idx]
    ci_vals, _ = clf.calculate_ci(sub["irradiance"].values, sub.index)
    wsi_vals = clf.calculate_wsi(
        sub["pressure"].values,
        sub["humidity"].values,
        sub["temperature"].values,
        time_delta=freq_minutes
    )
    ci_vals = np.asarray(ci_vals, dtype=float)
    wsi_vals = np.asarray(wsi_vals, dtype=float)
    prog.update(100, "ok")
    prog.done(True)

    cls_mask = {i: (labels == i) for i in [0, 1, 2]}  # 0 sunny, 1 cloudy, 2 overcast
    ci_sunny, ci_cloudy, ci_over = ci_vals[cls_mask[0]], ci_vals[cls_mask[1]], ci_vals[cls_mask[2]]
    wsi_sunny, wsi_cloudy, wsi_over = wsi_vals[cls_mask[0]], wsi_vals[cls_mask[1]], wsi_vals[cls_mask[2]]

    # CI thresholds: lower (cloudy-overcast), upper (sunny-cloudy)
    prog.start("Thresholds")
    # Validate band quantiles
    ql = float(args.band_q_low)
    qh = float(args.band_q_high)
    if not (0.0 < ql < 0.5 and 0.5 < qh < 1.0 and ql < qh):
        logger.warning("非法 band 分位点，回退为 0.10/0.90: q_low=%.3f, q_high=%.3f", ql, qh)
        ql, qh = 0.10, 0.90

    ci_low_est = estimate_pair_threshold(ci_over, ci_cloudy, ql, qh)
    prog.update(20, "ci low")
    ci_high_est = estimate_pair_threshold(ci_cloudy, ci_sunny, ql, qh)
    prog.update(35, "ci high")
    ci_p10_low, ci_p90_low = bootstrap_band(
        ci_over, ci_cloudy, base_method=ci_low_est.method, n_bootstrap=int(args.bootstrap), q_low=ql, q_high=qh
    )
    prog.update(55, "ci band low")
    ci_p10_high, ci_p90_high = bootstrap_band(
        ci_cloudy, ci_sunny, base_method=ci_high_est.method, n_bootstrap=int(args.bootstrap), q_low=ql, q_high=qh
    )
    prog.update(70, "ci band high")
    # WSI thresholds: lower (sunny-cloudy), upper (cloudy-overcast)
    wsi_low_est = estimate_pair_threshold(wsi_sunny, wsi_cloudy, ql, qh)
    prog.update(80, "wsi low")
    wsi_high_est = estimate_pair_threshold(wsi_cloudy, wsi_over, ql, qh)
    prog.update(85, "wsi high")
    wsi_p10_low, wsi_p90_low = bootstrap_band(
        wsi_sunny, wsi_cloudy, base_method=wsi_low_est.method, n_bootstrap=int(args.bootstrap), q_low=ql, q_high=qh
    )
    prog.update(92, "wsi band low")
    wsi_p10_high, wsi_p90_high = bootstrap_band(
        wsi_cloudy, wsi_over, base_method=wsi_high_est.method, n_bootstrap=int(args.bootstrap), q_low=ql, q_high=qh
    )
    prog.update(98, "wsi band high")

    ci_pred = classify_with_thresholds_ci(ci_vals, th_low=ci_low_est.value, th_high=ci_high_est.value)
    wsi_pred = classify_with_thresholds_wsi(wsi_vals, th_low=wsi_low_est.value, th_high=wsi_high_est.value)
    ci_agree = float(np.mean(ci_pred == labels)) if ci_pred.size else float("nan")
    wsi_agree = float(np.mean(wsi_pred == labels)) if wsi_pred.size else float("nan")
    agreement = float(np.nanmean([ci_agree, wsi_agree]))
    prog.update(100, f"agree={agreement:.3f}")
    prog.done(True)

    prog.start("Diagnostics")
    if args.overlap_mode == "maxnorm":
        ci_overlap = float(np.nanmean([
            density_overlap_maxnorm(ci_cloudy, ci_over),
            density_overlap_maxnorm(ci_sunny, ci_cloudy),
        ]))
        wsi_overlap = float(np.nanmean([
            density_overlap_maxnorm(wsi_sunny, wsi_cloudy),
            density_overlap_maxnorm(wsi_cloudy, wsi_over),
        ]))
    else:
        ci_overlap = float(np.nanmean([
            density_overlap(ci_cloudy, ci_over),
            density_overlap(ci_sunny, ci_cloudy),
        ]))
        wsi_overlap = float(np.nanmean([
            density_overlap(wsi_sunny, wsi_cloudy),
            density_overlap(wsi_cloudy, wsi_over),
        ]))

    gray_ci = float(args.gray_width_ci)
    gray_wsi = float(args.gray_width_wsi)
    gray_ratio = float(
        np.mean((np.abs(ci_vals - ci_low_est.value) <= gray_ci) |
                (np.abs(ci_vals - ci_high_est.value) <= gray_ci) |
                (np.abs(wsi_vals - wsi_low_est.value) <= gray_wsi) |
                (np.abs(wsi_vals - wsi_high_est.value) <= gray_wsi))
    )
    prog.update(100, "ok")
    prog.done(True)

    prog.start("Write outputs")
    # Class imbalance guard
    class_counts = {
        "sunny": int(np.sum(labels == 0)),
        "cloudy": int(np.sum(labels == 1)),
        "overcast": int(np.sum(labels == 2)),
    }
    eff = max(int(len(idx)), 1)
    class_ratios = {k: (v / eff) for k, v in class_counts.items()}
    min_ratio = float(min(class_ratios.values())) if class_ratios else float("nan")
    imbalance = bool(np.isfinite(min_ratio) and (min_ratio < float(args.min_class_ratio)))

    thresholds_yaml = {
        "thresholds": {
            "ci": {
                "clear_partly": float(ci_high_est.value),
                "clear_partly_band": {
                    "lower": float(ci_p10_high) if np.isfinite(ci_p10_high) else None,
                    "upper": float(ci_p90_high) if np.isfinite(ci_p90_high) else None,
                },
                "partly_over": float(ci_low_est.value),
                "partly_over_band": {
                    "lower": float(ci_p10_low) if np.isfinite(ci_p10_low) else None,
                    "upper": float(ci_p90_low) if np.isfinite(ci_p90_low) else None,
                },
                "method": "gaussian" if "gaussian" in {ci_low_est.method, ci_high_est.method} else ("kde" if "kde" in {ci_low_est.method, ci_high_est.method} else "quantile"),
                "gray_width": gray_ci,
            },
            "wsi": {
                "clear_partly": float(wsi_low_est.value),
                "clear_partly_band": {
                    "lower": float(wsi_p10_low) if np.isfinite(wsi_p10_low) else None,
                    "upper": float(wsi_p90_low) if np.isfinite(wsi_p90_low) else None,
                },
                "partly_over": float(wsi_high_est.value),
                "partly_over_band": {
                    "lower": float(wsi_p10_high) if np.isfinite(wsi_p10_high) else None,
                    "upper": float(wsi_p90_high) if np.isfinite(wsi_p90_high) else None,
                },
                "method": "gaussian" if "gaussian" in {wsi_low_est.method, wsi_high_est.method} else ("kde" if "kde" in {wsi_low_est.method, wsi_high_est.method} else "quantile"),
                "gray_width": gray_wsi,
            },
        },
        "diagnostics": {
            "silhouette": float(cluster_info.get("silhouette", float("nan"))),
            "agreement": float(agreement),
            "ci_overlap": float(ci_overlap),
            "wsi_overlap": float(wsi_overlap),
            "overlap_mode": str(args.overlap_mode),
            "gray_ratio": float(gray_ratio),
            "gmm_agreement": (float(gmm_agreement) if gmm_agreement is not None else None),
            "class_counts": class_counts,
            "class_ratios": {k: float(v) for k, v in class_ratios.items()},
            "min_class_ratio": float(min_ratio),
            "min_class_ratio_threshold": float(args.min_class_ratio),
        },
        "status": "ok" if (np.isfinite(agreement) and agreement >= float(args.agreement_min) and not imbalance) else "fallback",
    }

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = DEFAULT_OUTPUT_ROOT / f"init_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(thresholds_yaml, out_dir / "thresholds.yaml")
    diag = {
        "cluster_info": cluster_info,
        "ci": {
            "low_boundary": ci_low_est.__dict__,
            "high_boundary": ci_high_est.__dict__,
        },
        "wsi": {
            "low_boundary": wsi_low_est.__dict__,
            "high_boundary": wsi_high_est.__dict__,
        },
        "gmm": gmm_block,
        "notes": "Use thresholds.ci.* and thresholds.wsi.* to populate feature_engineering in config.yaml.",
    }
    dump_json(diag, out_dir / "diagnostics.json")
    prog.update(100, "yaml/json")
    prog.done(True)

    # Export labels/timestamps for downstream reuse
    try:
        timestamps = pd.to_datetime(idx)
        label_array = np.asarray(labels, dtype=np.int8)
        np.save(out_dir / "cluster_labels.npy", label_array)
        pd.DataFrame({"timestamp": timestamps}).to_parquet(out_dir / "cluster_timestamps.parquet", index=False)
        label_meta = {
            "split": args.split,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "daytime_mode": used_day_mode,
            "daytime_ge_min": float(used_ge_min),
            "daytime_ghi_min": float(used_ghi_min),
            "window_mins": int(args.window_mins),
            "min_samples_per_window": int(args.min_samples_per_window),
            "scaler": args.scaler,
            "use_pca": bool(args.use_pca),
            "pca_n": int(args.pca_n),
            "feature_count": int(X.shape[1]),
            "sample_count": int(label_array.size),
            "feature_cols": meta.get("feature_cols", []),
        }
        dump_json(label_meta, out_dir / "cluster_labels_meta.json")
        logger.info("聚类标签已导出: %s", out_dir)
    except Exception as exc:
        logger.warning("导出聚类标签失败: %s", exc)

    # Export scaler and pca for downstream inference
    try:
        scaler_obj = meta.get("scaler")
        if scaler_obj is not None:
            with open(out_dir / "weather_scaler.pkl", "wb") as f:
                pickle.dump(scaler_obj, f)
            logger.info("Scaler 已保存: %s", out_dir / "weather_scaler.pkl")
        
        pca_obj = meta.get("pca")
        if pca_obj is not None:
            with open(out_dir / "weather_pca.pkl", "wb") as f:
                pickle.dump(pca_obj, f)
            logger.info("PCA 已保存: %s", out_dir / "weather_pca.pkl")
        
        # Save feature weights
        weights = meta.get("weights")
        if weights is not None:
            np.save(out_dir / "feature_weights.npy", np.array(weights))
            logger.info("特征权重已保存: %s", out_dir / "feature_weights.npy")
    except Exception as exc:
        logger.warning("导出 scaler/pca/weights 失败: %s", exc)

    # Train and export LightGBM weather classifier
    prog.start("LightGBM")
    if _HAVE_LIGHTGBM:
        try:
            lgb_params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "seed": 42,
            }
            # Create LightGBM dataset
            lgb_train = lgb.Dataset(X, label=label_array.astype(int))
            # Train with early stopping simulation (fixed rounds)
            lgb_model = lgb.train(
                lgb_params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train],
                callbacks=[lgb.log_evaluation(period=0)],  # Suppress logs
            )
            # Save model (use booster_to_string for paths with non-ASCII chars)
            lgb_model_path = out_dir / "weather_lgb_model.txt"
            model_str = lgb_model.model_to_string()
            with open(lgb_model_path, "w", encoding="utf-8") as f:
                f.write(model_str)
            
            # Validate model accuracy (train set)
            lgb_pred = lgb_model.predict(X)
            lgb_pred_labels = np.argmax(lgb_pred, axis=1)
            lgb_accuracy = float(np.mean(lgb_pred_labels == label_array))
            
            # Cross-validation to check generalization (5-fold)
            cv_scores = []
            n_samples = X.shape[0]
            fold_size = n_samples // 5
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for fold_i in range(5):
                val_start = fold_i * fold_size
                val_end = val_start + fold_size if fold_i < 4 else n_samples
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                
                X_cv_train, y_cv_train = X[train_idx], label_array[train_idx]
                X_cv_val, y_cv_val = X[val_idx], label_array[val_idx]
                
                cv_train_data = lgb.Dataset(X_cv_train, label=y_cv_train.astype(int))
                cv_model = lgb.train(
                    lgb_params, cv_train_data, num_boost_round=100,
                    callbacks=[lgb.log_evaluation(period=0)]
                )
                cv_pred = np.argmax(cv_model.predict(X_cv_val), axis=1)
                cv_scores.append(float(np.mean(cv_pred == y_cv_val)))
            
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
            
            lgb_meta = {
                "train_accuracy": lgb_accuracy,
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std,
                "cv_scores": cv_scores,
                "num_features": int(X.shape[1]),
                "num_samples": int(X.shape[0]),
                "params": lgb_params,
            }
            dump_json(lgb_meta, out_dir / "weather_lgb_meta.json")
            logger.info(
                "LightGBM 天气分类器已训练, 训练准确率: %.2f%%, 5折交叉验证: %.2f%% (+/- %.2f%%)",
                lgb_accuracy * 100, cv_mean * 100, cv_std * 100
            )
            prog.update(100, f"acc={lgb_accuracy:.2%}")
            prog.done(True)
        except Exception as exc:
            logger.warning("LightGBM 训练失败: %s", exc)
            prog.done(False, str(exc))
    else:
        logger.warning("LightGBM 未安装, 跳过天气分类器训练。请运行: pip install lightgbm")
        prog.done(False, "lightgbm not installed")

    # Auto-update config to point to latest cluster results
    prog.start("Update config")
    try:
        cfg_path = Path(args.config)
        cfg_obj = load_yaml(cfg_path)
        fe = cfg_obj.get("feature_engineering", {})
        
        # Update reuse_cluster_labels.splits.<split>
        reuse_cfg = fe.get("reuse_cluster_labels", {})
        splits_cfg = reuse_cfg.get("splits", {})
        split_name = args.split  # train, val, or test
        
        # Calculate relative path from project root
        try:
            rel_path = out_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = out_dir
        
        splits_cfg[split_name] = {
            "labels_path": str(rel_path / "cluster_labels.npy"),
            "timestamps_path": str(rel_path / "cluster_timestamps.parquet"),
            "meta_path": str(rel_path / "cluster_labels_meta.json"),
        }
        reuse_cfg["splits"] = splits_cfg
        fe["reuse_cluster_labels"] = reuse_cfg
        
        # Update weather_lgb_classifier.model_dir for train split only
        if split_name == "train":
            lgb_cfg = fe.get("weather_lgb_classifier", {})
            lgb_cfg["model_dir"] = str(rel_path)
            fe["weather_lgb_classifier"] = lgb_cfg
        
        cfg_obj["feature_engineering"] = fe
        dump_yaml(cfg_obj, cfg_path)
        logger.info("已自动更新配置 %s.splits.%s -> %s", "reuse_cluster_labels", split_name, rel_path)
        if split_name == "train":
            logger.info("已自动更新配置 weather_lgb_classifier.model_dir -> %s", rel_path)
        prog.update(100, "ok")
        prog.done(True)
    except Exception as e:
        logger.warning("自动更新配置失败: %s", e)
        prog.done(False, str(e))

    # Optionally write thresholds back to config
    if args.write_config:
        try:
            cfg_path = Path(args.config)
            cfg_obj = load_yaml(cfg_path)
            fe = cfg_obj.get("feature_engineering", {})
            # Threshold arrays expected as [low, high]
            fe["ci_thresholds"] = [float(ci_low_est.value), float(ci_high_est.value)]
            fe["wsi_thresholds"] = [float(wsi_low_est.value), float(wsi_high_est.value)]
            # Attach metadata for traceability
            fe_meta = fe.get("thresholds_meta", {})
            fe_meta.update({
                "calibrated_by": "tools.init_thresholds",
                "calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "counts": cluster_info.get("counts", {}),
                "method": thresholds_yaml["thresholds"]["ci"]["method"],
                "overlap_mode": str(args.overlap_mode),
            })
            fe["thresholds_meta"] = fe_meta
            cfg_obj["feature_engineering"] = fe
            dump_yaml(cfg_obj, cfg_path)
            logger.info("已将阈值写回配置: %s （feature_engineering.ci_thresholds / wsi_thresholds）", cfg_path)
        except Exception as e:
            logger.warning("写回配置失败: %s", e)
    prog.start("Plots")
    plot_diagnostics(out_dir,
                     ci_vals=ci_vals,
                     wsi_vals=wsi_vals,
                     weather_labels=labels,
                     power_vals=sub["power"].values,
                     irradiance_vals=sub["irradiance"].values)
    prog.update(100, "ok")
    prog.done(True, str(out_dir))
    logger.info("Thresholds and diagnostics written to: %s", out_dir)


if __name__ == "__main__":
    main()
