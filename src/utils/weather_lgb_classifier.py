# -*- coding: utf-8 -*-
"""
LightGBM Weather Classifier for inference.

This module loads a pre-trained LightGBM model and associated preprocessing
artifacts (scaler, pca, feature weights) to classify weather types (sunny,
cloudy, overcast) from input features.

Used in the test/evaluation phase to infer weather labels when true labels
are not available, ensuring no data leakage.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAVE_LIGHTGBM = True
except ImportError:
    _HAVE_LIGHTGBM = False

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

logger = logging.getLogger(__name__)

WEATHER_NAMES = {0: "sunny", 1: "cloudy", 2: "overcast"}


class WeatherLGBClassifier:
    """
    LightGBM-based weather classifier for inference.
    
    This classifier uses the same feature engineering pipeline as the
    clustering phase to ensure consistency.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Optional[Union[str, Path]] = None,
        pca_path: Optional[Union[str, Path]] = None,
        weights_path: Optional[Union[str, Path]] = None,
        meta_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the classifier by loading pre-trained model and transforms.
        
        Args:
            model_path: Path to LightGBM model file (.txt)
            scaler_path: Path to scaler pickle file
            pca_path: Path to PCA pickle file (optional)
            weights_path: Path to feature weights numpy file
            meta_path: Path to cluster labels meta JSON
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.pca_path = Path(pca_path) if pca_path else None
        self.weights_path = Path(weights_path) if weights_path else None
        self.meta_path = Path(meta_path) if meta_path else None
        
        self.model = None
        self.scaler = None
        self.pca = None
        self.feature_weights = None
        self.meta = None
        self.feature_cols = None
        
        self._load_model()
        self._load_transforms()
        self._load_meta()
    
    def _load_model(self):
        """Load the LightGBM model."""
        if not _HAVE_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model from string to handle non-ASCII paths
        with open(self.model_path, "r", encoding="utf-8") as f:
            model_str = f.read()
        self.model = lgb.Booster(model_str=model_str)
        logger.info("Loaded LightGBM model from %s", self.model_path)
    
    def _load_transforms(self):
        """Load scaler, PCA, and feature weights."""
        # Load scaler
        if self.scaler_path and self.scaler_path.exists():
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded scaler from %s", self.scaler_path)
        
        # Load PCA
        if self.pca_path and self.pca_path.exists():
            with open(self.pca_path, "rb") as f:
                self.pca = pickle.load(f)
            logger.info("Loaded PCA from %s", self.pca_path)
        
        # Load feature weights
        if self.weights_path and self.weights_path.exists():
            self.feature_weights = np.load(self.weights_path)
            logger.info("Loaded feature weights from %s", self.weights_path)
    
    def _load_meta(self):
        """Load metadata including feature columns."""
        if self.meta_path and self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self.feature_cols = self.meta.get("feature_cols", [])
            logger.info("Loaded meta from %s, feature_cols=%d", 
                       self.meta_path, len(self.feature_cols))
    
    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """Apply scaler transformation."""
        if self.scaler is None:
            return X
        
        if isinstance(self.scaler, dict):
            # NumPy fallback scaler
            if self.scaler.get("type") == "numpy_fallback":
                med = np.array(self.scaler["median"])
                iqr = np.array(self.scaler["iqr"])
                return (X - med) / iqr
        elif hasattr(self.scaler, "transform"):
            # sklearn scaler
            return self.scaler.transform(X)
        
        return X
    
    def _apply_pca(self, X: np.ndarray) -> np.ndarray:
        """Apply PCA transformation."""
        if self.pca is None:
            return X
        
        if isinstance(self.pca, dict):
            # Custom PCA dict (GPU or NumPy fallback)
            mean = np.array(self.pca["mean"])
            components = np.array(self.pca["components"])
            Xc = X - mean
            return Xc @ components
        elif hasattr(self.pca, "transform"):
            # sklearn PCA
            return self.pca.transform(X)
        
        return X
    
    def _apply_weights(self, X: np.ndarray) -> np.ndarray:
        """Apply feature weights."""
        if self.feature_weights is None:
            return X
        return X * self.feature_weights[np.newaxis, :]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict weather labels for input features.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
               Features should be in the same order as training.
        
        Returns:
            Array of weather labels (0=sunny, 1=cloudy, 2=overcast)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Apply transforms
        Xs = self._apply_scaler(X)
        Xs = self._apply_weights(Xs)
        Xs = self._apply_pca(Xs)
        
        # Predict probabilities
        probs = self.model.predict(Xs)
        
        # Get predicted labels
        labels = np.argmax(probs, axis=1)
        
        return labels.astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict weather probabilities for input features.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
        
        Returns:
            Array of probabilities of shape (n_samples, 3)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Apply transforms
        Xs = self._apply_scaler(X)
        Xs = self._apply_weights(Xs)
        Xs = self._apply_pca(Xs)
        
        # Predict probabilities
        probs = self.model.predict(Xs)
        
        return probs
    
    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "WeatherLGBClassifier":
        """
        Create classifier from a directory containing all artifacts.
        
        Expected files:
            - weather_lgb_model.txt
            - weather_scaler.pkl
            - weather_pca.pkl (optional)
            - feature_weights.npy
            - cluster_labels_meta.json
        
        Args:
            directory: Path to directory containing artifacts
        
        Returns:
            WeatherLGBClassifier instance
        """
        directory = Path(directory)
        
        model_path = directory / "weather_lgb_model.txt"
        scaler_path = directory / "weather_scaler.pkl"
        pca_path = directory / "weather_pca.pkl"
        weights_path = directory / "feature_weights.npy"
        meta_path = directory / "cluster_labels_meta.json"
        
        return cls(
            model_path=model_path,
            scaler_path=scaler_path if scaler_path.exists() else None,
            pca_path=pca_path if pca_path.exists() else None,
            weights_path=weights_path if weights_path.exists() else None,
            meta_path=meta_path if meta_path.exists() else None,
        )


def compute_clustering_features(
    df: pd.DataFrame,
    window_mins: int,
    min_samples_per_window: int,
    freq_minutes: int,
    feature_cols: List[str],
    features_basic_only: bool = False,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Compute clustering features for a DataFrame.
    
    This function replicates the feature engineering from init_thresholds.py
    to ensure consistency between training and inference.
    
    Args:
        df: Input DataFrame with required columns
        window_mins: Rolling window size in minutes
        min_samples_per_window: Minimum samples per window
        freq_minutes: Data frequency in minutes
        feature_cols: List of feature column names to extract
        features_basic_only: If True, skip rolling statistics
    
    Returns:
        Tuple of (feature_array, valid_index)
    """
    df = df.copy()
    required = ["power", "irradiance", "temperature", "pressure", "humidity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    
    # Time-of-day encoding
    _mins = (df.index.hour * 60 + df.index.minute + (df.index.second / 60.0)).astype(float)
    _t = _mins / 1440.0
    df["tod_sin"] = np.sin(2 * np.pi * _t)
    df["tod_cos"] = np.cos(2 * np.pi * _t)
    df["tod_bin"] = (df.index.hour * 60 + df.index.minute).astype(int)
    
    # Robust group z-score
    def _robust_group_z(series: pd.Series, group: pd.Series, clip: float = 5.0) -> np.ndarray:
        g = series.groupby(group)
        med = g.transform("median")
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
    
    # Z-scores
    if not features_basic_only:
        df["z_power"] = _robust_group_z(df["power"].astype(float), df["tod_bin"])
        df["z_ghi"] = _robust_group_z(df["irradiance"].astype(float), df["tod_bin"])
        df["z_hum"] = _robust_group_z(df["humidity"].astype(float), df["tod_bin"])
        df["z_temp"] = _robust_group_z(df["temperature"].astype(float), df["tod_bin"])
        df["z_pres"] = _robust_group_z(df["pressure"].astype(float), df["tod_bin"])
    
    # Rolling statistics
    steps = max(int(round(window_mins / max(freq_minutes, 1))), 1)
    if min_samples_per_window > steps:
        min_samples_per_window = steps
    
    roll_cols = ["power", "irradiance", "humidity", "pressure"]
    
    if not features_basic_only:
        for col in roll_cols:
            r = df[col].rolling(window=steps, min_periods=min_samples_per_window)
            df[f"{col}_rmean"] = r.mean()
            df[f"{col}_rstd"] = r.std()
            df[f"{col}_diff1"] = df[col].diff(1)
    
    # Extract features
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) != len(feature_cols):
        missing_cols = set(feature_cols) - set(available_cols)
        logger.warning("Missing feature columns: %s", missing_cols)
    
    df_feat = df[available_cols].dropna()
    
    if df_feat.empty:
        return np.empty((0, len(available_cols)), dtype=float), pd.DatetimeIndex([])
    
    X = df_feat.values.astype(float)
    
    return X, df_feat.index
