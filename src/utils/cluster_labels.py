"""Helpers for reusing clustering labels produced by tools.init_thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import logging

import numpy as np
import pandas as pd


@dataclass
class ClusterLabelBundle:
    """Container for previously clustered weather labels."""

    series: pd.Series
    meta: Dict[str, Any]
    source: str


def _read_timestamps(path: Path) -> pd.DatetimeIndex:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported timestamp file format: {path}")

    if "timestamp" not in df.columns:
        raise ValueError(f"Timestamp file缺少 'timestamp' 列: {path}")
    return pd.to_datetime(df["timestamp"])


def load_cluster_label_bundle(cfg: Dict[str, Any]) -> ClusterLabelBundle:
    """Load labels/timestamps/meta according to reuse_cluster_labels config."""

    labels_path = cfg.get("labels_path")
    timestamps_path = cfg.get("timestamps_path")
    if not labels_path or not timestamps_path:
        raise ValueError("reuse_cluster_labels 需要 labels_path 和 timestamps_path")

    labels_file = Path(labels_path)
    ts_file = Path(timestamps_path)
    if not labels_file.exists() or not ts_file.exists():
        raise FileNotFoundError("Cluster label files 未找到，请检查配置路径")

    labels = np.load(labels_file)
    timestamps = _read_timestamps(ts_file)
    if len(labels) != len(timestamps):
        raise ValueError(
            f"标签数量({len(labels)})与时间戳数量({len(timestamps)})不一致，请检查输入文件"
        )

    meta: Dict[str, Any] = {}
    meta_path = cfg.get("meta_path")
    if meta_path:
        meta_file = Path(meta_path)
        if meta_file.exists():
            try:
                with meta_file.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh) or {}
            except Exception as exc:
                raise ValueError(f"无法读取 meta 文件: {meta_path}: {exc}") from exc

    series = pd.Series(labels.astype(int), index=pd.to_datetime(timestamps))
    return ClusterLabelBundle(series=series, meta=meta, source=str(labels_file))


def assign_cluster_labels(
    index: pd.DatetimeIndex,
    base_labels: np.ndarray,
    day_mask: Optional[np.ndarray],
    bundle: ClusterLabelBundle,
    fallback_strategy: str,
    split_name: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Merge reusable labels onto the provided index并返回覆盖率统计."""

    fallback = (fallback_strategy or "classify").strip().lower()
    if fallback not in {"classify"}:
        raise ValueError("reuse_cluster_labels.fallback 当前仅支持 'classify'")

    if base_labels.shape[0] != len(index):
        raise ValueError(
            f"基础标签长度({base_labels.shape[0]})与索引长度({len(index)})不一致，请检查输入文件"
        )

    if len(index) == 0:
        return np.asarray(base_labels, dtype=np.int64), {
            "total_samples": 0,
            "matched_total": 0,
            "coverage_total": float("nan"),
            "day_samples": 0,
            "day_matched": 0,
            "day_coverage": float("nan"),
            "fallback_day": 0,
            "night_samples": 0,
        }

    day_mask_arr = (
        np.asarray(day_mask, dtype=bool) if day_mask is not None else np.ones(len(index), dtype=bool)
    )
    
    # 诊断：打印时间范围对比
    bundle_index = bundle.series.index
    logger.info(
        "[聚类标签诊断] %s: 当前数据时间范围=[%s ~ %s], 样本数=%d, 白天样本=%d",
        split_name,
        index.min() if len(index) > 0 else "N/A",
        index.max() if len(index) > 0 else "N/A",
        len(index),
        int(day_mask_arr.sum()),
    )
    logger.info(
        "[聚类标签诊断] %s: 聚类标签时间范围=[%s ~ %s], 标签数=%d, 来源=%s",
        split_name,
        bundle_index.min() if len(bundle_index) > 0 else "N/A",
        bundle_index.max() if len(bundle_index) > 0 else "N/A",
        len(bundle_index),
        bundle.source,
    )
    
    # 检查时间戳是否有交集
    common_count = len(index.intersection(bundle_index))
    logger.info(
        "[聚类标签诊断] %s: 精确匹配的时间戳数量=%d (%.2f%%)",
        split_name,
        common_count,
        (common_count / len(index) * 100) if len(index) > 0 else 0,
    )
    
    # 如果匹配数为0，打印样例帮助排查
    if common_count == 0 and len(index) > 0 and len(bundle_index) > 0:
        sample_idx = index[:3].tolist()
        sample_bundle = bundle_index[:3].tolist()
        logger.warning(
            "[聚类标签诊断] %s: 无精确匹配! 当前数据前3个时间戳=%s, 聚类标签前3个时间戳=%s",
            split_name,
            sample_idx,
            sample_bundle,
        )
        # 检查时间戳类型
        logger.warning(
            "[聚类标签诊断] %s: 当前索引dtype=%s, 聚类索引dtype=%s",
            split_name,
            index.dtype,
            bundle_index.dtype,
        )
    
    matched = bundle.series.reindex(index)
    reuse_mask = matched.notna()

    final = np.asarray(base_labels, dtype=np.int64).copy()
    if reuse_mask.any():
        final[reuse_mask.to_numpy()] = matched[reuse_mask].astype(int).to_numpy()

    day_samples = int(day_mask_arr.sum())
    day_matched = int((reuse_mask.to_numpy() & day_mask_arr).sum())
    fallback_day = int((~reuse_mask.to_numpy() & day_mask_arr).sum())
    coverage_day = (day_matched / day_samples) if day_samples else float("nan")
    
    logger.info(
        "[聚类标签结果] %s: 白天覆盖率=%.3f (匹配=%d, 总白天=%d, 回退=%d)",
        split_name,
        coverage_day,
        day_matched,
        day_samples,
        fallback_day,
    )
    
    stats = {
        "total_samples": len(index),
        "matched_total": int(reuse_mask.sum()),
        "coverage_total": float(reuse_mask.sum() / len(index)),
        "day_samples": day_samples,
        "day_matched": day_matched,
        "day_coverage": coverage_day,
        "fallback_day": fallback_day,
        "night_samples": int((~day_mask_arr).sum()),
    }

    return final, stats
