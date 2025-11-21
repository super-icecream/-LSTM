"""
双路径天气识别模块
功能：实现CI清晰度指数和WSI天气状态指数的双路径天气分类
作者：DLFE-LSTM-WSI Team
日期：2025-09-26
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class WeatherClassifier:
    """
    双路径天气识别分类器

    实现CI（清晰度指数）和WSI（天气状态指数）的双路径融合天气分类。
    CI基于太阳辐照度与理论地外辐照度的比值，
    WSI基于气压、湿度等气象参数的综合指数。

    Attributes:
        location_lat (float): 地理纬度（度）
        location_lon (float): 地理经度（度）
        ci_thresholds (list): CI分类阈值 [阴天阈值, 晴天阈值]
        wsi_thresholds (list): WSI分类阈值 [晴天阈值, 阴天阈值]
        fusion_weights (dict): 融合权重 {'ci': 权重, 'wsi': 权重}
    """

    def __init__(self,
                 location_lat: float = 38.5,  # 甘肃纬度
                 location_lon: float = 105.0,  # 甘肃经度
                 elevation: float = 1500,  # 海拔高度(米)
                 time_zone_hours: Optional[float] = None,  # 时区（小时，例：CST=8），若 None 则回退自动推断
                 ci_thresholds: List[float] = [0.2, 0.6],
                 wsi_thresholds: List[float] = [0.3, 0.7],
                 fusion_weights: Dict[str, float] = {'ci': 0.7, 'wsi': 0.3},
                 # 日间掩码参数：模式 + 阈值
                 daytime_ge_min: float = 20.0,
                 daytime_mode: str = "ge",         # ge | ghi | or | and
                 daytime_ghi_min: float = 5.0,     # 仅在模式包含 ghi 时使用
                 night_handling: str = "exclude"):
        """
        初始化天气分类器

        Args:
            location_lat: 地理纬度（度）
            location_lon: 地理经度（度）
            elevation: 海拔高度（米）
            ci_thresholds: CI分类阈值 [阴天阈值, 晴天阈值]
            wsi_thresholds: WSI分类阈值 [晴天阈值, 阴天阈值]
            fusion_weights: 初始融合权重
        """
        self.location_lat = location_lat
        self.location_lon = location_lon
        self.elevation = elevation
        # 明确记录时区（小时）；若为 None 则在计算时尝试从时间戳 tzinfo 推断，最后再回退经度/15
        self.time_zone_hours: Optional[float] = float(time_zone_hours) if time_zone_hours is not None else None

        # 转换为弧度
        self.lat_rad = np.radians(location_lat)
        self.lon_rad = np.radians(location_lon)

        # 分类阈值
        self.ci_thresholds = ci_thresholds
        self.wsi_thresholds = wsi_thresholds

        # 融合权重（归一化）
        total_weight = fusion_weights['ci'] + fusion_weights['wsi']
        self.fusion_weights = {
            'ci': fusion_weights['ci'] / total_weight,
            'wsi': fusion_weights['wsi'] / total_weight
        }

        # 白天判定阈值与夜间处理策略
        self.daytime_ge_min = float(daytime_ge_min)
        self.daytime_mode = str(daytime_mode).lower().strip()
        if self.daytime_mode not in {"ge", "ghi", "or", "and"}:
            raise ValueError(f"daytime_mode 仅支持 ge/ghi/or/and，收到: {daytime_mode}")
        self.daytime_ghi_min = float(daytime_ghi_min)
        night_handling = night_handling.lower()
        if night_handling not in {"exclude", "assign-cloudy"}:
            raise ValueError(f"night_handling 仅支持 'exclude' 或 'assign-cloudy'，收到: {night_handling}")
        self.night_handling = night_handling

        # 最近一次 classify 时的白天掩码缓存
        self.last_day_mask: Optional[np.ndarray] = None

        # 太阳常数
        self.solar_constant = 1367.0  # W/m²

        # 天气类别定义
        self.weather_types = {
            0: 'sunny',      # 晴天
            1: 'cloudy',     # 多云
            2: 'overcast'    # 阴天
        }

        logger.info(f"天气分类器初始化: 位置({location_lat:.2f}°N, {location_lon:.2f}°E), "
                   f"海拔{elevation}m"
                   f"{', 时区UTC%+g' % self.time_zone_hours if self.time_zone_hours is not None else ''}"
                   f"，日间掩码: {self.daytime_mode.upper()}"
                   f"{'(GE≥%.2f W/m²' % self.daytime_ge_min if self.daytime_mode in {'ge','or','and'} else ''}"
                   f"{' OR ' if self.daytime_mode=='or' else (' AND ' if self.daytime_mode=='and' else '')}"
                   f"{'GHI≥%.2f W/m²' % self.daytime_ghi_min if self.daytime_mode in {'ghi','or','and'} else ''}"
                   f"{')' if self.daytime_mode in {'ge','or','and'} else ''}")

    def calculate_ci(self, ghi: Union[float, np.ndarray],
                    timestamp: Union[datetime, pd.DatetimeIndex]) -> Tuple[Union[float, np.ndarray], Union[bool, np.ndarray]]:
        """
        计算清晰度指数CI

        CI = GHI / GE
        其中GHI为实测全球水平辐照度，GE为理论地外辐照度

        Args:
            ghi: 全球水平辐照度 (W/m²)
            timestamp: 时间戳

        Returns:
            Tuple[CI值/数组, 白天掩码]
        """
        # 计算理论地外辐照度
        ge = self._calculate_extraterrestrial_radiation(timestamp)

        ghi_arr = np.asarray(ghi, dtype=float)
        ge_arr = np.asarray(ge, dtype=float)

        # 白天判定
        # - ge: GE ≥ ge_min
        # - ghi: GHI ≥ ghi_min
        # - or: (GE ≥ ge_min) OR (GHI ≥ ghi_min)
        # - and: (GE ≥ ge_min) AND (GHI ≥ ghi_min)
        if self.daytime_mode == "ge":
            day_mask = ge_arr >= self.daytime_ge_min
        elif self.daytime_mode == "ghi":
            day_mask = ghi_arr >= self.daytime_ghi_min
        elif self.daytime_mode == "or":
            day_mask = (ge_arr >= self.daytime_ge_min) | (ghi_arr >= self.daytime_ghi_min)
        else:  # "and"
            day_mask = (ge_arr >= self.daytime_ge_min) & (ghi_arr >= self.daytime_ghi_min)

        # 避免除零，仅对白天样本计算
        safe_ge = np.where(day_mask, ge_arr, 1.0)
        ci = np.full_like(ge_arr, np.nan, dtype=float)
        valid_mask = day_mask & (safe_ge > 0)
        if np.any(valid_mask):
            ci[valid_mask] = np.clip(ghi_arr[valid_mask] / safe_ge[valid_mask], 0, 1.2)

        day_mask_bool = day_mask.astype(bool)
        if isinstance(day_mask_bool, np.ndarray) and day_mask_bool.ndim > 0:
            self.last_day_mask = day_mask_bool.copy()
            day_mask_out: Union[np.ndarray, bool] = day_mask_bool
        else:
            day_flag = bool(np.asarray(day_mask_bool))
            self.last_day_mask = np.array([day_flag])
            day_mask_out = day_flag

        if isinstance(ci, np.ndarray) and ci.ndim == 0:
            return float(ci), day_mask_out
        return ci, day_mask_out

    def _calculate_extraterrestrial_radiation(self,
                                             timestamp: Union[datetime, pd.DatetimeIndex]) -> np.ndarray:
        """
        计算理论地外辐照度GE

        GE = I_0 * E_0 * (sin(d_c)*sin(l) + cos(d_c)*cos(l)*cos(w_s))

        Args:
            timestamp: 时间戳

        Returns:
            理论地外辐照度 (W/m²)
        """
        # 处理时间戳格式
        if isinstance(timestamp, datetime):
            timestamps = [timestamp]
        elif isinstance(timestamp, pd.DatetimeIndex):
            timestamps = timestamp.to_pydatetime()
        else:
            timestamps = pd.to_datetime(timestamp).to_pydatetime()

        ge_values = []

        for ts in timestamps:
            # 计算年积日
            julian_day = ts.timetuple().tm_yday

            # 计算日角Γ (弧度)
            gamma = 2 * np.pi * (julian_day - 1) / 365

            # 轨道偏心率修正因子E_0
            e0 = (1.000110 +
                  0.034221 * np.cos(gamma) +
                  0.001280 * np.sin(gamma) +
                  0.000719 * np.cos(2 * gamma) +
                  0.000077 * np.sin(2 * gamma))

            # 太阳赤纬d_c (弧度)
            declination = (0.006918 -
                          0.399912 * np.cos(gamma) +
                          0.070257 * np.sin(gamma) -
                          0.006758 * np.cos(2 * gamma) +
                          0.000907 * np.sin(2 * gamma) -
                          0.002697 * np.cos(3 * gamma) +
                          0.00148 * np.sin(3 * gamma))

            # 时差方程E_t (分钟)
            equation_of_time = 4 * (0.0066 +
                                   7.3525 * np.cos(gamma + 1.498) +
                                   9.9359 * np.cos(2 * gamma + 1.1359) +
                                   0.3387 * np.cos(3 * gamma + 0.0607))

            # 计算真太阳时
            # 注意：数据时间轴假定为“站点当地标准时”。优先使用时间戳自身 tzinfo；
            # 否则使用 self.time_zone_hours；最后回退为 round(lon/15)。
            local_hour = ts.hour + ts.minute / 60 + ts.second / 3600

            if hasattr(ts, "tzinfo") and ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
                tz_hours = ts.tzinfo.utcoffset(ts).total_seconds() / 3600.0
            elif self.time_zone_hours is not None:
                tz_hours = float(self.time_zone_hours)
            else:
                tz_hours = float(round(self.location_lon / 15.0))

            # 经度时差修正：相对于时区中央经线(15°×tz_hours)
            longitude_correction = 4 * (self.location_lon - tz_hours * 15.0)  # 分钟

            # 真太阳时（小时）
            solar_time = local_hour + (equation_of_time + longitude_correction) / 60

            # 太阳时角w_s (弧度)
            hour_angle = np.pi * (solar_time - 12) / 12

            # 计算太阳高度角的正弦值
            sin_elevation = (np.sin(declination) * np.sin(self.lat_rad) +
                           np.cos(declination) * np.cos(self.lat_rad) * np.cos(hour_angle))

            # 理论地外辐照度
            if sin_elevation > 0:
                ge = self.solar_constant * e0 * sin_elevation
            else:
                ge = 0  # 太阳在地平线以下

            ge_values.append(ge)

        return np.array(ge_values)

    def calculate_wsi(self, pressure: Union[float, np.ndarray],
                     humidity: Union[float, np.ndarray],
                     temperature: Union[float, np.ndarray],
                     time_delta: float = 15) -> Union[float, np.ndarray]:
        """
        计算天气状态指数WSI

        WSI = α*Pre_norm + β*Hum_norm + γ*Press_change

        Args:
            pressure: 气压 (hPa)
            humidity: 相对湿度 (%)
            temperature: 温度 (°C)
            time_delta: 时间间隔（分钟）

        Returns:
            WSI值或WSI数组
        """
        # 确保输入为数组
        pressure = np.asarray(pressure)
        humidity = np.asarray(humidity)
        temperature = np.asarray(temperature)

        # 气压归一化项
        # 高压对应晴天，低压对应阴天
        pressure_norm = (1040 - pressure) / 60  # 基准1040 hPa
        pressure_norm = np.clip(pressure_norm, -1, 1)

        # 湿度归一化项
        # 高湿度对应阴天
        humidity_norm = humidity / 100
        humidity_norm = np.clip(humidity_norm, 0, 1)

        # 气压变化率项（使用差分计算）
        if len(pressure.shape) > 0 and len(pressure) > 1:
            # 计算气压变化率
            pressure_diff = np.diff(pressure, prepend=pressure[0])
            pressure_change = -pressure_diff / (time_delta / 60)  # hPa/hour
            pressure_change = np.maximum(0, pressure_change) / 10  # 归一化
            pressure_change = np.clip(pressure_change, 0, 1)
        else:
            pressure_change = 0

        # 温度修正因子（可选）
        # 极端温度可能影响天气判断
        temp_factor = 1.0
        if np.any(temperature < 0) or np.any(temperature > 35):
            temp_factor = 1.1  # 极端温度增加不确定性

        # WSI权重系数
        alpha = 0.4  # 气压权重
        beta = 0.5   # 湿度权重
        gamma = 0.1  # 气压变化率权重

        # 计算WSI
        wsi = alpha * pressure_norm + beta * humidity_norm + gamma * pressure_change
        wsi = wsi * temp_factor

        # 归一化到[0, 1]
        wsi = np.clip(wsi, 0, 1)

        return wsi

    def classify_ci(self, ci: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        基于CI进行天气分类

        Args:
            ci: 清晰度指数

        Returns:
            天气类别 (0=晴天, 1=多云, 2=阴天)
        """
        ci = np.asarray(ci)

        # 分类规则
        sunny_mask = ci >= self.ci_thresholds[1]  # CI >= 0.6
        overcast_mask = ci <= self.ci_thresholds[0]  # CI <= 0.2

        # 初始化为多云
        weather_type = np.ones_like(ci, dtype=int)

        # 应用分类
        weather_type[sunny_mask] = 0  # 晴天
        weather_type[overcast_mask] = 2  # 阴天

        return weather_type if ci.shape else int(weather_type)

    def classify_wsi(self, wsi: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        基于WSI进行天气分类

        Args:
            wsi: 天气状态指数

        Returns:
            天气类别 (0=晴天, 1=多云, 2=阴天)
        """
        wsi = np.asarray(wsi)

        # 分类规则（注意WSI的逻辑与CI相反）
        sunny_mask = wsi < self.wsi_thresholds[0]  # WSI < 0.3
        overcast_mask = wsi > self.wsi_thresholds[1]  # WSI > 0.7

        # 初始化为多云
        weather_type = np.ones_like(wsi, dtype=int)

        # 应用分类
        weather_type[sunny_mask] = 0  # 晴天
        weather_type[overcast_mask] = 2  # 阴天

        return weather_type if wsi.shape else int(weather_type)

    def classify(self, data: pd.DataFrame,
                ghi_col: str = 'irradiance',
                pressure_col: str = 'pressure',
                humidity_col: str = 'humidity',
                temperature_col: str = 'temperature') -> Dict[str, np.ndarray]:
        """
        双路径融合天气分类，并返回白天掩码与路径信息。

        Args:
            data: 输入数据，要求包含辐照度、气压、湿度、温度列且索引为 DatetimeIndex。
            ghi_col: 辐照度列名。
            pressure_col: 气压列名。
            humidity_col: 湿度列名。
            temperature_col: 温度列名。

        Returns:
            包含标签、白天掩码及路径诊断信息的字典。
        """
        required_cols = [ghi_col, pressure_col, humidity_col, temperature_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要数据列: {missing_cols}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError('数据索引需为 DatetimeIndex 时间索引')

        ci_values, day_mask = self.calculate_ci(data[ghi_col].values, data.index)
        wsi_values = self.calculate_wsi(
            data[pressure_col].values,
            data[humidity_col].values,
            data[temperature_col].values
        )

        ci_arr = np.asarray(ci_values, dtype=float)
        wsi_arr = np.asarray(wsi_values, dtype=float)
        day_arr = np.asarray(day_mask, dtype=bool)

        if ci_arr.ndim == 0:
            ci_arr = ci_arr.reshape(1)
            wsi_arr = wsi_arr.reshape(1)
            day_arr = day_arr.reshape(1)

        n_samples = ci_arr.shape[0]
        ci_path = np.full(n_samples, -1, dtype=int)
        wsi_path = np.full(n_samples, -1, dtype=int)

        day_indices = np.where(day_arr)[0]
        if day_indices.size:
            ci_path_day = np.asarray(self.classify_ci(ci_arr[day_indices]), dtype=int)
            wsi_path_day = np.asarray(self.classify_wsi(wsi_arr[day_indices]), dtype=int)
            ci_path[day_indices] = ci_path_day
            wsi_path[day_indices] = wsi_path_day

        final_labels = np.full(n_samples, -1, dtype=int)
        if day_indices.size:
            fused = self._fusion_decision(
                ci_path[day_indices],
                wsi_path[day_indices],
                ci_arr[day_indices],
                wsi_arr[day_indices]
            )
            final_labels[day_indices] = fused

        if self.night_handling == 'assign-cloudy':
            final_labels[~day_arr] = 1

        day_labels = final_labels[day_arr]
        stats = self.get_weather_statistics(day_labels) if day_labels.size else {
            'distribution': {},
            'percentages': {},
            'transitions': 0
        }

        logger.info('天气分类分布(白天末端): %s', stats.get('distribution', {}))

        if isinstance(day_arr, np.ndarray):
            self.last_day_mask = day_arr.copy()
        else:
            self.last_day_mask = np.array([bool(day_arr)])

        return {
            'labels': final_labels,
            'day_mask': day_arr,
            'ci': ci_arr,
            'wsi': wsi_arr,
            'ci_path': ci_path,
            'wsi_path': wsi_path,
            'statistics': stats,
        }

    def _fusion_decision(self, ci_weather: np.ndarray, wsi_weather: np.ndarray,
                        ci_values: np.ndarray, wsi_values: np.ndarray) -> np.ndarray:
        """
        融合CI和WSI的分类结果

        采用加权投票机制，考虑置信度

        Args:
            ci_weather: CI分类结果
            wsi_weather: WSI分类结果
            ci_values: CI原始值（用于计算置信度）
            wsi_values: WSI原始值（用于计算置信度）

        Returns:
            融合后的天气分类
        """
        n_samples = len(ci_weather)
        final_weather = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            if ci_weather[i] == wsi_weather[i]:
                # 两个指标一致，直接采用
                final_weather[i] = ci_weather[i]
            else:
                # 不一致，使用加权决策
                # 计算CI置信度（距离阈值的距离）
                ci_conf = self._calculate_confidence(ci_values[i], self.ci_thresholds)

                # 计算WSI置信度
                wsi_conf = self._calculate_confidence(wsi_values[i], self.wsi_thresholds)

                # 加权投票
                ci_vote = ci_weather[i] * self.fusion_weights['ci'] * ci_conf
                wsi_vote = wsi_weather[i] * self.fusion_weights['wsi'] * wsi_conf

                # 如果加权后仍然不明确，使用软决策
                if abs(ci_vote - wsi_vote) < 0.1:
                    # 倾向于中间类别（多云）
                    final_weather[i] = 1
                else:
                    # 选择权重更高的
                    final_weather[i] = ci_weather[i] if ci_vote > wsi_vote else wsi_weather[i]

        return final_weather

    def _calculate_confidence(self, value: float, thresholds: List[float]) -> float:
        """
        计算分类置信度

        基于值与阈值的距离计算置信度

        Args:
            value: 指标值
            thresholds: 分类阈值

        Returns:
            置信度 [0, 1]
        """
        # 计算到最近阈值的距离
        distances = [abs(value - th) for th in thresholds]
        min_distance = min(distances)

        # 归一化距离到置信度
        # 距离越远，置信度越高
        confidence = min(1.0, min_distance / 0.3)  # 0.3为归一化因子

        return confidence

    def update_fusion_weights(self, error_feedback: float, learning_rate: float = 0.01):
        """
        基于误差反馈更新融合权重（在线学习）

        Args:
            error_feedback: 预测误差反馈
            learning_rate: 学习率
        """
        # 简单的梯度下降更新
        # 如果误差大，增加WSI权重（气象数据可能更可靠）
        if error_feedback > 0.1:  # 误差阈值
            delta = learning_rate * error_feedback

            # 更新权重
            self.fusion_weights['wsi'] = min(0.9, self.fusion_weights['wsi'] + delta)
            self.fusion_weights['ci'] = 1 - self.fusion_weights['wsi']

            logger.info(f"融合权重更新: CI={self.fusion_weights['ci']:.3f}, "
                       f"WSI={self.fusion_weights['wsi']:.3f}")

    def get_weather_statistics(self, weather_types: np.ndarray) -> Dict:
        """
        统计天气类型分布

        Args:
            weather_types: 天气类型数组

        Returns:
            统计信息字典
        """
        weather_arr = np.asarray(weather_types, dtype=int)
        weather_arr = weather_arr[weather_arr >= 0]
        if weather_arr.size == 0:
            return {
                'distribution': {},
                'percentages': {},
                'transitions': 0
            }

        unique, counts = np.unique(weather_arr, return_counts=True)
        total = weather_arr.size

        stats = {
            'distribution': {},
            'percentages': {},
            'transitions': 0
        }

        for weather_id, count in zip(unique, counts):
            weather_name = self.weather_types[weather_id]
            stats['distribution'][weather_name] = int(count)
            stats['percentages'][weather_name] = float(count / total)

        # 计算天气转换次数
        if len(weather_types) > 1:
            transitions = np.sum(np.diff(weather_types) != 0)
            stats['transitions'] = int(transitions)

        return stats


def test_weather_classifier():
    """测试天气分类器"""
    # 创建模拟数据
    n_samples = 100
    timestamps = pd.date_range('2024-01-01 06:00', periods=n_samples, freq='15T')

    # 模拟天气数据
    np.random.seed(42)
    data = pd.DataFrame({
        'irradiance': np.random.uniform(0, 800, n_samples),
        'temperature': np.random.uniform(10, 30, n_samples),
        'pressure': np.random.uniform(1000, 1020, n_samples),
        'humidity': np.random.uniform(20, 80, n_samples)
    }, index=timestamps)

    # 初始化分类器
    classifier = WeatherClassifier(location_lat=38.5, location_lon=105.0)

    # 执行分类
    weather_types = classifier.classify(data)

    # 统计结果
    stats = classifier.get_weather_statistics(weather_types)
    print(f"天气分类统计: {stats}")

    return classifier, weather_types


if __name__ == "__main__":
    # 运行测试
    test_weather_classifier()
