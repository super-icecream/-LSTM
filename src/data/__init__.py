# -*- coding: utf-8 -*-
"""
数据处理模块
包含：数据读取、时间切分、滑窗生成、白天筛选
"""

from .loader import load_solar_data, DataLoadResult, print_load_summary
from .splitter import split_by_day, SplitResult, print_split_summary
from .window import generate_windows, WindowResult, WindowSample, WindowStats, print_window_summary
from .daylight import (
    add_daylight_flag, filter_daylight_rows, print_daylight_summary,
    DaylightResult, DaylightStats
)
