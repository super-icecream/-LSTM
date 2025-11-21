"""
可视化分析模块
提供丰富的性能可视化和分析工具
支持静态图表和交互式可视化
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from .metrics import MetricsResult
import torch
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# 设置中文字体支持（尝试多个字体，按优先级）
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Zen Hei',  # 文泉驿正黑
    'WenQuanYi Micro Hei',  # 文泉驿微米黑
    'Noto Sans CJK SC',  # Noto中文字体
    'SimHei',  # 黑体（Windows）
    'DejaVu Sans'  # 备用西文字体
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置Seaborn风格
sns.set_style("whitegrid")
sns.set_palette("husl")


class PerformanceVisualizer:
    """
    性能可视化分析工具
    
    提供丰富的可视化方法：
    - 静态图表（matplotlib/seaborn）
    - 交互式图表（plotly）
    - 实时监控图表
    """
    
    def __init__(self, 
                 style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 100,
                 chinese_font: bool = True):
        """
        初始化可视化器
        
        Args:
            style: 绘图风格（支持 matplotlib 内置样式）
            figsize: 默认图表大小
            dpi: 图像分辨率
            chinese_font: 是否使用中文字体
        """
        self._set_plot_style(style)
        self.figsize = figsize
        self.dpi = dpi
        
        # 验证中文字体是否可用
        if chinese_font:
            self._verify_chinese_font()
        
        # 配色方案
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#51A3A3',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6C969D'
        }
    
    def _set_plot_style(self, style: str):
        """
        设置绘图样式，支持自动降级
        
        Args:
            style: 期望的绘图样式
        """
        try:
            plt.style.use(style)
            return
        except OSError:
            pass

        fallback_styles = [
            'seaborn-v0_8-whitegrid',
            'seaborn-v0_8',
            'ggplot',
            'bmh',
            'fivethirtyeight',
            'tableau-colorblind10',
            'default',
        ]

        for fallback in fallback_styles:
            try:
                plt.style.use(fallback)
                warnings.warn(
                    f"样式 '{style}' 不可用，已降级使用 '{fallback}'",
                    UserWarning,
                )
                return
            except OSError:
                continue

        warnings.warn(
            f"所有样式设置失败，使用 matplotlib 默认样式",
            UserWarning,
        )
    
    def _verify_chinese_font(self):
        """
        验证中文字体是否可用，如果不可用则提供安装建议
        """
        import matplotlib.font_manager as fm
        import logging
        
        # 获取所有可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 检查是否有可用的中文字体
        chinese_fonts = [
            'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei',
            'Noto Sans CJK SC',
            'SimHei',
            'Microsoft YaHei'
        ]
        
        has_chinese_font = any(font in available_fonts for font in chinese_fonts)
        
        if not has_chinese_font:
            warnings.warn(
                "未检测到可用的中文字体！图表中的中文可能显示为方块。\n"
                "建议执行以下命令安装中文字体：\n"
                "  apt-get update && apt-get install -y fonts-wqy-zenhei\n"
                "安装后需要清除matplotlib字体缓存：\n"
                "  rm -rf ~/.cache/matplotlib",
                UserWarning
            )
        else:
            # 找到第一个可用的中文字体
            used_font = next((f for f in chinese_fonts if f in available_fonts), None)
            if used_font:
                logger = logging.getLogger(__name__)
                logger.info(f"✓ 使用中文字体: {used_font}")
        
    def plot_prediction_comparison(self,
                                  predictions: np.ndarray,
                                  targets: np.ndarray,
                                  timestamps: Optional[pd.DatetimeIndex] = None,
                                  title: str = "光伏功率预测对比",
                                  save_path: Optional[str] = None,
                                  show_metrics: bool = True) -> plt.Figure:
        """
        绘制预测值vs真实值对比曲线
        
        特点：
        - 双Y轴显示绝对误差
        - 标注关键时间点（日出、正午、日落）
        - 高亮显示误差较大区域
        """
        # 创建时间戳
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=len(targets), freq='5T')
        
        # 创建图表
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1]+2))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
        
        # 主图：预测vs真实
        ax1 = fig.add_subplot(gs[0])
        
        # 绘制曲线
        ax1.plot(timestamps, targets, 'b-', label='真实值', alpha=0.8, linewidth=1.5)
        ax1.plot(timestamps, predictions, 'r--', label='预测值', alpha=0.8, linewidth=1.5)
        
        # 填充误差区域
        ax1.fill_between(timestamps, targets, predictions, 
                         where=(predictions > targets), 
                         color='red', alpha=0.2, label='高估区域')
        ax1.fill_between(timestamps, targets, predictions,
                         where=(predictions <= targets),
                         color='blue', alpha=0.2, label='低估区域')
        
        # 标注关键时间点
        if len(timestamps) >= 288:  # 一天的数据点（5分钟间隔）
            sunrise_idx = 72   # 6:00 AM
            noon_idx = 144     # 12:00 PM
            sunset_idx = 216   # 6:00 PM
            
            ax1.axvline(x=timestamps[sunrise_idx], color='orange', 
                       linestyle=':', alpha=0.5, label='日出')
            ax1.axvline(x=timestamps[noon_idx], color='gold', 
                       linestyle=':', alpha=0.5, label='正午')
            ax1.axvline(x=timestamps[sunset_idx], color='purple', 
                       linestyle=':', alpha=0.5, label='日落')
        
        ax1.set_ylabel('功率 (kW)', fontsize=11)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', ncol=3, fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 添加性能指标
        if show_metrics:
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))
            mae = np.mean(np.abs(predictions - targets))
            
            metrics_text = f'RMSE: {rmse:.2f} kW | MAE: {mae:.2f} kW'
            ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 子图1：误差曲线
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        errors = predictions - targets
        ax2.plot(timestamps, errors, 'g-', alpha=0.6, linewidth=1)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.fill_between(timestamps, 0, errors, 
                         where=(errors > 0), color='red', alpha=0.3)
        ax2.fill_between(timestamps, 0, errors,
                         where=(errors <= 0), color='blue', alpha=0.3)
        ax2.set_ylabel('误差 (kW)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 子图2：相对误差百分比
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        relative_errors = np.where(targets != 0, 
                                  (errors / targets) * 100, 0)
        ax3.plot(timestamps, relative_errors, 'purple', alpha=0.6, linewidth=1)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_ylabel('相对误差 (%)', fontsize=10)
        ax3.set_xlabel('时间', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 隐藏中间图的x轴标签
        plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)
        plt.setp(ax2.xaxis.get_majorticklabels(), visible=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
            
        return fig
    
    def plot_error_distribution(self,
                               errors: np.ndarray,
                               title: str = "误差分布分析",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制误差分布图
        
        包含：
        - 直方图
        - 核密度估计
        - 正态分布拟合
        - Q-Q图
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. 直方图+KDE
        ax1 = fig.add_subplot(gs[0, :2])
        
        # 绘制直方图
        n, bins, patches = ax1.hist(errors, bins=50, density=True, 
                                    alpha=0.6, color=self.colors['primary'],
                                    edgecolor='black', label='误差分布')
        
        # 添加KDE曲线
        from scipy import stats
        kde = stats.gaussian_kde(errors)
        x_range = np.linspace(errors.min(), errors.max(), 200)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE曲线')
        
        # 添加正态分布拟合
        mu, sigma = errors.mean(), errors.std()
        ax1.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 
                'g--', linewidth=2, label=f'正态拟合\nμ={mu:.2f}, σ={sigma:.2f}')
        
        ax1.set_xlabel('误差值 (kW)', fontsize=11)
        ax1.set_ylabel('概率密度', fontsize=11)
        ax1.set_title('误差分布直方图', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 箱线图
        ax2 = fig.add_subplot(gs[0, 2])
        box = ax2.boxplot(errors, vert=True, patch_artist=True,
                         medianprops=dict(color='red', linewidth=2),
                         boxprops=dict(facecolor=self.colors['info'], alpha=0.7))
        ax2.set_ylabel('误差值 (kW)', fontsize=11)
        ax2.set_title('箱线图', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加统计信息
        stats_text = f'中位数: {np.median(errors):.2f}\n' \
                    f'Q1: {np.percentile(errors, 25):.2f}\n' \
                    f'Q3: {np.percentile(errors, 75):.2f}\n' \
                    f'IQR: {np.percentile(errors, 75) - np.percentile(errors, 25):.2f}'
        ax2.text(0.5, 0.02, stats_text, transform=ax2.transAxes,
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Q-Q图
        ax3 = fig.add_subplot(gs[1, 0])
        stats.probplot(errors, dist="norm", plot=ax3)
        ax3.set_title('Q-Q图', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 误差时序图
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(errors, color=self.colors['secondary'], alpha=0.6, linewidth=0.8)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.axhline(y=mu, color='r', linestyle=':', alpha=0.5, label=f'均值={mu:.2f}')
        ax4.axhline(y=mu+2*sigma, color='orange', linestyle=':', alpha=0.5, label=f'±2σ')
        ax4.axhline(y=mu-2*sigma, color='orange', linestyle=':', alpha=0.5)
        ax4.set_xlabel('样本索引', fontsize=11)
        ax4.set_ylabel('误差值 (kW)', fontsize=11)
        ax4.set_title('误差时序图', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. 累积分布函数
        ax5 = fig.add_subplot(gs[1, 2])
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
        ax5.plot(sorted_errors, cumulative, color=self.colors['success'], linewidth=2)
        ax5.set_xlabel('误差值 (kW)', fontsize=11)
        ax5.set_ylabel('累积概率', fontsize=11)
        ax5.set_title('累积分布函数', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 添加参考线
        for percentile in [25, 50, 75]:
            value = np.percentile(errors, percentile)
            ax5.axvline(x=value, color='gray', linestyle=':', alpha=0.5)
            ax5.text(value, 0.05, f'{percentile}%', rotation=90, fontsize=9)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
            
        return fig
    
    def plot_scatter_regression(self,
                              predictions: np.ndarray,
                              targets: np.ndarray,
                              title: str = "回归分析",
                              interactive: bool = True) -> Union[go.Figure, plt.Figure]:
        """
        散点图和回归分析
        
        Args:
            predictions: 预测值
            targets: 真实值
            title: 图表标题
            interactive: 是否生成交互式图表
        """
        if interactive:
            # 交互式Plotly图表
            df = pd.DataFrame({
                '真实值': targets.flatten(),
                '预测值': predictions.flatten(),
                '绝对误差': np.abs(predictions.flatten() - targets.flatten())
            })
            
            fig = px.scatter(df, x='真实值', y='预测值',
                           color='绝对误差',
                           color_continuous_scale='RdYlBu_r',
                           hover_data={'绝对误差': ':.2f'},
                           title=title,
                           labels={'color': '绝对误差 (kW)'})
            
            # 添加理想预测线
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='理想预测线',
                                    line=dict(color='black', dash='dash', width=2)))
            
            # 添加线性回归线
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(targets.reshape(-1, 1), predictions.flatten())
            y_pred = lr.predict(np.array([[min_val], [max_val]]))
            fig.add_trace(go.Scatter(x=[min_val, max_val],
                                    y=y_pred.flatten(),
                                    mode='lines',
                                    name='拟合线',
                                    line=dict(color='red', width=2)))
            
            # 更新布局
            fig.update_layout(
                xaxis_title="真实值 (kW)",
                yaxis_title="预测值 (kW)",
                hovermode='closest',
                showlegend=True,
                width=900,
                height=700
            )
            
            return fig
        
        else:
            # 静态matplotlib图表
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # 计算误差用于着色
            errors = np.abs(predictions - targets)
            
            # 绘制散点图
            scatter = ax.scatter(targets, predictions, 
                               c=errors, cmap='RdYlBu_r',
                               alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('绝对误差 (kW)', fontsize=10)
            
            # 添加理想预测线
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'k--', linewidth=2, label='理想预测线', alpha=0.7)
            
            # 添加线性回归线
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(targets.reshape(-1, 1), predictions)
            y_pred = lr.predict(targets.reshape(-1, 1))
            
            ax.plot(targets, y_pred, 'r-', linewidth=2, 
                   label='拟合线', alpha=0.7)
            
            # 设置标签和标题
            ax.set_xlabel('真实值 (kW)', fontsize=11)
            ax.set_ylabel('预测值 (kW)', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 添加文本信息
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))
            mae = np.mean(np.abs(predictions - targets))
            info_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}'
            ax.text(0.95, 0.05, info_text, transform=ax.transAxes,
                   fontsize=10, ha='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            return fig
    
    def plot_multi_horizon_comparison(self,
                                     results: Dict,
                                     title: str = "多时间尺度性能对比") -> go.Figure:
        """
        多时间尺度性能对比图（雷达图）
        """
        # 准备数据
        horizons = list(results.keys())
        metrics_names = ['RMSE', 'MAE', 'NRMSE']
        
        fig = go.Figure()
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, horizon in enumerate(horizons):
            if hasattr(results[horizon], 'to_dict'):
                metrics_dict = results[horizon].to_dict()
            else:
                metrics_dict = results[horizon]
            
            # 归一化指标到0-1范围
            values = []
            for metric in metrics_names:
                if metric in metrics_dict:
                    # 对于误差指标，值越小越好
                    values.append(1 / (1 + metrics_dict[metric]))
                else:
                    values.append(0)
            
            # 闭合雷达图
            values.append(values[0])
            metrics_names_closed = metrics_names + [metrics_names[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_names_closed,
                fill='toself',
                fillcolor=colors[i % len(colors)],
                opacity=0.3,
                name=f'{horizon}步预测',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            showlegend=True,
            title=dict(
                text=title,
                font=dict(size=16, color='black'),
                x=0.5,
                xanchor='center'
            ),
            legend=dict(
                x=1.1,
                y=1,
                font=dict(size=11)
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_weather_performance_heatmap(self,
                                        performance_df: pd.DataFrame,
                                        title: str = "分天气类型性能热力图",
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        天气类型性能热力图
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 确保数据格式正确
        if 'weather_type' in performance_df.index.names:
            heatmap_data = performance_df
        else:
            # 假设performance_df已经是透视表格式
            heatmap_data = performance_df
        
        # 对指标进行归一化（可选）
        # 将误差指标转换为性能分数（越小越好的指标反转）
        display_data = heatmap_data.copy()
        for col in display_data.columns:
            if col in ['RMSE', 'MAE', 'NRMSE']:
                # 误差指标归一化并反转
                display_data[col] = 1 / (1 + display_data[col])
        
        # 绘制热力图
        sns.heatmap(display_data, annot=True, fmt='.3f',
                   cmap='RdYlGn', center=0.5,
                   cbar_kws={'label': '性能分数'},
                   square=True, linewidths=1,
                   ax=ax)
        
        # 设置标签
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('评估指标', fontsize=11)
        ax.set_ylabel('天气类型', fontsize=11)
        
        # 旋转标签
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
            
        return fig
    
    def plot_weather_performance_bar(self,
                                     per_weather: Dict[str, Dict[str, float]],
                                     metrics: Optional[List[str]] = None,
                                     title: str = "分天气性能对比",
                                     save_path: Optional[str] = None) -> plt.Figure:
        metrics = metrics or ['RMSE', 'MAE']
        weather_types = list(per_weather.keys())
        x = np.arange(len(weather_types))
        bar_width = 0.8 / max(len(metrics), 1)
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] / 1.5), dpi=self.dpi)
        for idx, metric in enumerate(metrics):
            values = [per_weather[w].get(metric, np.nan) for w in weather_types]
            offset = (idx - (len(metrics) - 1) / 2) * bar_width
            ax.bar(x + offset, values, width=bar_width, label=metric)
            for i, value in enumerate(values):
                if not np.isnan(value):
                    ax.text(x[i] + offset, value, f"{value:.2f}", ha='center', va='bottom', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(weather_types, fontsize=11)
        ax.set_ylabel('指标数值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def create_realtime_dashboard(self,
                                  predictions: np.ndarray,
                                  targets: np.ndarray,
                                  metrics_dict: Optional[Dict[str, float]] = None,
                                  weather_distribution: Optional[Dict[str, int]] = None,
                                  multi_horizon: Optional[Dict[int, Dict[str, float]]] = None) -> go.Figure:
        predictions = np.asarray(predictions).reshape(-1)
        targets = np.asarray(targets).reshape(-1)
        errors = predictions - targets
        timesteps = np.arange(len(targets))

        metrics_dict = metrics_dict or {}
        rmse_val = float(metrics_dict.get('RMSE', np.sqrt(np.mean(errors ** 2))))
        mae_val = float(metrics_dict.get('MAE', np.mean(np.abs(errors))))
        range_den = float(np.max(targets) - np.min(targets)) if len(targets) > 0 else 0.0
        nrmse_val = float(metrics_dict.get('NRMSE', rmse_val / max(range_den, 1e-8)))

        fig = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=(
                '功率对比', '误差趋势', 'RMSE',
                '散点回归', '误差分布', 'MAE',
                '多时间尺度', '天气占比', 'NRMSE'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'indicator'}],
                [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'indicator'}],
                [{'type': 'bar'}, {'type': 'pie'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        fig.add_trace(go.Scatter(x=timesteps, y=targets, mode='lines', name='真实', line=dict(color=self.colors['primary'])), row=1, col=1)
        fig.add_trace(go.Scatter(x=timesteps, y=predictions, mode='lines', name='预测', line=dict(color=self.colors['secondary'], dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=timesteps, y=errors, mode='lines', name='误差', line=dict(color=self.colors['warning'])), row=1, col=2)

        fig.add_trace(go.Indicator(mode="gauge+number", value=rmse_val,
                                   title={'text': "RMSE (kW)"},
                                   gauge={'axis': {'range': [0, max(1.0, rmse_val * 1.5)]},
                                          'bar': {'color': "darkblue"}}),
                      row=1, col=3)

        fig.add_trace(go.Scatter(x=targets, y=predictions, mode='markers',
                                 marker=dict(color=np.abs(errors), colorscale='RdYlBu_r', showscale=True, size=6,
                                             colorbar=dict(title='绝对误差')), name='样本'), row=2, col=1)
        min_val, max_val = float(np.min(targets)), float(np.max(targets))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='理想线',
                                 line=dict(color='black', dash='dash')), row=2, col=1)

        fig.add_trace(go.Histogram(x=errors, nbinsx=40, marker_color=self.colors['info'], opacity=0.7, name='误差分布'),
                      row=2, col=2)

        fig.add_trace(go.Indicator(mode="gauge+number", value=mae_val,
                                   title={'text': "MAE (kW)"},
                                   gauge={'axis': {'range': [0, max(1.0, mae_val * 1.5)]},
                                          'bar': {'color': "darkgreen"}}),
                      row=2, col=3)

        if multi_horizon:
            horizon_labels = [f"{h}步" for h in multi_horizon.keys()]
            rmse_scores = [multi_horizon[h].get('RMSE', 0) for h in multi_horizon.keys()]
            fig.add_trace(go.Bar(x=horizon_labels, y=rmse_scores, marker_color=self.colors['success'], name='RMSE'),
                          row=3, col=1)
        else:
            fig.add_trace(go.Bar(x=['N/A'], y=[0], marker_color=self.colors['success'], name='RMSE'), row=3, col=1)

        if weather_distribution:
            fig.add_trace(go.Pie(labels=list(weather_distribution.keys()),
                                 values=list(weather_distribution.values()),
                                 hole=.35,
                                 textinfo='label+percent'), row=3, col=2)
        else:
            fig.add_trace(go.Pie(labels=['N/A'], values=[1]), row=3, col=2)

        fig.add_trace(go.Indicator(mode="gauge+number", value=max(0.0, nrmse_val),
                                   title={'text': "NRMSE"},
                                   gauge={'axis': {'range': [0, max(1.0, nrmse_val * 1.5)]},
                                          'bar': {'color': "darkred"}}),
                      row=3, col=3)

        fig.update_layout(title_text="DLFE-LSTM-WSI 预测监控", showlegend=False, height=900, width=1400)
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=11)
        return fig

    def save_all_figures(self,
                        results: Dict,
                        predictions: np.ndarray,
                        targets: np.ndarray,
                        output_dir: str = './experiments/results/'):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        overall_metrics = results.get('overall')
        per_weather = results.get('per_weather') or {}
        multi_horizon = results.get('multi_horizon') or {}
        weather_distribution = results.get('weather_distribution') or {}

        def as_dict(metric_obj):
            if metric_obj is None:
                return {}
            if hasattr(metric_obj, 'to_dict'):
                return metric_obj.to_dict()
            return metric_obj

        overall_dict = as_dict(overall_metrics)
        per_weather_dict = {k: as_dict(v) for k, v in per_weather.items()}
        multi_horizon_dict = {k: as_dict(v) for k, v in multi_horizon.items()}

        print("开始生成评估图表...")
        fig1 = self.plot_prediction_comparison(predictions, targets, save_path=os.path.join(output_dir, f'prediction_comparison_{timestamp}.png'))
        plt.close(fig1)

        errors = predictions - targets
        fig2 = self.plot_error_distribution(errors, save_path=os.path.join(output_dir, f'error_distribution_{timestamp}.png'))
        plt.close(fig2)

        fig3 = self.plot_scatter_regression(predictions, targets, interactive=False)
        fig3.savefig(os.path.join(output_dir, f'scatter_regression_{timestamp}.png'), dpi=self.dpi, bbox_inches='tight')
        plt.close(fig3)

        if per_weather_dict:
            fig_weather = self.plot_weather_performance_bar(per_weather_dict, save_path=os.path.join(output_dir, f'weather_performance_{timestamp}.png'))
            plt.close(fig_weather)

        if multi_horizon_dict:
            fig4 = self.plot_multi_horizon_comparison(multi_horizon_dict)
            fig4.write_html(os.path.join(output_dir, f'multi_horizon_{timestamp}.html'))

        fig5 = self.create_realtime_dashboard(
            predictions=predictions,
            targets=targets,
            metrics_dict=overall_dict,
            weather_distribution=weather_distribution,
            multi_horizon=multi_horizon_dict
        )
        fig5.write_html(os.path.join(output_dir, f'dashboard_{timestamp}.html'))

        index_content = f"""
        # 评估报告图表索引
        生成时间: {timestamp}

        ## 静态图表
        - prediction_comparison_{timestamp}.png - 预测对比图
        - error_distribution_{timestamp}.png - 误差分布图  
        - scatter_regression_{timestamp}.png - 散点回归图
        - weather_performance_{timestamp}.png - 分天气性能对比

        ## 交互式图表
        - multi_horizon_{timestamp}.html - 多时间尺度对比
        - dashboard_{timestamp}.html - 监控仪表板
        """
        with open(os.path.join(output_dir, 'index.md'), 'w', encoding='utf-8') as f:
            f.write(index_content)
        print("图表生成完成!")


# 综合评估器类
class ModelEvaluator:
    """
    模型评估器：整合指标计算和可视化
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        config = config or {}
        
        self.metrics = PerformanceMetrics(
            device=config.get('device', 'cuda'),
            prated=config.get('prated')
        )
        self.visualizer = PerformanceVisualizer(
            figsize=config.get('figsize', (12, 8)),
            dpi=config.get('dpi', 100)
        )
        self.config = config
        
    def comprehensive_evaluation(self,
                                model: torch.nn.Module,
                                test_loader: torch.utils.data.DataLoader,
                                save_results: bool = True,
                                output_dir: str = './experiments/results/') -> Dict:
        """
        综合评估流程
        
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            save_results: 是否保存结果
            output_dir: 输出目录
            
        Returns:
            dict: 评估结果
        """
        print("="*50)
        print("开始综合评估...")
        print("="*50)
        
        # 收集预测和真实值
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(batch_data, (tuple, list)):
                    features, targets = batch_data
                else:
                    features = batch_data['features']
                    targets = batch_data['targets']
                
                # GPU推理
                if torch.cuda.is_available():
                    features = features.cuda()
                    targets = targets.cuda()
                
                predictions = model(features)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 合并结果
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 计算指标
        metrics_result = self.metrics.calculate_all_metrics(
            all_predictions, all_targets
        )
        
        print(f"\n评估指标:")
        print(metrics_result)
        
        # 生成可视化
        if save_results:
            self.visualizer.save_all_figures(
                {'overall': metrics_result, 'per_weather': {}, 'multi_horizon': {}, 'weather_distribution': {}},
                all_predictions,
                all_targets,
                output_dir
            )
        
        # 生成报告
        if save_results:
            self.generate_report(
                metrics_result,
                output_path=os.path.join(output_dir, 'evaluation_report.html')
            )
        
        return {
            'metrics': metrics_result,
            'predictions': all_predictions,
            'targets': all_targets
        }
    


    
    
        
    
    def generate_report(self,
                       results,
                       output_path: str = './experiments/results/report.html'):
        """
        生成HTML格式的评估报告（仅保留 RMSE/MAE/NRMSE）
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DLFE-LSTM-WSI 评估报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2E86AB; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric-good { color: green; font-weight: bold; }
                .metric-bad { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>DLFE-LSTM-WSI 模型评估报告</h1>
            <h2>性能指标汇总</h2>
            <table>
                <tr><th>指标</th><th>数值</th><th>评价</th></tr>
                <tr><td>RMSE</td><td>{rmse:.4f}</td><td class="{rmse_class}">{rmse_eval}</td></tr>
                <tr><td>MAE</td><td>{mae:.4f}</td><td class="{mae_class}">{mae_eval}</td></tr>
                <tr><td>NRMSE</td><td>{nrmse:.4f}</td><td class="{nrmse_class}">{nrmse_eval}</td></tr>
            </table>
            <h2>结论</h2>
            <p>{conclusion}</p>
            <p>报告生成时间: {timestamp}</p>
        </body>
        </html>
        """

        def evaluate_metric(value, thresholds):
            if value < thresholds[0]:
                return 'metric-good', '优秀'
            elif value < thresholds[1]:
                return 'metric-normal', '良好'
            else:
                return 'metric-bad', '需改进'

        thresholds = {
            'RMSE': [20, 30],
            'MAE': [15, 25],
            'NRMSE': [0.1, 0.2],
        }

        rmse_class, rmse_eval = evaluate_metric(results.rmse, thresholds['RMSE'])
        mae_class, mae_eval = evaluate_metric(results.mae, thresholds['MAE'])
        nrmse_class, nrmse_eval = evaluate_metric(results.nrmse, thresholds['NRMSE'])

        if results.rmse < thresholds['RMSE'][0] and results.mae < thresholds['MAE'][0]:
            conclusion = "模型表现优秀，预测精度高，可以投入实际应用。"
        elif results.rmse < thresholds['RMSE'][1] and results.mae < thresholds['MAE'][1]:
            conclusion = "模型表现良好，预测精度满足要求，建议进一步优化。"
        else:
            conclusion = "模型需要改进，建议调整超参数或增加训练数据。"

        html_content = html_template.format(
            rmse=results.rmse,
            rmse_class=rmse_class,
            rmse_eval=rmse_eval,
            mae=results.mae,
            mae_class=mae_class,
            mae_eval=mae_eval,
            nrmse=results.nrmse,
            nrmse_class=nrmse_class,
            nrmse_eval=nrmse_eval,
            conclusion=conclusion,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"评估报告已生成: {output_path}")


def generate_markdown_report(output_path: Union[str, Path],
                             overall: MetricsResult,
                             per_weather: Dict[str, MetricsResult],
                             multi_horizon: Dict[int, MetricsResult],
                             significance: Dict[str, Dict[str, float]],
                             weather_distribution: Dict[str, int]) -> None:
    header_lines: List[str] = []
    body_lines: List[str] = []
    header_lines.append("# DLFE-LSTM-WSI 评估摘要")
    header_lines.append("")
    header_lines.append("## 整体指标")
    overall_dict = overall.to_dict() if hasattr(overall, "to_dict") else overall
    for name, value in overall_dict.items():
        if name == "CI_95%":
            header_lines.append(f"- {name}: {value[0]:.4f} ~ {value[1]:.4f}")
        else:
            header_lines.append(f"- {name}: {value:.4f}")
    header_lines.append("")

    if per_weather:
        body_lines.append("## 分天气表现")
        for weather, metrics in per_weather.items():
            metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics
            metrics_text = ", ".join(
                f"{key}={metrics_dict[key]:.4f}" if key != "CI_95%"
                else f"{key}={metrics_dict[key][0]:.4f}~{metrics_dict[key][1]:.4f}"
                for key in metrics_dict
            )
            body_lines.append(f"- {weather}: {metrics_text}")
        body_lines.append("")

    if multi_horizon:
        body_lines.append("## 多时间尺度结果")
        for horizon, metrics in multi_horizon.items():
            metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics
            metrics_text = ", ".join(
                f"{key}={metrics_dict[key]:.4f}" if key != "CI_95%"
                else f"{key}={metrics_dict[key][0]:.4f}~{metrics_dict[key][1]:.4f}"
                for key in metrics_dict
            )
            body_lines.append(f"- {horizon}步: {metrics_text}")
        body_lines.append("")

    if significance:
        body_lines.append("## 显著性检验")
        for pair, result in significance.items():
            body_lines.append(
                f"- {pair}: p={result['p_value']:.4f}, Cohens_d={result['cohens_d']:.4f}, 结论={result['conclusion']}"
            )
        body_lines.append("")

    if weather_distribution:
        body_lines.append("## 样本分布")
        total = sum(weather_distribution.values())
        for weather, count in weather_distribution.items():
            ratio = count / total if total > 0 else 0
            body_lines.append(f"- {weather}: {count} ({ratio:.1%})")
        body_lines.append("")

    content_lines = header_lines + body_lines
    Path(output_path).write_text("\n".join(content_lines), encoding="utf-8")

