# -*- coding: utf-8 -*-
"""
DLFE-LSTM-WSI Evaluation Module
GPU-optimized performance evaluation and visualization components

This module provides comprehensive evaluation tools for the DLFE-LSTM-WSI
photovoltaic power prediction system, including:
- Performance metrics calculation with GPU acceleration
- Rich visualization capabilities (static and interactive)
- Multi-horizon and multi-weather evaluation support
- Real-time monitoring dashboard generation

Components:
    MetricsResult: Data class for storing evaluation metrics results
    PerformanceMetrics: GPU-accelerated metrics calculator
    PerformanceVisualizer: Comprehensive visualization toolkit
    ModelEvaluator: Integrated evaluation orchestrator

Usage Example:
    >>> from src.evaluation import ModelEvaluator
    >>> 
    >>> # Initialize evaluator with configuration
    >>> config = {'device': 'cuda', 'figsize': (12, 8)}
    >>> evaluator = ModelEvaluator(config)
    >>> 
    >>> # Run comprehensive evaluation
    >>> results = evaluator.comprehensive_evaluation(
    ...     model=trained_model,
    ...     test_loader=test_dataloader,
    ...     save_results=True,
    ...     output_dir='./results/'
    ... )
    >>> 
    >>> # Access metrics
    >>> print(f"RMSE: {results['metrics'].rmse:.4f}")
    >>> print(f"R²: {results['metrics'].r2:.4f}")

Author: DLFE-LSTM-WSI Development Team
Version: 1.0.0
License: MIT
"""

import sys
import warnings
from typing import Dict, Optional

# Version and metadata
__version__ = "1.0.0"
__author__ = "DLFE-LSTM-WSI Development Team"
__email__ = "dlfe-lstm-wsi@example.com"
__description__ = "GPU-optimized solar PV power prediction evaluation module"
__license__ = "MIT"

# Import core components with error handling
try:
    from .metrics import MetricsResult, PerformanceMetrics, export_metrics_bundle, export_weather_distribution
    from .visualizer import (
        PerformanceVisualizer,
        ModelEvaluator,
        generate_markdown_report,
    )
except ImportError as exc:
    print(f"Warning: evaluation module import failure: {exc}")
    print("Install dependencies: torch numpy pandas scipy matplotlib seaborn plotly")
    sys.exit(1)

# Public API exports
__all__ = [
    # Core classes
    'MetricsResult',
    'PerformanceMetrics',
    'PerformanceVisualizer',
    'ModelEvaluator',
    'generate_markdown_report',
    'export_metrics_bundle',
    'export_weather_distribution',
    
    # Module metadata
    '__version__',
    '__author__',
    '__description__',
    
    # Utility functions
    'check_environment',
    'get_evaluation_config'
]


def check_environment(verbose: bool = True) -> Dict[str, bool]:
    """
    Check evaluation module environment and dependencies
    
    Args:
        verbose: Whether to print detailed information
        
    Returns:
        Dict containing environment status flags
    """
    env_status = {
        'gpu_available': False,
        'cuda_version': None,
        'chinese_font': False,
        'plotly_available': False,
        'all_dependencies': False
    }
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            env_status['gpu_available'] = True
            env_status['cuda_version'] = torch.version.cuda
            
            if verbose:
                device_count = torch.cuda.device_count()
                print(f"┌─ GPU Environment ─────────────────────────────┐")
                print(f"│ Status: ✓ Enabled                             │")
                print(f"│ Devices: {device_count} GPU(s) detected                      │")
                print(f"│ Primary: {torch.cuda.get_device_name(0)[:30]:<30} │")
                print(f"│ CUDA: {torch.version.cuda:<40} │")
                print(f"└────────────────────────────────────────────────┘")
        else:
            if verbose:
                print(f"┌─ GPU Environment ─────────────────────────────┐")
                print(f"│ Status: ✗ CPU Mode                            │")
                print(f"│ Note: GPU acceleration unavailable            │")
                print(f"└────────────────────────────────────────────────┘")
    except ImportError:
        if verbose:
            print("Warning: PyTorch not installed")
    
    # Check matplotlib Chinese font support
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # Try multiple Chinese fonts
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        font_found = False
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                env_status['chinese_font'] = True
                font_found = font
                break
        
        if verbose:
            print(f"\n┌─ Visualization Environment ────────────────────┐")
            if font_found:
                print(f"│ Chinese Font: ✓ {font_found:<29} │")
            else:
                print(f"│ Chinese Font: ✗ Using default font            │")
    except ImportError:
        if verbose:
            print("│ Matplotlib: ✗ Not installed                   │")
    
    # Check Plotly availability
    try:
        import plotly
        env_status['plotly_available'] = True
        if verbose:
            print(f"│ Interactive Charts: ✓ Plotly {plotly.__version__:<17} │")
    except ImportError:
        if verbose:
            print(f"│ Interactive Charts: ✗ Plotly not installed    │")
    
    if verbose:
        print(f"└────────────────────────────────────────────────┘")
    
    # Check all dependencies
    env_status['all_dependencies'] = all([
        env_status['gpu_available'],
        env_status['chinese_font'],
        env_status['plotly_available']
    ])
    
    return env_status


def get_evaluation_config(device: Optional[str] = None) -> Dict:
    """
    Get default evaluation configuration
    
    Args:
        device: Computation device ('cuda', 'cpu', or None for auto-detect)
        
    Returns:
        Dictionary containing evaluation configuration
    """
    import torch
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        # Device configuration
        'device': device,
        
        # Metrics configuration
        'metrics': {
            'calculate_confidence': True,
            'confidence_level': 0.95,
            'n_bootstrap': 1000,
            'epsilon': 1e-8
        },
        
        # Visualization configuration
        'visualization': {
            'figsize': (12, 8),
            'dpi': 100,
            'style': 'seaborn',
            'save_formats': ['png', 'pdf'],
            'interactive': True
        },
        
        # Evaluation configuration
        'evaluation': {
            'batch_size': 64,
            'num_workers': 4,
            'pin_memory': True,
            'horizons': [1, 2, 4],  # 15min, 30min, 60min
            'weather_types': ['sunny', 'cloudy', 'overcast']
        },
        
        # Output configuration
        'output': {
            'save_results': True,
            'output_dir': './experiments/results/',
            'generate_report': True,
            'report_format': 'html'
        }
    }
    
    return config


# Module initialization message
def _initialize_module():
    """Initialize evaluation module and print status"""
    print("\n" + "="*50)
    print("DLFE-LSTM-WSI Evaluation Module v{}".format(__version__))
    print("="*50)
    
    # Check environment silently first
    env_status = check_environment(verbose=False)
    
    # Print summary
    status_icon = "✓" if env_status['all_dependencies'] else "⚠"
    print(f"\nModule Status: {status_icon}")
    
    print("\nSupported Metrics:")
    print("  • RMSE (Root Mean Square Error)")
    print("  • MAE (Mean Absolute Error)")
    print("  • NRMSE (Normalized RMSE)")
    print("  • R² (Coefficient of Determination)")
    print("  • MAPE (Mean Absolute Percentage Error)")
    
    print("\nSupported Features:")
    print("  • Multi-horizon evaluation (10/30/60 min)")
    print("  • Weather-specific performance analysis")
    print("  • Statistical significance testing")
    print("  • Bootstrap confidence intervals")
    print("  • Real-time monitoring dashboard")
    
    if env_status['gpu_available']:
        print(f"\n⚡ GPU Acceleration: ENABLED (CUDA {env_status['cuda_version']})")
    else:
        print("\n⚠ GPU Acceleration: DISABLED (CPU mode)")
    
    print("\nFor detailed environment check, run:")
    print("  >>> from src.evaluation import check_environment")
    print("  >>> check_environment(verbose=True)")
    print("\n" + "="*50 + "\n")


# Initialize module when imported
try:
    # Only show initialization in interactive mode or debug mode
    import os
    if os.environ.get('DLFE_DEBUG', '').lower() in ['true', '1', 'yes']:
        _initialize_module()
    elif hasattr(sys, 'ps1'):  # Interactive mode
        _initialize_module()
except:
    pass  # Silent fail for initialization message


# Convenience function for quick evaluation
def quick_evaluate(model, test_loader, device='cuda'):
    """
    Quick evaluation helper function
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Computation device
        
    Returns:
        Evaluation results dictionary
        
    Example:
        >>> from src.evaluation import quick_evaluate
        >>> results = quick_evaluate(model, test_loader)
        >>> print(f"Test RMSE: {results['metrics'].rmse:.4f}")
    """
    config = get_evaluation_config(device)
    evaluator = ModelEvaluator(config)
    return evaluator.comprehensive_evaluation(model, test_loader)


# Register custom matplotlib style for consistent plotting
def _register_custom_style():
    """Register DLFE-LSTM-WSI custom plotting style"""
    try:
        import matplotlib.pyplot as plt
        
        custom_style = {
            'figure.figsize': (12, 8),
            'figure.dpi': 100,
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.size': 10,
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 1,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        }
        
        # Register style
        plt.rcParams.update(custom_style)
        
    except ImportError:
        pass


# Apply custom style
_register_custom_style()
