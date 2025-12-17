# -*- coding: utf-8 -*-
"""
随机种子固定工具
确保实验可复现性
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    设置全局随机种子，确保实验可复现
    
    Args:
        seed: 随机种子
        deterministic: 是否启用确定性模式 (cudnn.deterministic=True, benchmark=False)
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU
    
    # 环境变量 (部分操作的哈希种子)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # cuDNN 确定性设置
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 非确定性模式，允许 benchmark 加速
        torch.backends.cudnn.deterministic = False
        # benchmark 由外部配置决定


def get_dataloader_generator(seed: int = 42) -> torch.Generator:
    """
    获取 DataLoader 的随机数生成器
    
    Args:
        seed: 随机种子
    
    Returns:
        torch.Generator
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker 初始化函数
    确保多进程数据加载的可复现性
    
    Args:
        worker_id: worker 进程 ID
    """
    # 每个 worker 使用不同但确定的种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_seed_info(seed: int, deterministic: bool = True) -> None:
    """
    打印种子设置信息
    
    Args:
        seed: 随机种子
        deterministic: 是否启用确定性模式
    """
    print(f"\n[随机种子设置]")
    print(f"  Seed: {seed}")
    print(f"  Deterministic: {deterministic}")
    print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
