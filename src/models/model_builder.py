"""
DLFE-LSTM-WSI项目 - GPU感知的模型构建器
严格按照指导文件第210-446行实现

GPU优化特性：
- 自动检测和分配GPU资源
- 支持多GPU训练策略
- 动态批大小优化
- 融合优化器支持
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from typing import Dict, List, Optional
import yaml
import psutil
try:
    import GPUtil
except ImportError:  # allow running without GPUtil
    GPUtil = None
from .lstm_model import LSTMPredictor


class ModelBuilder:
    """
    GPU感知的模型构建器
    - 自动检测和分配GPU资源
    - 支持多GPU训练策略
    - 动态批大小优化
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)

        # GPU资源检测和分配
        self.gpu_config = self._detect_gpu_resources()
        self.device = self._setup_device()

    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """加载配置文件"""
        if config_path is None:
            # 使用默认配置
            return {
                'input_dim': 30,
                'hidden_dims': [100, 50],
                'dropout_rates': [0.3, 0.2],
                'output_dim': 3,
                'sequence_length': 24
                # batch_size 交由 DataLoader 管理
            }
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)

    def _detect_gpu_resources(self) -> Dict:
        """检测GPU资源并返回配置"""
        gpu_config = {
            'available': torch.cuda.is_available(),
            'device_count': 0,
            'devices': [],
            'memory': [],
            'compute_capability': [],
            'optimal_batch_size': 64
        }

        if gpu_config['available']:
            gpu_config['device_count'] = torch.cuda.device_count()

            gpus = GPUtil.getGPUs() if GPUtil is not None else []
            for i in range(gpu_config['device_count']):
                gpu = gpus[i] if i < len(gpus) else None
                if gpu:
                    gpu_config['devices'].append(i)
                    gpu_config['memory'].append(gpu.memoryTotal)

                    capability = torch.cuda.get_device_capability(i)
                    gpu_config['compute_capability'].append(f"{capability[0]}.{capability[1]}")

            if gpu_config['memory']:
                min_memory = min(gpu_config['memory'])
                if min_memory > 16000:
                    gpu_config['optimal_batch_size'] = 256
                elif min_memory > 8000:
                    gpu_config['optimal_batch_size'] = 128
                elif min_memory > 4000:
                    gpu_config['optimal_batch_size'] = 64
                else:
                    gpu_config['optimal_batch_size'] = 32

        return gpu_config

    def _setup_device(self) -> torch.device:
        """设置计算设备，优先选择负载最低的GPU"""
        if not self.gpu_config['available']:
            print("警告: CUDA不可用，使用CPU训练")
            return torch.device('cpu')

        # 获取负载最低的GPU
        if self.gpu_config['device_count'] > 1 and GPUtil is not None:
            gpus = GPUtil.getGPUs()
            if gpus:
                min_load_idx = min(range(len(gpus)), key=lambda i: gpus[i].memoryUtil)
                device = torch.device(f'cuda:{min_load_idx}')
                print(f"选择GPU {min_load_idx} (负载最低)")
            else:
                device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:0')

        # 清理GPU缓存
        torch.cuda.empty_cache()

        return device

    def build_model(self, model_type: str = 'standard',
                   use_data_parallel: bool = False) -> nn.Module:
        """
        构建GPU优化的模型

        参数：
        - model_type: 模型类型
        - use_data_parallel: 是否使用数据并行（多GPU）
        """
        # 更新配置以使用GPU
        self.config.update({
            'use_cuda': self.gpu_config['available'],
            'use_mixed_precision': self.gpu_config['available'],
            'device_id': self.device.index if self.device.type == 'cuda' else 0
        })

        # 创建模型（过滤掉与模型无关的训练参数）
        lstm_params = {
            'input_dim': self.config.get('input_dim', 30),
            'hidden_dims': self.config.get('hidden_dims') or self.config.get('hidden_sizes', [100, 50]),
            'dropout_rates': self.config.get('dropout_rates', [0.3, 0.2]),
            'output_dim': self.config.get('output_dim', 1),
            'sequence_length': self.config.get('sequence_length', 24),
            'use_cuda': self.config.get('use_cuda', True),
            'use_mixed_precision': self.config.get('use_mixed_precision', True),
            'device_id': self.config.get('device_id', 0),
        }
        model = LSTMPredictor(**lstm_params)

        # 权重初始化（GPU优化）
        self._initialize_weights_gpu(model)

        # 多GPU封装
        if use_data_parallel and self.gpu_config['device_count'] > 1:
            print(f"使用{self.gpu_config['device_count']}个GPU进行数据并行训练")
            model = DataParallel(model, device_ids=self.gpu_config['devices'])

        # 编译模型以获得额外加速（PyTorch 2.0+）
        if hasattr(torch, 'compile') and self.gpu_config['available']:
            model = torch.compile(model, mode='reduce-overhead')

        return model

    def _initialize_weights_gpu(self, model: nn.Module):
        """GPU优化的权重初始化"""
        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        # 输入权重使用Xavier初始化
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        # 隐藏权重使用正交初始化（更稳定）
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        # 偏置初始化
                        nn.init.zeros_(param)
                        # 遗忘门偏置设为1
                        n = param.size(0)
                        param[n//4:n//2].fill_(1.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 输出层偏置初始化为 0.3（接近目标值均值），加速收敛
                    nn.init.constant_(m.bias, 0.3)

        model.apply(init_weights)

    def create_optimizer_gpu(self, model: nn.Module,
                           optimizer_type: str = 'AdamW',
                           lr: float = 0.001) -> torch.optim.Optimizer:
        """
        创建GPU优化的优化器

        支持：
        - 融合优化器（GPU加速）
        - 梯度累积
        - 混合精度优化
        """
        # 检查是否有融合优化器（NVIDIA Apex或原生支持）
        if optimizer_type == 'AdamW' and self.gpu_config['available']:
            # 尝试使用融合AdamW（更快）
            try:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.999),
                    weight_decay=1e-2,
                    fused=True  # GPU融合操作
                )
                print("使用融合AdamW优化器（GPU加速）")
            except:
                # 降级到标准AdamW
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=1e-2
                )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=1e-4
            )

        return optimizer

    def estimate_memory_usage(self, model: nn.Module,
                             batch_size: int,
                             seq_len: int = 24) -> Dict:
        """估算GPU内存使用"""
        # 模型参数内存
        param_memory = sum(p.numel() * p.element_size()
                          for p in model.parameters())

        # 梯度内存（训练时）
        grad_memory = param_memory

        # 激活值内存（估算）
        input_size = batch_size * seq_len * 30 * 4  # float32
        hidden_memory = batch_size * (100 + 50) * 4 * seq_len
        activation_memory = input_size + hidden_memory

        # 优化器状态内存（Adam约为参数的2倍）
        optimizer_memory = param_memory * 2

        total_memory = (param_memory + grad_memory +
                       activation_memory + optimizer_memory)

        return {
            'model_params_mb': param_memory / 1024 / 1024,
            'gradients_mb': grad_memory / 1024 / 1024,
            'activations_mb': activation_memory / 1024 / 1024,
            'optimizer_mb': optimizer_memory / 1024 / 1024,
            'total_mb': total_memory / 1024 / 1024,
            'recommended_batch_size': self._recommend_batch_size(total_memory)
        }

    def _recommend_batch_size(self, estimated_memory: float) -> int:
        """根据GPU内存推荐批大小"""
        if not self.gpu_config['available']:
            return 32

        available_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        free_memory = available_memory - used_memory

        # 保留20%安全边际
        usable_memory = free_memory * 0.8

        # 计算可容纳的批次数
        memory_per_sample = estimated_memory / self.config.get('batch_size', 64)
        recommended = int(usable_memory / memory_per_sample)

        # 对齐到2的幂次（更高效）
        recommended = 2 ** (recommended.bit_length() - 1)

        return max(16, min(recommended, 512))
