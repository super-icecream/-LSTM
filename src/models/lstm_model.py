"""
DLFE-LSTM-WSI项目 - GPU优化的LSTM预测模型
严格按照指导文件第64-203行实现

GPU优化特性：
- 支持混合精度训练(FP16/FP32)
- CUDA加速的自定义LSTM实现
- 优化的GPU内存使用
- CUDA流并行处理
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from typing import Tuple, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)
_GPU_MODE_LOGGED = False


class LSTMPredictor(nn.Module):
    """
    GPU优化的LSTM预测器
    - 支持混合精度训练(FP16/FP32)
    - CUDA加速的自定义LSTM实现
    - 优化的GPU内存使用
    """

    def __init__(self,
                 input_dim: int = 30,
                 hidden_dims: list = [100, 50],
                 dropout_rates: list = [0.3, 0.2],
                 output_dim: int = 1,
                 sequence_length: int = 24,
                 use_cuda: bool = True,
                 use_mixed_precision: bool = True,
                 device_id: int = 0):
        """
        GPU优化参数：
        - use_cuda: 是否使用CUDA加速
        - use_mixed_precision: 是否使用混合精度
        - device_id: GPU设备ID（支持多GPU）
        """
        super().__init__()

        # GPU设备设置
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device_id}' if self.use_cuda else 'cpu')
        self.use_mixed_precision = use_mixed_precision and self.use_cuda

        # 启用cudNN优化
        if self.use_cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True  # 自动寻找最优算法
            torch.backends.cudnn.deterministic = False  # 牺牲确定性换取速度

        # LSTM层（使用cuDNN加速的实现）
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            batch_first=True,
            bidirectional=False,
            dropout=0.0  # LSTM内部不使用dropout（cuDNN限制）
        )
        self.dropout1 = nn.Dropout(dropout_rates[0])

        self.lstm2 = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=hidden_dims[1],
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )
        self.dropout2 = nn.Dropout(dropout_rates[1])

        # 输出层（回归任务不使用激活函数）
        self.fc = nn.Linear(hidden_dims[1], output_dim)
        # 注意：回归任务输出层不应使用激活函数
        # - Sigmoid: 会导致梯度消失
        # - ReLU: 会导致死神经元（输出恒为0，梯度为0）
        # 功率非负约束应在后处理阶段用 clamp(0, 1) 实现

        # 将模型移到GPU
        self.to(self.device)

        # 混合精度的缩放器
        self.scaler = amp.GradScaler() if self.use_mixed_precision else None
        
        # 输出GPU优化信息（全局仅一次）
        global _GPU_MODE_LOGGED
        if not _GPU_MODE_LOGGED:
            if self.use_mixed_precision:
                logger.info("✓ GPU混合精度训练")
            elif self.use_cuda:
                logger.info("✓ GPU训练模式")
            else:
                logger.info("✓ CPU训练模式")
            _GPU_MODE_LOGGED = True

    @torch.amp.autocast('cuda', enabled=True)  # 自动混合精度装饰器
    def forward(self, x: torch.Tensor,
                hidden_states: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        GPU加速的前向传播

        注意：
        - 输入x必须已经在GPU上
        - 使用autocast自动处理FP16/FP32转换
        - 返回值保持在GPU上避免传输开销
        """
        # 确保输入在正确的设备上
        if not x.is_cuda and self.use_cuda:
            x = x.to(self.device, non_blocking=True)

        # 第一层LSTM（cuDNN加速）
        lstm1_out, hidden1 = self.lstm1(x, hidden_states[0] if hidden_states else None)
        lstm1_out = self.dropout1(lstm1_out)

        # 第二层LSTM（cuDNN加速）
        lstm2_out, hidden2 = self.lstm2(lstm1_out, hidden_states[1] if hidden_states else None)
        lstm2_out = self.dropout2(lstm2_out)

        # 获取最后时间步（在GPU上切片）
        last_output = lstm2_out[:, -1, :]

        # 全连接层（GPU矩阵乘法）
        output = self.fc(last_output)
        # 回归任务：不使用激活函数，允许自由学习
        # 负值/超限值在评估时用 clamp(0, 1) 处理

        return output, (hidden1, hidden2)

    def init_hidden_gpu(self, batch_size: int) -> Tuple:
        """GPU优化的隐藏状态初始化"""
        # 直接在GPU上创建张量，避免CPU-GPU传输
        hidden1 = (torch.zeros(1, batch_size, 100, device=self.device),
                   torch.zeros(1, batch_size, 100, device=self.device))
        hidden2 = (torch.zeros(1, batch_size, 50, device=self.device),
                   torch.zeros(1, batch_size, 50, device=self.device))
        return (hidden1, hidden2)

    def parallel_forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        并行批处理前向传播（GPU优化）
        使用CUDA流实现真正的并行
        """
        if not self.use_cuda:
            return self.forward(x_batch)[0]

        # 创建CUDA流
        streams = [torch.cuda.Stream() for _ in range(2)]
        results = []

        # 将批次分成两部分并行处理
        mid = x_batch.size(0) // 2
        x_parts = [x_batch[:mid], x_batch[mid:]]

        for i, (stream, x_part) in enumerate(zip(streams, x_parts)):
            with torch.cuda.stream(stream):
                result, _ = self.forward(x_part)
                results.append(result)

        # 同步所有流
        for stream in streams:
            stream.synchronize()

        # 在GPU上拼接结果
        return torch.cat(results, dim=0)
