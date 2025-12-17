# -*- coding: utf-8 -*-
"""
Global LSTM 模型
多步预测：输入 Lx 个时间步，输出 Hmax 个时间步的 Power_pu
"""

import torch
import torch.nn as nn
from typing import Optional


class GlobalLSTM(nn.Module):
    """
    Global LSTM 多步预测模型
    
    输入: (batch, Lx, n_features) - 历史序列
    输出: (batch, Hmax) - 未来 Hmax 步的 Power_pu 预测
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_steps: int = 16,
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM 隐藏层维度
            num_layers: LSTM 层数
            output_steps: 输出步数 (Hmax)
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 全连接输出层：将最后时刻的隐藏状态映射到 Hmax 步预测
        self.fc = nn.Linear(hidden_size, output_steps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, Lx, n_features) 输入序列
        
        Returns:
            (batch, Hmax) 预测输出
        """
        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一层的最后时刻隐藏状态
        # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # 全连接输出
        output = self.fc(last_hidden)  # (batch, Hmax)
        
        return output
    
    def __repr__(self) -> str:
        return (f"GlobalLSTM(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, "
                f"output_steps={self.output_steps})")
