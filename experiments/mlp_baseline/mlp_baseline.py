"""MLP Baseline 模型定义"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from model.base import VectorModel


class MLPBaseline(VectorModel):
    """
    MLP Baseline 模型
    
    结构：
    - 输入层: 6 → 64（自动处理3维输入的padding）
    - 隐藏层: 64 → 32
    - 输出层: 32 → 3
    """
    
    def __init__(self):
        super().__init__(input_dim=6, output_dim=3)
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 处理不同长度的输入（3维或6维）
        if x.shape[-1] == 3:
            # 如果是3维输入，padding到6维
            x = torch.cat([x, torch.zeros_like(x)], dim=-1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
