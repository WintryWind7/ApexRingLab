"""MLP Baseline模型 - 使用One-Hot地图编码"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from model.base import VectorModel


class MLPBaseline(VectorModel):
    """
    MLP Baseline模型（One-Hot地图编码）
    
    输入维度: 9 ([map_onehot(2), x1, y1, r1, r2, r3, dx_prev, dy_prev])
    输出维度: 2 ([dx, dy])
    结构: 9 → 64 → 32 → 2
    """
    
    def __init__(self):
        super().__init__(input_dim=9, output_dim=2)
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 9:
            raise ValueError(f"MLPBaseline expects 9-dim input, got {x.shape[-1]}-dim")
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
