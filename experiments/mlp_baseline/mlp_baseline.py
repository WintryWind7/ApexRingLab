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
    
    输入维度: 8 (6维坐标 + 2维One-Hot地图编码)
    输出维度: 3
    结构: 8 → 64 → 32 → 3
    """
    
    def __init__(self):
        super().__init__(input_dim=8, output_dim=3)
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 8:
            raise ValueError(f"MLPBaseline expects 8-dim input, got {x.shape[-1]}-dim")
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
