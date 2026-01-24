"""深层MLP模型"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from model.base import VectorModel


class MLPDeep(VectorModel):
    """
    深层MLP模型
    
    输入维度: 9 ([map_onehot(2), x1, y1, r1, r2, r3, dx_prev, dy_prev])
    输出维度: 2 ([dx, dy])
    结构: 9 → 256 → 128 → 64 → 32 → 2
    """
    
    def __init__(self):
        super().__init__(input_dim=9, output_dim=2)
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 9:
            raise ValueError(f"MLPDeep expects 9-dim input, got {x.shape[-1]}-dim")
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class MLPVeryDeep(VectorModel):
    """
    超深层MLP模型 - 测试深度瓶颈
    
    输入维度: 9
    输出维度: 2
    结构: 9 → 512 → 256 → 256 → 128 → 128 → 64 → 32 → 2
    """
    
    def __init__(self):
        super().__init__(input_dim=9, output_dim=2)
        self.fc1 = nn.Linear(9, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 9:
            raise ValueError(f"MLPVeryDeep expects 9-dim input, got {x.shape[-1]}-dim")
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        return x
