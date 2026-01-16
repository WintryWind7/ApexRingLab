"""深层MLP的Predictor"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from typing import Dict, Optional, Tuple
from model.predictor import Predictor


class DeepMLPPredictor(Predictor):
    """深层MLP模型的预测器"""
    
    # 地图到One-Hot的映射
    MAP_TO_ONEHOT = {
        "mp_rr_district": [1.0, 0.0],
        "mp_rr_tropic": [0.0, 1.0]
    }
    
    def __init__(self, model, device: str = "cuda"):
        super().__init__(device)
        self.model = model.to(device)
        self.model.eval()
    
    def predict(self, map_name: str, ring1_data: Dict, ring2_data: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        预测Ring2和Ring3
        
        Args:
            map_name: 地图名称
            ring1_data: {"x": x1, "y": y1, "r": r1} (归一化坐标)
            ring2_data: Ring2数据（可选）
        
        Returns:
            (ring2_dict, ring3_dict): 绝对坐标
        """
        if map_name not in self.MAP_TO_ONEHOT:
            raise ValueError(f"未知地图: {map_name}")
        
        map_onehot = torch.tensor(self.MAP_TO_ONEHOT[map_name], dtype=torch.float32).to(self.device)
        x1, y1, r1 = ring1_data["x"], ring1_data["y"], ring1_data["r"]
        
        with torch.no_grad():
            if ring2_data is None:
                # 场景1：只给Ring1，预测Ring2和Ring3
                
                # 预测Ring2
                ring1 = torch.tensor([x1, y1, r1], dtype=torch.float32).to(self.device)
                input1 = torch.cat([map_onehot, ring1, torch.zeros(3).to(self.device)]).unsqueeze(0)
                output1 = self.model(input1).cpu().numpy()[0]
                
                dx2, dy2, r2 = output1[0], output1[1], output1[2]
                x2, y2 = x1 + dx2, y1 + dy2
                
                # 预测Ring3
                ring2_rel = torch.tensor([dx2, dy2, r2], dtype=torch.float32).to(self.device)
                input2 = torch.cat([map_onehot, ring1, ring2_rel]).unsqueeze(0)
                output2 = self.model(input2).cpu().numpy()[0]
                
                dx3, dy3, r3 = output2[0], output2[1], output2[2]
                x3, y3 = x2 + dx3, y2 + dy3
                
                return (
                    {"x": float(x2), "y": float(y2), "r": float(r2)},
                    {"x": float(x3), "y": float(y3), "r": float(r3)}
                )
            
            else:
                # 场景2：给Ring1+Ring2，预测Ring3
                x2, y2, r2 = ring2_data["x"], ring2_data["y"], ring2_data["r"]
                dx2, dy2 = x2 - x1, y2 - y1
                
                ring1 = torch.tensor([x1, y1, r1], dtype=torch.float32).to(self.device)
                ring2_rel = torch.tensor([dx2, dy2, r2], dtype=torch.float32).to(self.device)
                input2 = torch.cat([map_onehot, ring1, ring2_rel]).unsqueeze(0)
                output2 = self.model(input2).cpu().numpy()[0]
                
                dx3, dy3, r3 = output2[0], output2[1], output2[2]
                x3, y3 = x2 + dx3, y2 + dy3
                
                return (
                    {"x": float(x2), "y": float(y2), "r": float(r2)},
                    {"x": float(x3), "y": float(y3), "r": float(r3)}
                )
