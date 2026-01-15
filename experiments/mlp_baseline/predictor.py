"""Baseline模型的Predictor实现"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from typing import Dict, Optional, Tuple
from model.predictor import Predictor


class BaselinePredictor(Predictor):
    """
    Baseline模型的预测器
    
    使用One-Hot编码区分地图的单一模型
    """
    
    # 地图到One-Hot的映射
    MAP_TO_ONEHOT = {
        "mp_rr_district": [1.0, 0.0],
        "mp_rr_tropic": [0.0, 1.0]
    }
    
    def __init__(self, model, device: str = "cuda"):
        """
        初始化Baseline预测器
        
        Args:
            model: MLPBaseline模型实例
            device: 设备
        """
        super().__init__(device)
        self.model = model.to(device)
        self.model.eval()
    
    def predict(self, map_name: str, ring1_data: Dict, ring2_data: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        预测Ring2和Ring3
        
        Args:
            map_name: 地图名称（用于生成One-Hot编码）
            ring1_data: {"x": x1, "y": y1, "r": r1}
                坐标已归一化（0-1范围）
            ring2_data: Ring2数据（可选）
                - 如果为None，预测Ring2和Ring3
                - 如果提供，直接返回Ring2，只预测Ring3
        
        Returns:
            (ring2_dict, ring3_dict): 两个字典，格式为 {"x": x, "y": y, "r": r}
                坐标为归一化的绝对坐标
        """
        # 获取One-Hot编码
        if map_name not in self.MAP_TO_ONEHOT:
            raise ValueError(f"未知地图: {map_name}，可用地图: {list(self.MAP_TO_ONEHOT.keys())}")
        
        map_onehot = torch.tensor(self.MAP_TO_ONEHOT[map_name], dtype=torch.float32).to(self.device)
        x1, y1, r1 = ring1_data["x"], ring1_data["y"], ring1_data["r"]
        
        with torch.no_grad():
            if ring2_data is None:
                # 场景1：只给Ring1，预测Ring2和Ring3
                
                # 预测Ring2（相对坐标）
                ring1 = torch.tensor([x1, y1, r1], dtype=torch.float32).to(self.device)
                input1 = torch.cat([ring1, torch.zeros(3).to(self.device), map_onehot]).unsqueeze(0)
                output1 = self.model(input1).cpu().numpy()[0]  # [dx2, dy2, r2]
                
                # 转换为绝对坐标
                x2 = x1 + output1[0]
                y2 = y1 + output1[1]
                r2 = output1[2]
                
                # 预测Ring3（相对Ring2）
                dx2, dy2 = output1[0], output1[1]
                ring2_rel = torch.tensor([dx2, dy2, r2], dtype=torch.float32).to(self.device)
                input2 = torch.cat([ring1, ring2_rel, map_onehot]).unsqueeze(0)
                output2 = self.model(input2).cpu().numpy()[0]  # [dx3, dy3, r3] 相对Ring2
                
                # 转换为绝对坐标
                x3 = x2 + output2[0]
                y3 = y2 + output2[1]
                r3 = output2[2]
                
                return (
                    {"x": float(x2), "y": float(y2), "r": float(r2)},
                    {"x": float(x3), "y": float(y3), "r": float(r3)}
                )
            
            else:
                # 场景2：给Ring1+Ring2，预测Ring3
                x2, y2, r2 = ring2_data["x"], ring2_data["y"], ring2_data["r"]
                
                # 计算Ring2相对Ring1的坐标
                dx2 = x2 - x1
                dy2 = y2 - y1
                
                # 预测Ring3（相对Ring2）
                ring1 = torch.tensor([x1, y1, r1], dtype=torch.float32).to(self.device)
                ring2_rel = torch.tensor([dx2, dy2, r2], dtype=torch.float32).to(self.device)
                input2 = torch.cat([ring1, ring2_rel, map_onehot]).unsqueeze(0)
                output2 = self.model(input2).cpu().numpy()[0]  # [dx3, dy3, r3] 相对Ring2
                
                # 转换为绝对坐标
                x3 = x2 + output2[0]
                y3 = y2 + output2[1]
                r3 = output2[2]
                
                # 返回输入的Ring2和预测的Ring3
                return (
                    {"x": float(x2), "y": float(y2), "r": float(r2)},
                    {"x": float(x3), "y": float(y3), "r": float(r3)}
                )
