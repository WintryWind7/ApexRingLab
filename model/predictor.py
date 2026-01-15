"""预测器基类"""

from typing import Dict, Optional, Tuple


class Predictor:
    """
    预测器基类
    
    职责：封装模型推理逻辑，支持单模型或多模型决策系统
    
    每个实验需要继承此类并实现predict方法
    """
    
    def __init__(self, device: str = "cuda"):
        """
        初始化预测器
        
        Args:
            device: 设备
        """
        self.device = device
        self.grid_size = 16384
    
    def predict(self, map_name: str, ring1_data: Dict, ring2_data: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        预测Ring2和Ring3
        
        Args:
            map_name: 地图名称
            ring1_data: {"x": x1, "y": y1, "r": r1}
                坐标已归一化（0-1范围）
            ring2_data: Ring2数据（可选）
                - 如果为None，预测Ring2和Ring3
                - 如果提供，直接返回Ring2，只预测Ring3
        
        Returns:
            (ring2_dict, ring3_dict): 两个字典，格式为 {"x": x, "y": y, "r": r}
                坐标为归一化的绝对坐标
        """
        raise NotImplementedError("子类必须实现predict方法")
