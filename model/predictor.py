"""预测器基类"""

from typing import Dict, Optional, Tuple


# 固定半径（像素值）
MAP_RADII = {
    "mp_rr_district": {"ring1": 4930, "ring2": 2419, "ring3": 1488},
    "mp_rr_tropic": {"ring1": 4894, "ring2": 2407, "ring3": 1284}
}


class Predictor:
    """
    预测器基类
    
    职责：封装模型推理逻辑，处理坐标归一化/反归一化
    
    模型输出：[dx, dy] (2维，不预测半径)
    半径使用固定值：MAP_RADII[map_name]
    
    每个实验需要继承此类并实现predict方法
    """
    
    def __init__(self, device: str = "cuda", grid_size: int = 16384):
        """
        初始化预测器
        
        Args:
            device: 设备
            grid_size: 地图网格大小（用于坐标归一化）
        """
        self.device = device
        self.grid_size = grid_size
    
    def predict(self, map_name: str, ring1_data: Dict, ring2_data: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        预测Ring2和Ring3
        
        Args:
            map_name: 地图名称
            ring1_data: {"x": x1, "y": y1, "r": r1}
                原始像素坐标
            ring2_data: Ring2数据（可选）
                - 如果为None，预测Ring2和Ring3
                - 如果提供，直接返回Ring2，只预测Ring3
                原始像素坐标
        
        Returns:
            (ring2_dict, ring3_dict): 两个字典，格式为 {"x": x, "y": y, "r": r}
                原始像素坐标，半径使用固定值 MAP_RADII[map_name]
        """
        raise NotImplementedError("子类必须实现predict方法")
    
    def get_fixed_radius(self, map_name: str, ring_level: int) -> float:
        """
        获取固定半径
        
        Args:
            map_name: 地图名称
            ring_level: 圈级别 (1, 2, 3)
            
        Returns:
            半径值（像素）
        """
        if map_name not in MAP_RADII:
            raise ValueError(f"未知地图: {map_name}")
        
        ring_key = f"ring{ring_level}"
        if ring_key not in MAP_RADII[map_name]:
            raise ValueError(f"未知圈级别: {ring_level}")
        
        return MAP_RADII[map_name][ring_key]
