"""GAN预测器"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from gan_model import Generator
from model.predictor import Predictor


class GANPredictor(Predictor):
    """GAN预测器 - 使用确定性预测（零噪声）"""
    
    def __init__(self, generator: Generator, device: str = "cuda", grid_size: int = 16384):
        super().__init__(device, grid_size)
        self.generator = generator.to(device)
        self.generator.eval()
    
    def predict(self, map_name: str, ring1_data: dict, ring2_data: dict = None) -> tuple:
        """预测Ring2和Ring3"""
        # 归一化输入
        x1 = ring1_data["x"] / self.grid_size
        y1 = ring1_data["y"] / self.grid_size
        r1 = ring1_data["r"] / self.grid_size
        
        if ring2_data is None:
            # 预测Ring2
            ring2 = self.predict_ring2((x1, y1, r1), map_name)
            x2, y2, r2 = ring2
            ring2_dict = {"x": int(x2 * self.grid_size), "y": int(y2 * self.grid_size), "r": int(r2 * self.grid_size)}
        else:
            # 使用提供的Ring2
            x2 = ring2_data["x"] / self.grid_size
            y2 = ring2_data["y"] / self.grid_size
            r2 = ring2_data["r"] / self.grid_size
            ring2_dict = ring2_data
        
        # 预测Ring3
        ring3 = self.predict_ring3((x1, y1, r1), (x2, y2, r2), map_name)
        x3, y3, r3 = ring3
        ring3_dict = {"x": int(x3 * self.grid_size), "y": int(y3 * self.grid_size), "r": int(r3 * self.grid_size)}
        
        return ring2_dict, ring3_dict
    
    def predict_ring2(self, ring1: tuple, map_name: str = None) -> tuple:
        x1, y1, r1 = ring1
        map_onehot = self._get_map_onehot(map_name)
        
        input_tensor = torch.tensor(
            [*map_onehot, x1, y1, r1, 0.0, 0.0, 0.0],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # 确定性预测（零噪声）
        with torch.no_grad():
            output = self.generator(input_tensor, deterministic=True).squeeze(0).cpu()
        
        dx2, dy2, r2 = output.tolist()
        x2, y2 = x1 + dx2, y1 + dy2
        
        return (x2, y2, r2)
    
    def predict_ring3(self, ring1: tuple, ring2: tuple, map_name: str = None) -> tuple:
        x1, y1, r1 = ring1
        x2, y2, r2 = ring2
        map_onehot = self._get_map_onehot(map_name)
        
        dx2, dy2 = x2 - x1, y2 - y1
        
        input_tensor = torch.tensor(
            [*map_onehot, x1, y1, r1, dx2, dy2, r2],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # 确定性预测（零噪声）
        with torch.no_grad():
            output = self.generator(input_tensor, deterministic=True).squeeze(0).cpu()
        
        dx3, dy3, r3 = output.tolist()
        x3, y3 = x2 + dx3, y2 + dy3
        
        return (x3, y3, r3)
    
    def _get_map_onehot(self, map_name: str) -> list:
        MAP_TO_ONEHOT = {
            "mp_rr_district": [1.0, 0.0],
            "mp_rr_tropic": [0.0, 1.0]
        }
        return MAP_TO_ONEHOT.get(map_name, [0.0, 0.0])
