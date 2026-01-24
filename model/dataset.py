"""毒圈数据集加载器 - 自回归模式"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Any, Literal


# ==================== 配置区域 ====================
# 获取项目根目录（dataset.py在model/目录下，需要向上一级）
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "use"
GRID_SIZE = 16384  # 坐标系大小，用于归一化

# 排除的地图
EXCLUDED_MAPS = ["mp_rr_desertlands_hu"]

# 地图到One-Hot索引的映射
MAP_TO_ONEHOT = {
    "mp_rr_district": [1.0, 0.0],
    "mp_rr_tropic": [0.0, 1.0]
}

# 固定半径
MAP_RADII = {
    "mp_rr_district": {"ring1": 4930, "ring2": 2419, "ring3": 1488},
    "mp_rr_tropic": {"ring1": 4894, "ring2": 2407, "ring3": 1284}
}
# ==================================================


class RingDataset(Dataset):
    """
    毒圈数据集 - 自回归模式
    
    每条原始数据包含 3 个圈 [ring1, ring2, ring3]
    生成 2 个训练样本：
    - 样本 1: input=ring1 (9,) → target=ring2 (2,)
    - 样本 2: input=ring1+ring2 (9,) → target=ring3 (2,)
    
    输入: [map_onehot(2), x1, y1, r1, r2, r3, dx_prev, dy_prev] = 9维
    输出: [dx, dy] = 2维（不预测半径，使用固定值）
    """
    
    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        data_dir: Path = DATA_DIR,
        normalize: bool = True,
        use_map_encoding: bool = True
    ):
        """
        初始化数据集
        
        Args:
            split: 数据集分割 (train/val/test)
            data_dir: 数据目录（Path对象）
            normalize: 是否归一化坐标
            use_map_encoding: 是否使用One-Hot地图编码（默认True）
        """
        self.split = split
        self.normalize = normalize
        self.grid_size = GRID_SIZE
        self.use_map_encoding = use_map_encoding
        
        # 加载数据
        data_path = Path(data_dir) / f"{split}.json"
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        
        # 生成训练样本
        self.samples = self._generate_samples()
        
        map_info = "with map encoding" if use_map_encoding else "without map encoding"
        print(f"加载 {split} 数据集 ({map_info}): {len(self.raw_data)} 条原始数据 → {len(self.samples)} 个训练样本")
    
    def _generate_samples(self) -> List[Tuple[List[float], List[float]]]:
        """
        生成训练样本（自回归）
        
        Returns:
            [(input, target), ...] 列表
        """
        samples = []
        
        for item in self.raw_data:
            # 过滤排除的地图
            map_name = item.get("map", "")
            if map_name in EXCLUDED_MAPS:
                continue
            
            # 获取One-Hot编码
            if self.use_map_encoding:
                if map_name not in MAP_TO_ONEHOT:
                    continue  # 跳过未知地图
                map_onehot = MAP_TO_ONEHOT[map_name]
            else:
                map_onehot = []
            
            # 获取固定半径
            if map_name not in MAP_RADII:
                continue
            radii = MAP_RADII[map_name]
            r1 = radii["ring1"]
            r2 = radii["ring2"]
            r3 = radii["ring3"]
            
            rings = item.get("rings", [])
            
            if len(rings) < 3:
                continue
            
            # 提取坐标
            x1, y1 = rings[0]["x"], rings[0]["y"]
            x2, y2 = rings[1]["x"], rings[1]["y"]
            x3, y3 = rings[2]["x"], rings[2]["y"]
            
            # 归一化
            if self.normalize:
                x1, y1, r1 = x1 / self.grid_size, y1 / self.grid_size, r1 / self.grid_size
                x2, y2, r2 = x2 / self.grid_size, y2 / self.grid_size, r2 / self.grid_size
                x3, y3, r3 = x3 / self.grid_size, y3 / self.grid_size, r3 / self.grid_size
            
            # 计算相对坐标
            dx2, dy2 = x2 - x1, y2 - y1
            dx3, dy3 = x3 - x2, y3 - y2
            
            # 样本1: Ring1 → Ring2
            # 输入: [map(2), x1, y1, r1, r2, r3, 0, 0]
            # 输出: [dx2, dy2]
            input1 = map_onehot + [x1, y1, r1, r2, r3, 0.0, 0.0]
            target1 = [dx2, dy2]
            samples.append((input1, target1))
            
            # 样本2: Ring1 + Ring2 → Ring3
            # 输入: [map(2), x1, y1, r1, r2, r3, dx2, dy2]
            # 输出: [dx3, dy3]
            input2 = map_onehot + [x1, y1, r1, r2, r3, dx2, dy2]
            target2 = [dx3, dy3]
            samples.append((input2, target2))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            (input, target) - 输入和目标张量
        """
        input_data, target_data = self.samples[idx]
        
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)
        
        return input_tensor, target_tensor


def get_dataloader(
    split: Literal["train", "val", "test"] = "train",
    batch_size: int = 32,
    shuffle: bool = None,
    num_workers: int = 0,
    use_map_encoding: bool = True,
    **kwargs
) -> DataLoader:
    """
    获取 DataLoader
    
    Args:
        split: 数据集分割
        batch_size: 批次大小
        shuffle: 是否打乱，默认 train=True, val/test=False
        num_workers: 数据加载线程数
        use_map_encoding: 是否使用One-Hot地图编码（默认True）
        **kwargs: 其他 DataLoader 参数
        
    Returns:
        DataLoader 实例
    """
    # 默认 shuffle 设置
    if shuffle is None:
        shuffle = (split == "train")
    
    dataset = RingDataset(
        split=split,
        use_map_encoding=use_map_encoding
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据集
    print("测试数据集加载:\n")
    
    # 加载训练集
    train_loader = get_dataloader("train", batch_size=4)
    
    # 查看一个 batch
    for inputs, targets in train_loader:
        print(f"输入形状: {inputs.shape}")  # (batch_size, 9)
        print(f"目标形状: {targets.shape}")  # (batch_size, 2)
        print(f"\n第一个样本:")
        print(f"  输入: {inputs[0]}")
        print(f"  目标: {targets[0]}")
        break
    
    # 统计信息
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_loader.dataset)} 个样本")
    print(f"  验证集: {len(get_dataloader('val').dataset)} 个样本")
    print(f"  测试集: {len(get_dataloader('test').dataset)} 个样本")
