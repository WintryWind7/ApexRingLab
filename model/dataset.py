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
# ==================================================


class RingDataset(Dataset):
    """
    毒圈数据集 - 自回归模式
    
    每条原始数据包含 3 个圈 [ring1, ring2, ring3]
    生成 2 个训练样本：
    - 样本 1: input=ring1 (3,) → target=ring2 (3,)
    - 样本 2: input=ring1+ring2 (6,) → target=ring3 (3,)
    
    支持两种坐标模式：
    - 'absolute': 全绝对坐标（原始方案）
    - 'relative': Ring1绝对 + Ring2相对Ring1 + Ring3相对Ring2（默认）
    """
    
    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        data_dir: Path = DATA_DIR,
        normalize: bool = True,
        coordinate_mode: Literal["absolute", "relative"] = "relative"
    ):
        """
        初始化数据集
        
        Args:
            split: 数据集分割 (train/val/test)
            data_dir: 数据目录（Path对象）
            normalize: 是否归一化坐标
            coordinate_mode: 坐标模式 ('absolute' 或 'relative')
        """
        self.split = split
        self.normalize = normalize
        self.grid_size = GRID_SIZE
        self.coordinate_mode = coordinate_mode
        
        # 加载数据
        data_path = Path(data_dir) / f"{split}.json"
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        
        # 生成训练样本
        self.samples = self._generate_samples()
        
        print(f"加载 {split} 数据集 ({coordinate_mode}): {len(self.raw_data)} 条原始数据 → {len(self.samples)} 个训练样本")
    
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
            
            rings = item.get("rings", [])
            
            if len(rings) < 3:
                continue
            
            # 提取坐标
            x1, y1, r1 = rings[0]["x"], rings[0]["y"], rings[0]["r"]
            x2, y2, r2 = rings[1]["x"], rings[1]["y"], rings[1]["r"]
            x3, y3, r3 = rings[2]["x"], rings[2]["y"], rings[2]["r"]
            
            # 归一化
            if self.normalize:
                x1, y1, r1 = x1 / self.grid_size, y1 / self.grid_size, r1 / self.grid_size
                x2, y2, r2 = x2 / self.grid_size, y2 / self.grid_size, r2 / self.grid_size
                x3, y3, r3 = x3 / self.grid_size, y3 / self.grid_size, r3 / self.grid_size
            
            if self.coordinate_mode == "absolute":
                # 绝对坐标模式（原始方案）
                ring1 = [x1, y1, r1]
                ring2 = [x2, y2, r2]
                ring3 = [x3, y3, r3]
                
                # 样本 1: ring1 → ring2
                samples.append((ring1, ring2))
                
                # 样本 2: ring1+ring2 → ring3
                samples.append((ring1 + ring2, ring3))
            
            elif self.coordinate_mode == "relative":
                # 相对坐标模式
                # Ring1绝对，Ring2相对Ring1，Ring3相对Ring2
                
                # 计算相对坐标
                dx2, dy2 = x2 - x1, y2 - y1  # Ring2相对Ring1
                dx3, dy3 = x3 - x2, y3 - y2  # Ring3相对Ring2
                
                # 样本 1: ring1 → ring2 (相对坐标)
                # 输入: [x1, y1, r1, 0, 0, 0]
                # 输出: [dx2, dy2, r2]
                input1 = [x1, y1, r1, 0, 0, 0]
                target1 = [dx2, dy2, r2]
                samples.append((input1, target1))
                
                # 样本 2: ring1 + ring2 → ring3 (相对坐标)
                # 输入: [x1, y1, r1, dx2, dy2, r2]
                # 输出: [dx3, dy3, r3]
                input2 = [x1, y1, r1, dx2, dy2, r2]
                target2 = [dx3, dy3, r3]
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


def collate_fn(batch):
    """
    自定义 collate 函数，处理不同长度的输入
    
    Args:
        batch: [(input, target), ...] 列表
        
    Returns:
        (inputs, targets) - 批次张量
    """
    inputs, targets = zip(*batch)
    
    # 按输入长度分组
    inputs_3 = [x for x in inputs if len(x) == 3]
    inputs_6 = [x for x in inputs if len(x) == 6]
    
    # 如果混合了不同长度，需要 padding 或分开处理
    # 这里简单起见，只返回相同长度的
    if len(inputs_3) > 0 and len(inputs_6) > 0:
        # 混合 batch，使用 padding
        max_len = 6
        padded_inputs = []
        for x in inputs:
            if len(x) == 3:
                # padding 到 6
                padded = torch.cat([x, torch.zeros(3)])
            else:
                padded = x
            padded_inputs.append(padded)
        inputs = torch.stack(padded_inputs)
    else:
        inputs = torch.stack(list(inputs))
    
    targets = torch.stack(list(targets))
    
    return inputs, targets


def get_dataloader(
    split: Literal["train", "val", "test"] = "train",
    batch_size: int = 32,
    shuffle: bool = None,
    num_workers: int = 0,
    coordinate_mode: Literal["absolute", "relative"] = "relative",
    **kwargs
) -> DataLoader:
    """
    获取 DataLoader
    
    Args:
        split: 数据集分割
        batch_size: 批次大小
        shuffle: 是否打乱，默认 train=True, val/test=False
        num_workers: 数据加载线程数
        coordinate_mode: 坐标模式 ('absolute' 或 'relative')
        **kwargs: 其他 DataLoader 参数
        
    Returns:
        DataLoader 实例
    """
    # 默认 shuffle 设置
    if shuffle is None:
        shuffle = (split == "train")
    
    dataset = RingDataset(split=split, coordinate_mode=coordinate_mode)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
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
        print(f"输入形状: {inputs.shape}")  # (batch_size, 3 或 6)
        print(f"目标形状: {targets.shape}")  # (batch_size, 3)
        print(f"\n第一个样本:")
        print(f"  输入: {inputs[0]}")
        print(f"  目标: {targets[0]}")
        break
    
    # 统计信息
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_loader.dataset)} 个样本")
    print(f"  验证集: {len(get_dataloader('val').dataset)} 个样本")
    print(f"  测试集: {len(get_dataloader('test').dataset)} 个样本")
