"""模型评估器"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


# ==================== 配置区域 ====================
# 获取项目根目录（evaluator.py在model/目录下，需要向上一级）
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "use"
GRID_SIZE = 16384

# 排除的地图
EXCLUDED_MAPS = ["mp_rr_desertlands_hu"]

# 固定半径（像素值）
MAP_RADII = {
    "mp_rr_district": {"ring1": 4930, "ring2": 2419, "ring3": 1488},
    "mp_rr_tropic": {"ring1": 4894, "ring2": 2407, "ring3": 1284}
}
# ==================================================


class Evaluator:
    """
    模型评估器
    
    功能：
    计算评估指标（使用 test.json）
    """
    
    def __init__(
        self,
        predictor = None,
        model: nn.Module = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        data_dir: Path = DATA_DIR
    ):
        """
        初始化评估器
        
        Args:
            predictor: 预测器（优先使用）
            model: 模型（如果未提供predictor，则从model创建默认predictor）
            device: 设备
            data_dir: 数据目录（Path对象）
        """
        self.device = device
        self.data_dir = Path(data_dir)
        self.grid_size = GRID_SIZE
        
        # 创建predictor
        if predictor is not None:
            self.predictor = predictor
        elif model is not None:
            # 如果只提供model，要求用户必须提供predictor
            raise ValueError("请为模型创建对应的Predictor并传入predictor参数")
        else:
            raise ValueError("必须提供predictor")
        
        # 保持向后兼容
        self.model = model if model is not None else getattr(predictor, 'model', None)
    

    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        评估模型（使用 test.json）
        
        按场景分别评估：
        - 场景1：只提供Ring1 → 预测Ring2和Ring3
        - 场景2：提供Ring1+Ring2 → 预测Ring3
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            评估指标字典
        """
        
        # 获取原始数据
        test_data_path = self.data_dir / "test.json"
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        # 过滤排除的地图
        filtered_data = [item for item in test_data if item.get("map") not in EXCLUDED_MAPS]
        
        # 分场景收集预测结果
        scenario1_ring2_preds = []  # 场景1: Ring2预测
        scenario1_ring2_targets = []
        scenario1_ring1_positions = []  # Ring1位置（用于计算Ring2的相对指标）
        
        scenario1_ring3_preds = []  # 场景1: Ring3预测（基于预测的Ring2）
        scenario1_ring3_targets = []
        scenario1_ring2_pred_positions = []  # 预测的Ring2位置（用于计算Ring3的相对指标）
        
        scenario2_ring3_preds = []  # 场景2: Ring3预测（基于真实Ring2）
        scenario2_ring3_targets = []
        scenario2_ring2_positions = []  # 真实Ring2位置（用于计算Ring3的相对指标）
        
        scenario1_maps = []
        scenario2_maps = []
        
        with torch.no_grad():
            for data_idx, item in enumerate(filtered_data):
                rings = item.get("rings", [])
                if len(rings) < 3:
                    continue
                
                map_name = item.get("map", "unknown")
                
                # 提取原始坐标
                x1, y1, r1 = rings[0]["x"], rings[0]["y"], rings[0]["r"]
                x2, y2, r2 = rings[1]["x"], rings[1]["y"], rings[1]["r"]
                x3, y3, r3 = rings[2]["x"], rings[2]["y"], rings[2]["r"]
                
                # 准备输入数据（原始坐标）
                ring1_data = {"x": x1, "y": y1, "r": r1}
                ring2_true_data = {"x": x2, "y": y2, "r": r2}
                
                # 场景1：只提供Ring1，预测Ring2和Ring3
                ring2_pred_data, ring3_pred_data = self.predictor.predict(map_name, ring1_data)
                
                # 归一化用于计算（模型内部使用归一化坐标）
                ring2_pred = torch.tensor([
                    ring2_pred_data["x"] / self.grid_size,
                    ring2_pred_data["y"] / self.grid_size,
                    ring2_pred_data["r"] / self.grid_size
                ], dtype=torch.float32)
                ring2_true = torch.tensor([x2 / self.grid_size, y2 / self.grid_size, r2 / self.grid_size], dtype=torch.float32)
                ring3_pred = torch.tensor([
                    ring3_pred_data["x"] / self.grid_size,
                    ring3_pred_data["y"] / self.grid_size,
                    ring3_pred_data["r"] / self.grid_size
                ], dtype=torch.float32)
                ring3_true = torch.tensor([x3 / self.grid_size, y3 / self.grid_size, r3 / self.grid_size], dtype=torch.float32)
                
                scenario1_ring2_preds.append(ring2_pred)
                scenario1_ring2_targets.append(ring2_true)
                scenario1_ring1_positions.append(torch.tensor([x1 / self.grid_size, y1 / self.grid_size], dtype=torch.float32))
                
                scenario1_ring3_preds.append(ring3_pred)
                scenario1_ring3_targets.append(ring3_true)
                scenario1_ring2_pred_positions.append(torch.tensor([
                    ring2_pred_data["x"] / self.grid_size,
                    ring2_pred_data["y"] / self.grid_size
                ], dtype=torch.float32))
                scenario1_maps.append(map_name)
                
                # 场景2：提供Ring1+真实Ring2，预测Ring3
                _, ring3_pred_s2_data = self.predictor.predict(map_name, ring1_data, ring2_true_data)
                ring3_pred_s2 = torch.tensor([
                    ring3_pred_s2_data["x"] / self.grid_size,
                    ring3_pred_s2_data["y"] / self.grid_size,
                    ring3_pred_s2_data["r"] / self.grid_size
                ], dtype=torch.float32)
                
                scenario2_ring3_preds.append(ring3_pred_s2)
                scenario2_ring3_targets.append(ring3_true)
                scenario2_ring2_positions.append(torch.tensor([x2 / self.grid_size, y2 / self.grid_size], dtype=torch.float32))
                scenario2_maps.append(map_name)
        
        # 转换为张量
        scenario1_ring2_preds = torch.stack(scenario1_ring2_preds)
        scenario1_ring2_targets = torch.stack(scenario1_ring2_targets)
        scenario1_ring1_positions = torch.stack(scenario1_ring1_positions)
        
        scenario1_ring3_preds = torch.stack(scenario1_ring3_preds)
        scenario1_ring3_targets = torch.stack(scenario1_ring3_targets)
        scenario1_ring2_pred_positions = torch.stack(scenario1_ring2_pred_positions)
        
        scenario2_ring3_preds = torch.stack(scenario2_ring3_preds)
        scenario2_ring3_targets = torch.stack(scenario2_ring3_targets)
        scenario2_ring2_positions = torch.stack(scenario2_ring2_positions)
        
        # 计算指标（传入前一个Ring的位置）
        scenario1_ring2_metrics = self._compute_metrics(
            scenario1_ring2_preds, scenario1_ring2_targets, scenario1_ring1_positions
        )
        scenario1_ring3_metrics = self._compute_metrics(
            scenario1_ring3_preds, scenario1_ring3_targets, scenario1_ring2_pred_positions
        )
        scenario2_ring3_metrics = self._compute_metrics(
            scenario2_ring3_preds, scenario2_ring3_targets, scenario2_ring2_positions
        )
        
        # 按地图计算指标
        scenario1_ring2_by_map = self._compute_metrics_by_map(
            scenario1_ring2_preds, scenario1_ring2_targets, scenario1_maps
        )
        scenario1_ring3_by_map = self._compute_metrics_by_map(
            scenario1_ring3_preds, scenario1_ring3_targets, scenario1_maps
        )
        scenario2_ring3_by_map = self._compute_metrics_by_map(
            scenario2_ring3_preds, scenario2_ring3_targets, scenario2_maps
        )
        
        return {
            "scenario_1_only_ring1": {
                "ring2_error": scenario1_ring2_metrics,
                "ring3_error": scenario1_ring3_metrics,
                "by_map": {
                    "ring2_error": scenario1_ring2_by_map,
                    "ring3_error": scenario1_ring3_by_map,
                }
            },
            "scenario_2_ring1_and_ring2": {
                "ring3_error": scenario2_ring3_metrics,
                "by_map": {
                    "ring3_error": scenario2_ring3_by_map,
                }
            }
        }
    
    def _compute_metrics(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor,
        prev_positions: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            preds: 预测值 (N, 3) - [x, y, r] 归一化坐标 (0-1)
            targets: 真实值 (N, 3) - [x, y, r] 归一化坐标 (0-1)
            prev_positions: 前一个Ring的位置 (N, 2) - [x_prev, y_prev]，用于计算相对位置指标
            
        Returns:
            指标字典（center_distance 为像素值，其他为归一化值）
        """
        # 圆心距离误差 (像素)
        center_pred = preds[:, :2]
        center_target = targets[:, :2]
        center_distance = torch.sqrt(((center_pred - center_target) ** 2).sum(dim=1)).mean().item()
        center_distance_px = center_distance * self.grid_size
        
        # 各维度误差 (归一化)
        x_error = (preds[:, 0] - targets[:, 0]).abs().mean().item()
        y_error = (preds[:, 1] - targets[:, 1]).abs().mean().item()
        
        # MSE/MAE/RMSE 只计算位置 (x, y)
        position_preds = preds[:, :2]
        position_targets = targets[:, :2]
        mse = ((position_preds - position_targets) ** 2).mean().item()
        mae = (position_preds - position_targets).abs().mean().item()
        rmse = np.sqrt(mse)
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "center_distance": center_distance_px,  # 像素
            "x_error": x_error,
            "y_error": y_error,
        }
        
        # 计算相对位置指标（如果提供了前一个Ring的位置）
        if prev_positions is not None:
            # 真实方向向量
            dx_true = targets[:, 0] - prev_positions[:, 0]
            dy_true = targets[:, 1] - prev_positions[:, 1]
            
            # 预测方向向量
            dx_pred = preds[:, 0] - prev_positions[:, 0]
            dy_pred = preds[:, 1] - prev_positions[:, 1]
            
            # 角度误差（归一化到 0-1，180度对称性）
            angle_true = torch.atan2(dy_true, dx_true)
            angle_pred = torch.atan2(dy_pred, dx_pred)
            angle_diff = torch.abs(angle_pred - angle_true)
            
            # 考虑180度对称性：0度和180度等价
            # 将角度差归一化到 [0, π/2]，再除以 π/2 得到 [0, 1]
            angle_error = torch.min(angle_diff, 2 * np.pi - angle_diff)
            angle_error = torch.min(angle_error, torch.abs(np.pi - angle_error))
            angle_error_normalized = (angle_error / (np.pi / 2)).mean().item()
            
            # 距离误差比例（归一化到 0-1）
            dist_true = torch.sqrt(dx_true**2 + dy_true**2)
            dist_pred = torch.sqrt(dx_pred**2 + dy_pred**2)
            
            # 避免除零
            valid_mask = dist_true > 1e-6
            if valid_mask.sum() > 0:
                distance_ratio = dist_pred[valid_mask] / dist_true[valid_mask]
                distance_error_ratio = torch.abs(distance_ratio - 1.0).mean().item()
            else:
                distance_error_ratio = 0.0
            
            metrics["angle_error"] = angle_error_normalized
            metrics["distance_error_ratio"] = distance_error_ratio
        
        return metrics
    
    def _compute_metrics_by_map(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor, 
        maps: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        按地图计算评估指标
        
        Args:
            preds: 预测值 (N, 3)
            targets: 真实值 (N, 3)
            maps: 地图名称列表
            
        Returns:
            {map_name: metrics} 字典
        """
        from collections import defaultdict
        
        # 按地图分组
        map_data = defaultdict(lambda: {"preds": [], "targets": []})
        for i, map_name in enumerate(maps):
            if i < len(preds):
                map_data[map_name]["preds"].append(preds[i])
                map_data[map_name]["targets"].append(targets[i])
        
        # 计算每个地图的指标
        map_metrics = {}
        for map_name, data in map_data.items():
            if len(data["preds"]) > 0:
                map_preds = torch.stack(data["preds"])
                map_targets = torch.stack(data["targets"])
                map_metrics[map_name] = self._compute_metrics(map_preds, map_targets)
        
        return map_metrics
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        打印评估指标
        
        Args:
            metrics: 指标字典，包含两个场景的评估结果
        """
        scenario1 = metrics.get("scenario_1_only_ring1", {})
        scenario2 = metrics.get("scenario_2_ring1_and_ring2", {})
        
        ring2_metrics = scenario1.get("ring2_error", {})
        scenario1_ring3_metrics = scenario1.get("ring3_error", {})
        scenario2_ring3_metrics = scenario2.get("ring3_error", {})
        
        # 场景1
        print(f"\n{'='*70}")
        print("场景1：只提供 Ring1")
        print(f"{'='*70}")
        
        # Ring2 误差
        if ring2_metrics:
            print(f"\nRing2 预测误差:")
            print(f"  圆心距离误差: {ring2_metrics['center_distance']:.1f} px")
            print(f"  MSE:          {ring2_metrics['mse']:.6f}")
            print(f"  MAE:          {ring2_metrics['mae']:.6f}")
        
        # Ring3 误差（基于预测的Ring2）
        if scenario1_ring3_metrics:
            print(f"\nRing3 预测误差（基于预测的Ring2）:")
            print(f"  圆心距离误差: {scenario1_ring3_metrics['center_distance']:.1f} px")
            print(f"  MSE:          {scenario1_ring3_metrics['mse']:.6f}")
            print(f"  MAE:          {scenario1_ring3_metrics['mae']:.6f}")
        
        # 按地图展示
        by_map = scenario1.get("by_map", {})
        if by_map:
            ring2_by_map = by_map.get("ring2_error", {})
            ring3_by_map = by_map.get("ring3_error", {})
            
            if ring2_by_map or ring3_by_map:
                print(f"\n各地图详细结果:")
                for map_name in sorted(set(list(ring2_by_map.keys()) + list(ring3_by_map.keys()))):
                    print(f"\n  {map_name}:")
                    
                    if map_name in ring2_by_map:
                        r2_metrics = ring2_by_map[map_name]
                        print(f"    Ring2 圆心误差: {r2_metrics['center_distance']:.1f} px")
                    
                    if map_name in ring3_by_map:
                        r3_metrics = ring3_by_map[map_name]
                        print(f"    Ring3 圆心误差: {r3_metrics['center_distance']:.1f} px")
        
        # 场景2
        print(f"\n{'='*70}")
        print("场景2：提供 Ring1 + Ring2")
        print(f"{'='*70}")
        
        # Ring3 误差（基于真实Ring2）
        if scenario2_ring3_metrics:
            print(f"\nRing3 预测误差（基于真实Ring2）:")
            print(f"  圆心距离误差: {scenario2_ring3_metrics['center_distance']:.1f} px")
            print(f"  MSE:          {scenario2_ring3_metrics['mse']:.6f}")
            print(f"  MAE:          {scenario2_ring3_metrics['mae']:.6f}")
        
        # 按地图展示
        by_map = scenario2.get("by_map", {})
        if by_map:
            ring3_by_map = by_map.get("ring3_error", {})
            
            if ring3_by_map:
                print(f"\n各地图详细结果:")
                for map_name in sorted(ring3_by_map.keys()):
                    r3_metrics = ring3_by_map[map_name]
                    print(f"  {map_name} 圆心误差: {r3_metrics['center_distance']:.1f} px")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # 测试评估器
    print("测试评估器:\n")
    
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from model.base import VectorModel
    from model.dataset import get_dataloader
    import torch.nn as nn
    
    # 创建简单模型
    class SimpleMLP(VectorModel):
        def __init__(self):
            super().__init__(input_dim=6, output_dim=3)
            self.fc1 = nn.Linear(6, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 3)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 加载模型
    model = SimpleMLP()
    model.load_checkpoint("tests/temp/checkpoints/best_model.pth")
    
    # 创建评估器
    evaluator = Evaluator(model)
    
    # 评估
    test_loader = get_dataloader("test", batch_size=32)
    metrics = evaluator.evaluate(test_loader)
    evaluator.print_metrics(metrics)
