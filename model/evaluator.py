"""模型评估器"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


# ==================== 配置区域 ====================
# 获取项目根目录（evaluator.py在model/目录下，需要向上一级）
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "use"
TEST_RINGS_DIR = PROJECT_ROOT / "data" / "use" / "test_rings"
MAP_DIR = PROJECT_ROOT / "data" / "map"
GRID_SIZE = 16384

# 排除的地图
EXCLUDED_MAPS = ["mp_rr_desertlands_hu"]

# 地图文件映射
MAP_FILES = {
    "mp_rr_desertlands_hu": "mp_rr_desertlands_hu.png",
    "mp_rr_district": "mp_rr_district.png",
    "mp_rr_tropic": "mp_rr_tropic_island_mu2.png",
}
# ==================================================


class Evaluator:
    """
    模型评估器
    
    功能：
    1. 计算评估指标（使用 test.json）
    2. 可视化预测结果（使用 test_rings/）
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        data_dir: Path = DATA_DIR,
        test_rings_dir: Path = TEST_RINGS_DIR,
        map_dir: Path = MAP_DIR,
        baseline_model_path: Optional[str] = None,
        coordinate_mode: str = "relative"
    ):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 设备
            data_dir: 数据目录（Path对象）
            test_rings_dir: 测试样本目录（Path对象）
            map_dir: 地图目录（Path对象）
            baseline_model_path: baseline模型路径（可选），用于对比评估
            coordinate_mode: 坐标模式 ('absolute' 或 'relative')
        """
        self.model = model.to(device)
        self.device = device
        self.data_dir = Path(data_dir)
        self.test_rings_dir = Path(test_rings_dir)
        self.map_dir = Path(map_dir)
        self.grid_size = GRID_SIZE
        self.baseline_model_path = baseline_model_path
        self.baseline_metrics = None
        self.coordinate_mode = coordinate_mode
        
        # 如果提供了baseline路径，加载并评估baseline
        if baseline_model_path:
            self._load_baseline_metrics()
    
    def _load_baseline_metrics(self) -> None:
        """
        加载baseline模型并评估，保存结果用于对比
        """
        from model.dataset import get_dataloader
        from experiments.mlp_baseline.mlp_baseline import MLPBaseline
        
        print(f"\n加载baseline模型用于对比: {self.baseline_model_path}")
        
        # 保存当前模型
        current_model = self.model
        
        # 加载baseline模型（固定使用MLPBaseline类）
        baseline_model = MLPBaseline()
        baseline_model.load_checkpoint(self.baseline_model_path)
        baseline_model = baseline_model.to(self.device)
        
        # 临时替换为baseline模型
        self.model = baseline_model
        
        # 评估baseline（不使用对比，避免递归）
        test_loader = get_dataloader("test", batch_size=32, shuffle=False)
        self.baseline_metrics = self.evaluate(test_loader)
        
        # 恢复当前模型
        self.model = current_model
        
        print(f"✓ Baseline评估完成\n")
    
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
        self.model.eval()
        
        # 获取原始数据
        test_data_path = self.data_dir / "test.json"
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        # 过滤排除的地图
        filtered_data = [item for item in test_data if item.get("map") not in EXCLUDED_MAPS]
        
        # 分场景收集预测结果
        scenario1_ring2_preds = []  # 场景1: Ring2预测
        scenario1_ring2_targets = []
        scenario1_ring3_preds = []  # 场景1: Ring3预测（基于预测的Ring2）
        scenario1_ring3_targets = []
        scenario2_ring3_preds = []  # 场景2: Ring3预测（基于真实Ring2）
        scenario2_ring3_targets = []
        
        scenario1_maps = []
        scenario2_maps = []
        
        with torch.no_grad():
            for data_idx, item in enumerate(filtered_data):
                rings = item.get("rings", [])
                if len(rings) < 3:
                    continue
                
                map_name = item.get("map", "unknown")
                
                # 提取并归一化坐标
                x1, y1, r1 = rings[0]["x"] / self.grid_size, rings[0]["y"] / self.grid_size, rings[0]["r"] / self.grid_size
                x2, y2, r2 = rings[1]["x"] / self.grid_size, rings[1]["y"] / self.grid_size, rings[1]["r"] / self.grid_size
                x3, y3, r3 = rings[2]["x"] / self.grid_size, rings[2]["y"] / self.grid_size, rings[2]["r"] / self.grid_size
                
                ring1 = torch.tensor([x1, y1, r1], dtype=torch.float32).to(self.device)
                
                if self.coordinate_mode == "absolute":
                    # 绝对坐标模式
                    ring2_true = torch.tensor([x2, y2, r2], dtype=torch.float32)
                    ring3_true = torch.tensor([x3, y3, r3], dtype=torch.float32)
                    
                    # 场景1：只提供Ring1
                    # 预测Ring2
                    input_ring1 = torch.cat([ring1, torch.zeros(3).to(self.device)])
                    ring2_pred = self.model(input_ring1.unsqueeze(0)).squeeze(0).cpu()
                    
                    scenario1_ring2_preds.append(ring2_pred)
                    scenario1_ring2_targets.append(ring2_true)
                    scenario1_maps.append(map_name)
                    
                    # 预测Ring3（使用预测的Ring2）
                    input_ring1_ring2_pred = torch.cat([ring1, ring2_pred.to(self.device)])
                    ring3_pred_from_pred = self.model(input_ring1_ring2_pred.unsqueeze(0)).squeeze(0).cpu()
                    
                    scenario1_ring3_preds.append(ring3_pred_from_pred)
                    scenario1_ring3_targets.append(ring3_true)
                    
                    # 场景2：提供Ring1+真实Ring2
                    input_ring1_ring2_true = torch.cat([ring1, ring2_true.to(self.device)])
                    ring3_pred_from_true = self.model(input_ring1_ring2_true.unsqueeze(0)).squeeze(0).cpu()
                    
                    scenario2_ring3_preds.append(ring3_pred_from_true)
                    scenario2_ring3_targets.append(ring3_true)
                    scenario2_maps.append(map_name)
                
                elif self.coordinate_mode == "relative":
                    # 相对坐标模式
                    # 场景1：只提供Ring1
                    # 预测Ring2（相对坐标）
                    input_ring1 = torch.cat([ring1, torch.zeros(3).to(self.device)])
                    ring2_pred_rel = self.model(input_ring1.unsqueeze(0)).squeeze(0).cpu()  # [dx2, dy2, r2]
                    
                    # 转换为绝对坐标用于评估
                    ring2_pred_abs = torch.tensor([
                        x1 + ring2_pred_rel[0].item(),
                        y1 + ring2_pred_rel[1].item(),
                        ring2_pred_rel[2].item()
                    ])
                    ring2_true_abs = torch.tensor([x2, y2, r2])
                    
                    scenario1_ring2_preds.append(ring2_pred_abs)
                    scenario1_ring2_targets.append(ring2_true_abs)
                    scenario1_maps.append(map_name)
                    
                    # 预测Ring3（相对Ring2）
                    input_ring1_ring2_pred = torch.cat([ring1, ring2_pred_rel.to(self.device)])
                    ring3_pred_rel = self.model(input_ring1_ring2_pred.unsqueeze(0)).squeeze(0).cpu()  # [dx3, dy3, r3] 相对Ring2
                    
                    # 转换为绝对坐标
                    ring3_pred_abs = torch.tensor([
                        ring2_pred_abs[0] + ring3_pred_rel[0].item(),
                        ring2_pred_abs[1] + ring3_pred_rel[1].item(),
                        ring3_pred_rel[2].item()
                    ])
                    ring3_true_abs = torch.tensor([x3, y3, r3])
                    
                    scenario1_ring3_preds.append(ring3_pred_abs)
                    scenario1_ring3_targets.append(ring3_true_abs)
                    
                    # 场景2：提供Ring1+真实Ring2
                    dx2_true, dy2_true = x2 - x1, y2 - y1
                    ring2_true_rel = torch.tensor([dx2_true, dy2_true, r2], dtype=torch.float32).to(self.device)
                    
                    input_ring1_ring2_true = torch.cat([ring1, ring2_true_rel])
                    ring3_pred_rel_s2 = self.model(input_ring1_ring2_true.unsqueeze(0)).squeeze(0).cpu()  # [dx3, dy3, r3] 相对Ring2
                    
                    # 转换为绝对坐标
                    ring3_pred_abs_s2 = torch.tensor([
                        x2 + ring3_pred_rel_s2[0].item(),
                        y2 + ring3_pred_rel_s2[1].item(),
                        ring3_pred_rel_s2[2].item()
                    ])
                    
                    scenario2_ring3_preds.append(ring3_pred_abs_s2)
                    scenario2_ring3_targets.append(ring3_true_abs)
                    scenario2_maps.append(map_name)
        
        # 转换为张量
        scenario1_ring2_preds = torch.stack(scenario1_ring2_preds)
        scenario1_ring2_targets = torch.stack(scenario1_ring2_targets)
        scenario1_ring3_preds = torch.stack(scenario1_ring3_preds)
        scenario1_ring3_targets = torch.stack(scenario1_ring3_targets)
        scenario2_ring3_preds = torch.stack(scenario2_ring3_preds)
        scenario2_ring3_targets = torch.stack(scenario2_ring3_targets)
        
        # 计算指标
        scenario1_ring2_metrics = self._compute_metrics(scenario1_ring2_preds, scenario1_ring2_targets)
        scenario1_ring3_metrics = self._compute_metrics(scenario1_ring3_preds, scenario1_ring3_targets)
        scenario2_ring3_metrics = self._compute_metrics(scenario2_ring3_preds, scenario2_ring3_targets)
        
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
    
    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            preds: 预测值 (N, 3)
            targets: 真实值 (N, 3)
            
        Returns:
            指标字典
        """
        # MSE
        mse = ((preds - targets) ** 2).mean().item()
        
        # MAE
        mae = (preds - targets).abs().mean().item()
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # 圆心距离误差
        center_pred = preds[:, :2]
        center_target = targets[:, :2]
        center_distance = torch.sqrt(((center_pred - center_target) ** 2).sum(dim=1)).mean().item()
        
        # 半径误差
        radius_error = (preds[:, 2] - targets[:, 2]).abs().mean().item()
        
        # 各维度误差
        x_error = (preds[:, 0] - targets[:, 0]).abs().mean().item()
        y_error = (preds[:, 1] - targets[:, 1]).abs().mean().item()
        r_error = radius_error
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "center_distance": center_distance,
            "radius_error": radius_error,
            "x_error": x_error,
            "y_error": y_error,
            "r_error": r_error,
        }
        
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
    
    def visualize_predictions(self, output_dir: str = "results/visualizations") -> None:
        """
        可视化预测结果（使用 test_rings/）
        
        分两种场景输出：
        - ring2_3/: 场景1 - 只提供Ring1，预测Ring2和Ring3
        - ring3/: 场景2 - 提供Ring1+Ring2，预测Ring3
        
        在地图上绘制：
        - 真实圈（白色）
        - 预测圈（黄色）
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        
        # 创建两个子目录
        ring2_3_dir = output_path / "ring2_3"
        ring3_dir = output_path / "ring3"
        
        # 清空旧文件
        if ring2_3_dir.exists():
            import shutil
            shutil.rmtree(ring2_3_dir)
        if ring3_dir.exists():
            import shutil
            shutil.rmtree(ring3_dir)
        
        ring2_3_dir.mkdir(parents=True, exist_ok=True)
        ring3_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有测试样本，排除指定地图
        json_files = sorted(self.test_rings_dir.glob("*.json"))
        json_files = [f for f in json_files if not any(excluded in f.stem for excluded in EXCLUDED_MAPS)]
        
        for json_file in json_files:
            # 加载数据
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            map_name = data.get("map")
            rings = data.get("rings", [])
            
            if len(rings) < 3:
                continue
            
            # 场景1：只提供Ring1，预测Ring2和Ring3
            # 先预测Ring2
            pred_ring2 = self._predict_ring2(rings[0])
            # 再用预测的Ring2预测Ring3
            pred_ring3_from_pred_ring2 = self._predict_ring3(rings[0], pred_ring2)
            
            img_ring2_3 = self._draw_predictions(
                map_name=map_name,
                true_rings=rings,
                pred_rings=[rings[0], pred_ring2, pred_ring3_from_pred_ring2],
                pred_indices=[1, 2]  # Ring2和Ring3都是预测的
            )
            
            if img_ring2_3 is not None:
                output_file = ring2_3_dir / f"{json_file.stem}_pred.png"
                cv2.imwrite(str(output_file), img_ring2_3)
            
            # 场景2：提供Ring1+真实Ring2，预测Ring3
            pred_ring3_from_true_ring2 = self._predict_ring3(rings[0], rings[1])
            
            img_ring3 = self._draw_predictions(
                map_name=map_name,
                true_rings=rings,
                pred_rings=[rings[0], rings[1], pred_ring3_from_true_ring2],
                pred_indices=[2]  # 只有Ring3是预测的
            )
            
            if img_ring3 is not None:
                output_file = ring3_dir / f"{json_file.stem}_pred.png"
                cv2.imwrite(str(output_file), img_ring3)
        
        print(f"可视化完成！保存到: {output_path}")
        print(f"  - ring2_3/: 场景1（预测Ring2+Ring3）")
        print(f"  - ring3/: 场景2（预测Ring3）")
    
    def _predict_ring2(self, ring1: Dict[str, float]) -> Dict[str, float]:
        """
        只预测 Ring2
        
        Args:
            ring1: Ring1 数据
            
        Returns:
            pred_ring2 (绝对坐标)
        """
        self.model.eval()
        
        # 归一化
        x1, y1, r1 = ring1["x"] / self.grid_size, ring1["y"] / self.grid_size, ring1["r"] / self.grid_size
        
        with torch.no_grad():
            # 预测 Ring2（输入padding到6维）
            input1 = torch.tensor([x1, y1, r1, 0, 0, 0], dtype=torch.float32).unsqueeze(0).to(self.device)
            output1 = self.model(input1).cpu().numpy()[0]
        
        if self.coordinate_mode == "absolute":
            # 绝对坐标模式
            pred_ring2 = {
                "x": int(output1[0] * self.grid_size),
                "y": int(output1[1] * self.grid_size),
                "r": int(output1[2] * self.grid_size)
            }
        elif self.coordinate_mode == "relative":
            # 相对坐标模式，转换为绝对坐标
            pred_ring2 = {
                "x": int((x1 + output1[0]) * self.grid_size),
                "y": int((y1 + output1[1]) * self.grid_size),
                "r": int(output1[2] * self.grid_size)
            }
        
        return pred_ring2
    
    def _predict_ring3(
        self, 
        ring1: Dict[str, float], 
        ring2: Dict[str, float]
    ) -> Dict[str, float]:
        """
        预测 Ring3
        
        Args:
            ring1: Ring1 数据（绝对坐标）
            ring2: Ring2 数据（绝对坐标，可以是真实值或预测值）
            
        Returns:
            pred_ring3 (绝对坐标)
        """
        self.model.eval()
        
        # 归一化
        x1, y1, r1 = ring1["x"] / self.grid_size, ring1["y"] / self.grid_size, ring1["r"] / self.grid_size
        x2, y2, r2 = ring2["x"] / self.grid_size, ring2["y"] / self.grid_size, ring2["r"] / self.grid_size
        
        with torch.no_grad():
            if self.coordinate_mode == "absolute":
                # 绝对坐标模式
                input2 = torch.tensor([x1, y1, r1, x2, y2, r2], dtype=torch.float32).unsqueeze(0).to(self.device)
                output2 = self.model(input2).cpu().numpy()[0]
                
                pred_ring3 = {
                    "x": int(output2[0] * self.grid_size),
                    "y": int(output2[1] * self.grid_size),
                    "r": int(output2[2] * self.grid_size)
                }
            
            elif self.coordinate_mode == "relative":
                # 相对坐标模式
                # Ring2相对Ring1
                dx2, dy2 = x2 - x1, y2 - y1
                input2 = torch.tensor([x1, y1, r1, dx2, dy2, r2], dtype=torch.float32).unsqueeze(0).to(self.device)
                output2 = self.model(input2).cpu().numpy()[0]  # [dx3, dy3, r3] 相对Ring2
                
                # 转换为绝对坐标
                pred_ring3 = {
                    "x": int((x2 + output2[0]) * self.grid_size),
                    "y": int((y2 + output2[1]) * self.grid_size),
                    "r": int(output2[2] * self.grid_size)
                }
        
        return pred_ring3
    
    def _predict_rings(
        self, 
        ring1: Dict[str, float], 
        ring2: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        预测 Ring2 和 Ring3（保留用于兼容性）
        
        Args:
            ring1: Ring1 数据
            ring2: Ring2 数据（用于预测 Ring3）
            
        Returns:
            (pred_ring2, pred_ring3)
        """
        pred_ring2 = self._predict_ring2(ring1)
        pred_ring3 = self._predict_ring3(ring1, ring2)
        return pred_ring2, pred_ring3
    
    def _draw_predictions(
        self,
        map_name: str,
        true_rings: List[Dict[str, float]],
        pred_rings: List[Dict[str, float]],
        pred_indices: List[int] = None
    ) -> Optional[np.ndarray]:
        """
        在地图上绘制真实圈和预测圈
        
        Args:
            map_name: 地图名称
            true_rings: 真实圈列表
            pred_rings: 预测圈列表（包含真实和预测的混合）
            pred_indices: 哪些索引是预测的（默认除了Ring1都是预测）
            
        Returns:
            绘制后的图像
        """
        # 加载地图
        map_file = MAP_FILES.get(map_name)
        if not map_file:
            print(f"  ✗ 未找到地图: {map_name}")
            return None
        
        map_path = self.map_dir / map_file
        if not map_path.exists():
            print(f"  ✗ 地图文件不存在: {map_path}")
            return None
        
        img = cv2.imread(str(map_path))
        if img is None:
            return None
        
        # 计算缩放比例
        scale = img.shape[0] / self.grid_size
        
        # 绘制真实圈（白色）
        for ring in true_rings:
            x = int(ring["x"] * scale)
            y = int(ring["y"] * scale)
            r = int(ring["r"] * scale)
            if r > 0:  # 确保半径为正
                cv2.circle(img, (x, y), r, (255, 255, 255), 2)
        
        # 如果没有指定pred_indices，默认除了Ring1都是预测
        if pred_indices is None:
            pred_indices = list(range(1, len(pred_rings)))
        
        # 绘制预测圈（黄色）
        for i in pred_indices:
            if i >= len(pred_rings):
                continue
            ring = pred_rings[i]
            x = int(ring["x"] * scale)
            y = int(ring["y"] * scale)
            r = int(ring["r"] * scale)
            if r > 0:  # 确保半径为正
                cv2.circle(img, (x, y), r, (0, 255, 255), 2)
        
        return img
    
    def visualize_custom(self, *args, **kwargs):
        """
        自定义可视化 - 可由子类重写
        
        例如：热力图、Top-K 预测等
        """
        raise NotImplementedError("子类需要实现自定义可视化方法")
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        打印评估指标
        
        Args:
            metrics: 指标字典，包含两个场景的评估结果
        """
        scenario1 = metrics.get("scenario_1_only_ring1", {})
        scenario2 = metrics.get("scenario_2_ring1_and_ring2", {})
        
        # 收集所有半径误差
        ring2_metrics = scenario1.get("ring2_error", {})
        scenario1_ring3_metrics = scenario1.get("ring3_error", {})
        scenario2_ring3_metrics = scenario2.get("ring3_error", {})
        
        # 获取baseline指标（如果有）
        baseline_scenario1 = None
        baseline_scenario2 = None
        if self.baseline_metrics:
            baseline_scenario1 = self.baseline_metrics.get("scenario_1_only_ring1", {})
            baseline_scenario2 = self.baseline_metrics.get("scenario_2_ring1_and_ring2", {})
        
        # 打印半径误差（独立展示）
        print(f"\n{'='*70}")
        print("半径误差（理论上应为0）")
        print(f"{'='*70}")
        
        if ring2_metrics:
            radius_err_px = ring2_metrics['radius_error'] * self.grid_size
            if baseline_scenario1:
                baseline_r2 = baseline_scenario1.get("ring2_error", {})
                baseline_radius_px = baseline_r2.get('radius_error', 0) * self.grid_size
                print(f"  场景1 - Ring2: {radius_err_px:.1f} px ({baseline_radius_px:.1f} px)")
            else:
                print(f"  场景1 - Ring2: {radius_err_px:.1f} px")
        
        if scenario1_ring3_metrics:
            radius_err_px = scenario1_ring3_metrics['radius_error'] * self.grid_size
            if baseline_scenario1:
                baseline_r3 = baseline_scenario1.get("ring3_error", {})
                baseline_radius_px = baseline_r3.get('radius_error', 0) * self.grid_size
                print(f"  场景1 - Ring3: {radius_err_px:.1f} px ({baseline_radius_px:.1f} px)")
            else:
                print(f"  场景1 - Ring3: {radius_err_px:.1f} px")
        
        if scenario2_ring3_metrics:
            radius_err_px = scenario2_ring3_metrics['radius_error'] * self.grid_size
            if baseline_scenario2:
                baseline_r3 = baseline_scenario2.get("ring3_error", {})
                baseline_radius_px = baseline_r3.get('radius_error', 0) * self.grid_size
                print(f"  场景2 - Ring3: {radius_err_px:.1f} px ({baseline_radius_px:.1f} px)")
            else:
                print(f"  场景2 - Ring3: {radius_err_px:.1f} px")
        
        # 场景1
        print(f"\n{'='*70}")
        print("场景1：只提供 Ring1")
        print(f"{'='*70}")
        
        # Ring2 误差
        if ring2_metrics:
            center_dist_px = ring2_metrics['center_distance'] * self.grid_size
            
            print(f"\nRing2 预测误差:")
            if baseline_scenario1:
                baseline_r2 = baseline_scenario1.get("ring2_error", {})
                baseline_center_px = baseline_r2.get('center_distance', 0) * self.grid_size
                print(f"  圆心距离误差: {center_dist_px:.1f} px ({baseline_center_px:.1f} px)")
            else:
                print(f"  圆心距离误差: {center_dist_px:.1f} px")
            print(f"  MSE:          {ring2_metrics['mse']:.6f}")
            print(f"  MAE:          {ring2_metrics['mae']:.6f}")
        
        # Ring3 误差（基于预测的Ring2）
        if scenario1_ring3_metrics:
            center_dist_px = scenario1_ring3_metrics['center_distance'] * self.grid_size
            
            print(f"\nRing3 预测误差（基于预测的Ring2）:")
            if baseline_scenario1:
                baseline_r3 = baseline_scenario1.get("ring3_error", {})
                baseline_center_px = baseline_r3.get('center_distance', 0) * self.grid_size
                print(f"  圆心距离误差: {center_dist_px:.1f} px ({baseline_center_px:.1f} px)")
            else:
                print(f"  圆心距离误差: {center_dist_px:.1f} px")
            print(f"  MSE:          {scenario1_ring3_metrics['mse']:.6f}")
            print(f"  MAE:          {scenario1_ring3_metrics['mae']:.6f}")
        
        # 按地图展示
        by_map = scenario1.get("by_map", {})
        if by_map:
            ring2_by_map = by_map.get("ring2_error", {})
            ring3_by_map = by_map.get("ring3_error", {})
            
            # 获取baseline按地图的指标
            baseline_r2_by_map = {}
            baseline_r3_by_map = {}
            if baseline_scenario1:
                baseline_by_map = baseline_scenario1.get("by_map", {})
                baseline_r2_by_map = baseline_by_map.get("ring2_error", {})
                baseline_r3_by_map = baseline_by_map.get("ring3_error", {})
            
            if ring2_by_map or ring3_by_map:
                print(f"\n各地图详细结果:")
                for map_name in sorted(set(list(ring2_by_map.keys()) + list(ring3_by_map.keys()))):
                    print(f"\n  {map_name}:")
                    
                    if map_name in ring2_by_map:
                        r2_metrics = ring2_by_map[map_name]
                        center_px = r2_metrics['center_distance'] * self.grid_size
                        if map_name in baseline_r2_by_map:
                            baseline_center_px = baseline_r2_by_map[map_name]['center_distance'] * self.grid_size
                            print(f"    Ring2 圆心误差: {center_px:.1f} px ({baseline_center_px:.1f} px)")
                        else:
                            print(f"    Ring2 圆心误差: {center_px:.1f} px")
                    
                    if map_name in ring3_by_map:
                        r3_metrics = ring3_by_map[map_name]
                        center_px = r3_metrics['center_distance'] * self.grid_size
                        if map_name in baseline_r3_by_map:
                            baseline_center_px = baseline_r3_by_map[map_name]['center_distance'] * self.grid_size
                            print(f"    Ring3 圆心误差: {center_px:.1f} px ({baseline_center_px:.1f} px)")
                        else:
                            print(f"    Ring3 圆心误差: {center_px:.1f} px")
        
        # 场景2
        print(f"\n{'='*70}")
        print("场景2：提供 Ring1 + Ring2")
        print(f"{'='*70}")
        
        # Ring3 误差（基于真实Ring2）
        if scenario2_ring3_metrics:
            center_dist_px = scenario2_ring3_metrics['center_distance'] * self.grid_size
            
            print(f"\nRing3 预测误差（基于真实Ring2）:")
            if baseline_scenario2:
                baseline_r3 = baseline_scenario2.get("ring3_error", {})
                baseline_center_px = baseline_r3.get('center_distance', 0) * self.grid_size
                print(f"  圆心距离误差: {center_dist_px:.1f} px ({baseline_center_px:.1f} px)")
            else:
                print(f"  圆心距离误差: {center_dist_px:.1f} px")
            print(f"  MSE:          {scenario2_ring3_metrics['mse']:.6f}")
            print(f"  MAE:          {scenario2_ring3_metrics['mae']:.6f}")
        
        # 按地图展示
        by_map = scenario2.get("by_map", {})
        if by_map:
            ring3_by_map = by_map.get("ring3_error", {})
            
            # 获取baseline按地图的指标
            baseline_r3_by_map = {}
            if baseline_scenario2:
                baseline_by_map = baseline_scenario2.get("by_map", {})
                baseline_r3_by_map = baseline_by_map.get("ring3_error", {})
            
            if ring3_by_map:
                print(f"\n各地图详细结果:")
                for map_name in sorted(ring3_by_map.keys()):
                    r3_metrics = ring3_by_map[map_name]
                    center_px = r3_metrics['center_distance'] * self.grid_size
                    if map_name in baseline_r3_by_map:
                        baseline_center_px = baseline_r3_by_map[map_name]['center_distance'] * self.grid_size
                        print(f"  {map_name} 圆心误差: {center_px:.1f} px ({baseline_center_px:.1f} px)")
                    else:
                        print(f"  {map_name} 圆心误差: {center_px:.1f} px")
        
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
    
    # 可视化
    evaluator.visualize_predictions(output_dir="tests/temp/visualizations")
