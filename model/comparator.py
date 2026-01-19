"""模型对比工具"""

import torch
from pathlib import Path
from typing import List, Optional, Dict, Any


def compare_models(
    model_paths: List[str],
    model_names: Optional[List[str]] = None,
    grid_size: int = 16384
) -> None:
    """
    对比多个模型的测试结果
    
    从pth文件中读取test_metrics并进行对比展示
    
    Args:
        model_paths: 模型pth文件路径列表
        model_names: 模型名称列表（可选，默认使用文件名）
        grid_size: 地图网格大小（用于像素转换）
    
    Example:
        compare_models([
            "experiments/mlp_baseline/checkpoints/best_model.pth",
            "experiments/mlp_deep/checkpoints/best_model.pth",
            "experiments/gan/checkpoints/best_generator.pth"
        ], names=["Baseline", "Deep", "GAN"])
    """
    # 加载所有模型的测试结果
    models_data = []
    for i, model_path in enumerate(model_paths):
        path = Path(model_path)
        if not path.exists():
            print(f"⚠ 模型文件不存在: {model_path}")
            continue
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # 提取test_metrics
        if "test_metrics" not in checkpoint:
            print(f"⚠ {model_path} 中没有test_metrics，跳过")
            continue
        
        test_metrics_with_meta = checkpoint["test_metrics"]
        metrics = test_metrics_with_meta.get("metrics")
        
        if not metrics:
            print(f"⚠ {model_path} 中没有metrics数据，跳过")
            continue
        
        # 确定模型名称
        if model_names and i < len(model_names):
            name = model_names[i]
        else:
            name = path.parent.parent.name  # 使用实验目录名
        
        models_data.append({
            "name": name,
            "path": model_path,
            "metrics": metrics,
            "evaluated_at": test_metrics_with_meta.get("evaluated_at", "unknown"),
            "test_dataset_version": test_metrics_with_meta.get("test_dataset_version", "unknown")
        })
    
    if len(models_data) < 2:
        print("⚠ 至少需要2个有效模型才能对比")
        return
    
    # 打印对比结果
    _print_comparison(models_data, grid_size)


def _print_comparison(models_data: List[Dict[str, Any]], grid_size: int) -> None:
    """
    打印对比结果
    
    Args:
        models_data: 模型数据列表
        grid_size: 地图网格大小
    """
    print(f"\n{'='*80}")
    print(f"模型对比 ({len(models_data)} 个模型)")
    print(f"{'='*80}\n")
    
    # 打印模型信息
    for i, data in enumerate(models_data, 1):
        print(f"{i}. {data['name']}")
        print(f"   路径: {data['path']}")
        print(f"   评估时间: {data['evaluated_at']}")
        print(f"   测试集版本: {data['test_dataset_version']}\n")
    
    # 对比半径误差
    print(f"{'='*80}")
    print("半径误差对比（理论上应为0）")
    print(f"{'='*80}\n")
    
    _print_metric_comparison(
        models_data,
        metric_path=["scenario_1_only_ring1", "ring2_error", "radius_error"],
        label="场景1 - Ring2",
        grid_size=grid_size,
        unit="px"
    )
    
    _print_metric_comparison(
        models_data,
        metric_path=["scenario_1_only_ring1", "ring3_error", "radius_error"],
        label="场景1 - Ring3",
        grid_size=grid_size,
        unit="px"
    )
    
    _print_metric_comparison(
        models_data,
        metric_path=["scenario_2_ring1_and_ring2", "ring3_error", "radius_error"],
        label="场景2 - Ring3",
        grid_size=grid_size,
        unit="px"
    )
    
    # 对比圆心距离误差
    print(f"\n{'='*80}")
    print("圆心距离误差对比")
    print(f"{'='*80}\n")
    
    print("场景1：只提供 Ring1\n")
    
    _print_metric_comparison(
        models_data,
        metric_path=["scenario_1_only_ring1", "ring2_error", "center_distance"],
        label="Ring2 圆心误差",
        grid_size=grid_size,
        unit="px"
    )
    
    _print_metric_comparison(
        models_data,
        metric_path=["scenario_1_only_ring1", "ring3_error", "center_distance"],
        label="Ring3 圆心误差",
        grid_size=grid_size,
        unit="px"
    )
    
    print("\n场景2：提供 Ring1 + Ring2\n")
    
    _print_metric_comparison(
        models_data,
        metric_path=["scenario_2_ring1_and_ring2", "ring3_error", "center_distance"],
        label="Ring3 圆心误差",
        grid_size=grid_size,
        unit="px"
    )
    
    # 对比MSE
    print(f"\n{'='*80}")
    print("MSE 对比")
    print(f"{'='*80}\n")
    
    print("场景1：只提供 Ring1\n")
    
    _print_metric_comparison(
        models_data,
        metric_path=["scenario_1_only_ring1", "ring2_error", "mse"],
        label="Ring2 MSE",
        grid_size=None,
        unit=""
    )
    
    _print_metric_comparison(
        models_data,
        metric_path=["scenario_1_only_ring1", "ring3_error", "mse"],
        label="Ring3 MSE",
        grid_size=None,
        unit=""
    )
    
    print("\n场景2：提供 Ring1 + Ring2\n")
    
    _print_metric_comparison(
        models_data,
        metric_path=["scenario_2_ring1_and_ring2", "ring3_error", "mse"],
        label="Ring3 MSE",
        grid_size=None,
        unit=""
    )
    
    print(f"{'='*80}\n")


def _print_metric_comparison(
    models_data: List[Dict[str, Any]],
    metric_path: List[str],
    label: str,
    grid_size: Optional[int] = None,
    unit: str = ""
) -> None:
    """
    打印单个指标的对比
    
    Args:
        models_data: 模型数据列表
        metric_path: 指标路径（如 ["scenario_1_only_ring1", "ring2_error", "center_distance"]）
        label: 指标标签
        grid_size: 网格大小（如果需要转换为像素）
        unit: 单位
    """
    print(f"{label}:")
    
    values = []
    for data in models_data:
        # 按路径提取指标值
        value = data["metrics"]
        for key in metric_path:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break
        
        if value is not None:
            # 如果需要转换为像素
            if grid_size is not None:
                value = value * grid_size
            values.append(value)
        else:
            values.append(None)
    
    # 找出最佳值（最小）
    valid_values = [v for v in values if v is not None]
    if valid_values:
        best_value = min(valid_values)
    else:
        best_value = None
    
    # 打印每个模型的值
    for i, (data, value) in enumerate(zip(models_data, values)):
        if value is None:
            print(f"  {data['name']:20s}: N/A")
        else:
            # 标记最佳值
            mark = " *" if value == best_value else ""
            if unit:
                print(f"  {data['name']:20s}: {value:8.1f} {unit}{mark}")
            else:
                print(f"  {data['name']:20s}: {value:.6f}{mark}")
    
    print()


if __name__ == "__main__":
    # 测试对比工具
    compare_models([
        "experiments/mlp_baseline/checkpoints/best_model.pth",
        "experiments/mlp_deep/checkpoints/best_model.pth"
    ], model_names=["Baseline", "Deep"])
