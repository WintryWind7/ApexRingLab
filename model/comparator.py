"""模型对比工具"""

import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


def _camel_to_spaced(name: str) -> str:
    """
    将驼峰命名转换为空格分隔
    
    处理规则：
    - MLPBaseline -> MLP Baseline
    - GANGenerator -> GAN Generator
    - MLPVeryDeep -> MLP Very Deep
    
    Args:
        name: 驼峰命名字符串
        
    Returns:
        空格分隔的字符串
    """
    # 在大写字母前插入空格（但不在开头）
    # 处理连续大写字母（如 MLP, GAN）
    result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)  # aA -> a A
    result = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', result)  # ABc -> A Bc
    return result


def _format_timestamp(evaluated_at: str, reference_year: Optional[int] = None) -> str:
    """
    格式化时间戳用于重名区分
    
    规则：
    - 同年：MMDDhh（如 012019）
    - 跨年：YYYYMMDDhh（如 2025010110）
    
    Args:
        evaluated_at: 评估时间字符串（如 "2026-01-20 19:33:41"）
        reference_year: 参考年份（用于判断是否跨年，None 表示跨年）
        
    Returns:
        格式化的时间戳
    """
    try:
        dt = datetime.strptime(evaluated_at, "%Y-%m-%d %H:%M:%S")
        
        if reference_year is not None and dt.year == reference_year:
            # 同年：MMDDhh
            return dt.strftime("%m%d%H")
        else:
            # 跨年或无参考年份：YYYYMMDDhh
            return dt.strftime("%Y%m%d%H")
    except:
        return "unknown"


def _generate_display_names(models_data: List[Dict[str, Any]]) -> List[str]:
    """
    自动生成模型展示名称
    
    规则：
    1. 从 model_name 提取类名，转换为空格分隔
    2. 如果不重名，直接使用
    3. 如果重名，添加时间后缀（同年用 MMDDhh，跨年用 YYYYMMDDhh）
    
    Args:
        models_data: 模型数据列表
        
    Returns:
        展示名称列表
    """
    # 提取所有模型的基础名称
    base_names = []
    for data in models_data:
        model_name = data.get("model_name", "Unknown")
        base_name = _camel_to_spaced(model_name)
        base_names.append(base_name)
    
    # 按基础名称分组
    name_groups = {}
    for i, name in enumerate(base_names):
        if name not in name_groups:
            name_groups[name] = []
        name_groups[name].append(i)
    
    # 生成最终名称
    display_names = [""] * len(models_data)
    
    for base_name, indices in name_groups.items():
        if len(indices) == 1:
            # 不重名，直接使用
            display_names[indices[0]] = base_name
        else:
            # 重名，需要添加时间后缀
            # 提取该组的所有年份
            group_years = []
            for idx in indices:
                evaluated_at = models_data[idx].get("evaluated_at", "")
                try:
                    dt = datetime.strptime(evaluated_at, "%Y-%m-%d %H:%M:%S")
                    group_years.append(dt.year)
                except:
                    group_years.append(None)
            
            # 判断该组是否跨年
            unique_years = set(y for y in group_years if y is not None)
            reference_year = list(unique_years)[0] if len(unique_years) == 1 else None
            
            # 为该组的每个模型生成带时间后缀的名称
            for idx in indices:
                evaluated_at = models_data[idx].get("evaluated_at", "")
                timestamp = _format_timestamp(evaluated_at, reference_year)
                display_names[idx] = f"{base_name}-{timestamp}"
    
    return display_names


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
        model_names: 模型名称列表（可选，默认自动生成）
        grid_size: 地图网格大小（用于像素转换）
    
    Example:
        # 自动命名
        compare_models([
            "experiments/mlp_baseline/checkpoints/best_model.pth",
            "experiments/mlp_deep/checkpoints/best_model.pth"
        ])
        
        # 手动命名
        compare_models([
            "experiments/mlp_baseline/checkpoints/best_model.pth",
            "experiments/mlp_deep/checkpoints/best_model.pth"
        ], model_names=["Baseline", "Deep"])
    """
    # 加载所有模型的测试结果
    models_data = []
    for i, model_path in enumerate(model_paths):
        path = Path(model_path)
        if not path.exists():
            print(f"⚠ 模型文件不存在: {model_path}")
            continue
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # 提取test_metrics
        if "test_metrics" not in checkpoint:
            print(f"⚠ {model_path} 中没有test_metrics，跳过")
            continue
        
        test_metrics_with_meta = checkpoint["test_metrics"]
        metrics = test_metrics_with_meta.get("metrics")
        
        if not metrics:
            print(f"⚠ {model_path} 中没有metrics数据，跳过")
            continue
        
        models_data.append({
            "path": model_path,
            "model_name": checkpoint.get("model_name", "Unknown"),
            "metrics": metrics,
            "evaluated_at": test_metrics_with_meta.get("evaluated_at", "unknown"),
            "test_dataset_version": test_metrics_with_meta.get("test_dataset_version", "unknown")
        })
    
    if len(models_data) < 2:
        print("⚠ 至少需要2个有效模型才能对比")
        return
    
    # 生成展示名称
    if model_names:
        # 使用用户指定的名称
        display_names = model_names[:len(models_data)]
    else:
        # 自动生成名称
        display_names = _generate_display_names(models_data)
    
    # 添加展示名称到数据中
    for data, name in zip(models_data, display_names):
        data["display_name"] = name
    
    # 打印对比结果
    _print_comparison(models_data, grid_size)


def _print_comparison(models_data: List[Dict[str, Any]], grid_size: int) -> None:
    """
    打印对比结果（完全自适应，自动遍历所有指标）
    
    Args:
        models_data: 模型数据列表
        grid_size: 地图网格大小（未使用，保留兼容性）
    """
    print(f"\n{'='*80}")
    print(f"模型对比 ({len(models_data)} 个模型)")
    print(f"{'='*80}\n")
    
    # 打印模型信息
    for i, data in enumerate(models_data, 1):
        print(f"{i}. {data['display_name']}")
        print(f"   类名: {data['model_name']}")
        print(f"   路径: {data['path']}")
        print(f"   评估时间: {data['evaluated_at']}")
        print(f"   测试集版本: {data['test_dataset_version']}\n")
    
    # 获取第一个模型的指标结构（作为模板）
    first_metrics = models_data[0]["metrics"]
    
    # 遍历所有场景
    for scenario_key, scenario_data in first_metrics.items():
        # 打印场景标题
        if scenario_key == "scenario_1_only_ring1":
            print(f"{'='*80}")
            print("场景1：只提供 Ring1")
            print(f"{'='*80}\n")
        elif scenario_key == "scenario_2_ring1_and_ring2":
            print(f"\n{'='*80}")
            print("场景2：提供 Ring1 + Ring2")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"{scenario_key}")
            print(f"{'='*80}\n")
        
        # 遍历该场景下的所有误差类型（如 ring2_error, ring3_error）
        for error_key, error_data in scenario_data.items():
            # 跳过 by_map（按地图的详细结果）
            if error_key == "by_map":
                continue
            
            # 打印误差类型标题
            if error_key == "ring2_error":
                print("Ring2 预测误差:\n")
            elif error_key == "ring3_error":
                if scenario_key == "scenario_1_only_ring1":
                    print("\nRing3 预测误差（基于预测的Ring2）:\n")
                else:
                    print("Ring3 预测误差（基于真实Ring2）:\n")
            else:
                print(f"\n{error_key}:\n")
            
            # 遍历该误差类型下的所有指标
            if isinstance(error_data, dict):
                for metric_key in error_data.keys():
                    # 判断单位
                    if metric_key in ["center_distance", "radius_error"]:
                        unit = "px"
                    else:
                        unit = ""
                    
                    # 打印该指标的对比
                    _print_metric_comparison(
                        models_data,
                        metric_path=[scenario_key, error_key, metric_key],
                        label=metric_key,
                        unit=unit
                    )
    
    print(f"{'='*80}\n")


def _print_metric_comparison(
    models_data: List[Dict[str, Any]],
    metric_path: List[str],
    label: str,
    unit: str = ""
) -> None:
    """
    打印单个指标的对比
    
    Args:
        models_data: 模型数据列表
        metric_path: 指标路径（如 ["scenario_1_only_ring1", "ring2_error", "center_distance"]）
        label: 指标标签
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
            print(f"  {data['display_name']:20s}: N/A")
        else:
            # 标记最佳值
            mark = " *" if value == best_value else ""
            if unit:
                print(f"  {data['display_name']:20s}: {value:8.1f} {unit}{mark}")
            else:
                print(f"  {data['display_name']:20s}: {value:.6f}{mark}")
    
    print()


if __name__ == "__main__":
    # 测试对比工具
    compare_models([
        "experiments/mlp_baseline/checkpoints/best_model.pth",
        "experiments/mlp_deep/checkpoints/best_model.pth"
    ], model_names=["Baseline", "Deep"])
