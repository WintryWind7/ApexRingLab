"""模型检查工具 - 查看评估结果和对比模型"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import argparse
from typing import List, Optional
from model.comparator import compare_models


def view_model_metrics(model_path: str, grid_size: int = 16384) -> None:
    """
    查看单个模型的评估结果
    
    Args:
        model_path: 模型pth文件路径
        grid_size: 地图网格大小
    """
    path = Path(model_path)
    if not path.exists():
        print(f"⚠ 模型文件不存在: {model_path}")
        return
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查是否有test_metrics
    if "test_metrics" not in checkpoint:
        print(f"⚠ {model_path} 中没有test_metrics")
        print(f"   该模型可能未完成训练或使用旧版框架训练")
        return
    
    test_metrics_with_meta = checkpoint["test_metrics"]
    metrics = test_metrics_with_meta.get("metrics")
    
    if not metrics:
        print(f"⚠ {model_path} 中没有metrics数据")
        return
    
    # 打印模型信息
    print(f"\n{'='*80}")
    print(f"Model: {path.parent.parent.name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}\n")
    
    # 打印元信息
    print("Evaluation Info:")
    print(f"  Time: {test_metrics_with_meta.get('evaluated_at', 'unknown')}")
    print(f"  Test Dataset: {test_metrics_with_meta.get('test_dataset_version', 'unknown')}")
    print(f"  Duration: {test_metrics_with_meta.get('evaluation_duration_seconds', 'unknown')} s")
    
    # 打印训练信息
    if "best_epoch" in checkpoint:
        print(f"\nTraining Info:")
        print(f"  Best Epoch: {checkpoint['best_epoch']}")
        print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'unknown'):.6f}")
    
    # 提取场景数据
    scenario1 = metrics.get("scenario_1_only_ring1", {})
    scenario2 = metrics.get("scenario_2_ring1_and_ring2", {})
    
    ring2_metrics = scenario1.get("ring2_error", {})
    scenario1_ring3_metrics = scenario1.get("ring3_error", {})
    scenario2_ring3_metrics = scenario2.get("ring3_error", {})
    
    # 直接打印所有指标
    import json
    print(f"\n{json.dumps(metrics, indent=2)}\n")


def list_available_models() -> List[str]:
    """
    列出所有可用的模型
    
    Returns:
        模型路径列表
    """
    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / "experiments"
    
    models = []
    
    if not experiments_dir.exists():
        return models
    
    # 遍历所有实验目录
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # 查找 checkpoints 目录
        checkpoints_dirs = [
            exp_dir / "checkpoints",
            exp_dir / "checkpoints_stage1",
            exp_dir / "checkpoints_stage2",
        ]
        
        for ckpt_dir in checkpoints_dirs:
            if ckpt_dir.exists():
                # 查找 best_model.pth 或 best_generator.pth
                for model_file in ["best_model.pth", "best_generator.pth"]:
                    model_path = ckpt_dir / model_file
                    if model_path.exists():
                        models.append(str(model_path))
    
    return sorted(models)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="模型检查工具 - 查看评估结果和对比模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看单个模型
  python utils/model_inspector.py view experiments/mlp_baseline/checkpoints/best_model.pth
  
  # 对比两个模型
  python utils/model_inspector.py compare experiments/mlp_baseline/checkpoints/best_model.pth experiments/mlp_deep/checkpoints/best_model.pth
  
  # 对比多个模型并指定名称
  python utils/model_inspector.py compare experiments/mlp_baseline/checkpoints/best_model.pth experiments/mlp_deep/checkpoints/best_model.pth -n Baseline Deep
  
  # 列出所有可用模型
  python utils/model_inspector.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # view 命令
    view_parser = subparsers.add_parser("view", help="查看单个模型的评估结果")
    view_parser.add_argument("model_path", help="模型pth文件路径")
    
    # compare 命令
    compare_parser = subparsers.add_parser("compare", help="对比多个模型")
    compare_parser.add_argument("model_paths", nargs="+", help="模型pth文件路径列表")
    compare_parser.add_argument("-n", "--names", nargs="+", help="模型名称列表（可选）")
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出所有可用模型")
    
    args = parser.parse_args()
    
    if args.command == "view":
        view_model_metrics(args.model_path)
    
    elif args.command == "compare":
        compare_models(args.model_paths, model_names=args.names)
    
    elif args.command == "list":
        models = list_available_models()
        if not models:
            print("未找到任何模型")
        else:
            print(f"\n找到 {len(models)} 个模型:\n")
            for i, model_path in enumerate(models, 1):
                path = Path(model_path)
                exp_name = path.parent.parent.name
                ckpt_name = path.parent.name
                print(f"{i}. {exp_name}/{ckpt_name}/{path.name}")
                print(f"   {model_path}\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
