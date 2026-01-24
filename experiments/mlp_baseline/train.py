"""MLP Baseline训练脚本（One-Hot地图编码）"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.optim as optim
from mlp_baseline import MLPBaseline
from predictor import BaselinePredictor
from model.dataset import get_dataloader
from model.loss import get_loss_fn
from model.trainer import Trainer


def train_baseline(show_scenario_errors: bool = False):
    """
    训练MLP Baseline（One-Hot地图编码）
    
    Args:
        show_scenario_errors: 是否在训练时显示场景误差（会降低训练速度）
    """
    print(f"\n{'='*70}")
    print(f"MLP Baseline实验（One-Hot地图编码）")
    print(f"输入: 6维坐标 + 2维One-Hot地图编码")
    if show_scenario_errors:
        print(f"显示场景误差: 开启（训练会较慢）")
    print(f"{'='*70}\n")
    
    # 数据（使用框架的dataset，默认use_map_encoding=True）
    train_loader = get_dataloader("train", batch_size=32, shuffle=True)
    val_loader = get_dataloader("val", batch_size=32, shuffle=False)
    test_loader = get_dataloader("test", batch_size=32, shuffle=False)
    
    # 模型
    model = MLPBaseline()
    model.summary()
    
    # 训练
    loss_fn = get_loss_fn("mse")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    save_dir = Path(__file__).parent / "checkpoints"
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=str(save_dir),
        early_stopping_patience=20,
        verbose=True,
        compute_scenario_errors=show_scenario_errors,
        predictor_class=BaselinePredictor,  # 自动评估
        test_loader=test_loader,             # 自动评估
        auto_evaluate=True                   # 自动评估
    )
    
    trainer.train(num_epochs=100)
    
    print("\nBaseline实验完成！")
    print(f"  模型: {save_dir / model.model_name / 'best_model.pth'}")
    print(f"  可视化: 启动 Web 服务器查看 (python utils/ring_viewer_server.py)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-errors", action="store_true", 
                        help="在训练时显示场景误差（会降低训练速度）")
    args = parser.parse_args()
    
    train_baseline(show_scenario_errors=args.show_errors)
