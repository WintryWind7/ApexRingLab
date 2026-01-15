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
from model.evaluator import Evaluator


def train_baseline():
    """训练MLP Baseline（One-Hot地图编码）"""
    print(f"\n{'='*70}")
    print(f"MLP Baseline实验（One-Hot地图编码）")
    print(f"输入: 6维坐标 + 2维One-Hot地图编码")
    print(f"{'='*70}\n")
    
    # 数据（使用框架的dataset，默认use_map_encoding=True）
    train_loader = get_dataloader("train", batch_size=32, shuffle=True)
    val_loader = get_dataloader("val", batch_size=32, shuffle=False)
    
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
        coordinate_mode="relative",
        use_onehot=True
    )
    
    trainer.train(num_epochs=100)
    
    # 评估
    print("\n开始评估...")
    
    # 加载最佳模型
    best_model_path = save_dir / "best_model.pth"
    model.load_checkpoint(str(best_model_path))
    print(f"已加载最佳模型: {best_model_path}")
    
    # 创建Predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = BaselinePredictor(model, device)
    
    # 创建Evaluator并评估
    test_loader = get_dataloader("test", batch_size=32, shuffle=False)
    evaluator = Evaluator(predictor=predictor, device=device)
    metrics = evaluator.evaluate(test_loader)
    evaluator.print_metrics(metrics)
    
    # 可视化
    vis_dir = Path(__file__).parent / "visualizations"
    print(f"\n生成可视化...")
    evaluator.visualize_predictions(output_dir=str(vis_dir))
    
    print("\nBaseline实验完成！")
    print(f"  模型: {best_model_path}")
    print(f"  可视化: {vis_dir}")


if __name__ == "__main__":
    train_baseline()
