"""深层MLP训练脚本"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.optim as optim
from mlp_deep import MLPDeep, MLPVeryDeep
from predictor import DeepMLPPredictor
from model.dataset import get_dataloader
from model.loss import get_loss_fn
from model.trainer import Trainer
from model.evaluator import Evaluator


def train_deep_mlp(depth: str = "deep"):
    """
    训练深层MLP
    
    Args:
        depth: 'deep' 或 'very_deep'
    """
    if depth == "deep":
        model = MLPDeep()
        desc = "8 → 256 → 128 → 64 → 32 → 3"
        save_dir = Path(__file__).parent / "checkpoints"
    elif depth == "very_deep":
        model = MLPVeryDeep()
        desc = "8 → 512 → 256 → 256 → 128 → 128 → 64 → 32 → 3"
        save_dir = Path(__file__).parent / "checkpoints" / "very_deep"
    else:
        raise ValueError(f"Unknown depth: {depth}")
    
    print(f"\n{'='*70}")
    print(f"深层MLP实验 - {depth.upper()} - CircleLoss")
    print(f"结构: {desc}")
    print(f"损失: CircleLoss (alpha=2.0, beta=1.0)")
    print(f"{'='*70}\n")
    
    # 数据
    train_loader = get_dataloader("train", batch_size=32, shuffle=True)
    val_loader = get_dataloader("val", batch_size=32, shuffle=False)
    
    # 模型
    model.summary()
    
    # 训练（使用CircleLoss）
    loss_fn = get_loss_fn("circle", alpha=2.0, beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
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
    
    trainer.train(num_epochs=200)
    
    # 评估
    print("\n开始评估...")
    
    best_model_path = save_dir / "best_model.pth"
    model.load_checkpoint(str(best_model_path))
    print(f"已加载最佳模型: {best_model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DeepMLPPredictor(model, device)
    
    test_loader = get_dataloader("test", batch_size=32, shuffle=False)
    evaluator = Evaluator(predictor=predictor, device=device)  # 自动对比baseline
    metrics = evaluator.evaluate(test_loader)
    evaluator.print_metrics(metrics)
    
    # 可视化
    vis_dir = Path(__file__).parent / "visualizations" / depth
    print(f"\n生成可视化...")
    evaluator.visualize_predictions(output_dir=str(vis_dir))
    
    print(f"\n{depth.upper()}实验完成！")
    print(f"  模型: {best_model_path}")
    print(f"  可视化: {vis_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=str, default="very_deep", choices=["deep", "very_deep"],
                        help="模型深度")
    args = parser.parse_args()
    
    train_deep_mlp(args.depth)
