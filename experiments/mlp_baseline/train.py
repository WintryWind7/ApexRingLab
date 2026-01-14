"""MLP Baseline 实验

简单的多层感知机作为基准模型
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch.optim as optim

from model.dataset import get_dataloader
from model.loss import get_loss_fn
from model.trainer import Trainer
from model.evaluator import Evaluator

# 导入模型
from mlp_baseline import MLPBaseline


def train_baseline():
    """训练 baseline 模型"""
    
    print("="*70)
    print("MLP Baseline 实验")
    print("="*70)
    
    # 超参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    
    # 数据加载
    print("\n加载数据...")
    train_loader = get_dataloader("train", batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader("val", batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader("test", batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print("创建模型...")
    model = MLPBaseline()
    model.summary()
    
    # 损失函数和优化器
    loss_fn = get_loss_fn("mse")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 训练器
    exp_dir = Path(__file__).parent
    save_dir = exp_dir / "checkpoints"
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
        auto_evaluate=True,
        test_loader=test_loader
    )
    
    # 训练
    print("\n开始训练...")
    history = trainer.train(num_epochs=num_epochs)
    
    print("\n实验完成！")
    print(f"模型保存在: {save_dir}")
    print(f"可视化保存在: {save_dir.parent / 'visualizations'}")


if __name__ == "__main__":
    train_baseline()
