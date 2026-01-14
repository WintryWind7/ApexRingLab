"""统一训练器"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import time


class Trainer:
    """
    统一训练器 - 适用于所有模型
    
    功能：
    - 训练循环
    - 验证循环
    - 学习率调度
    - 早停
    - 检查点保存
    - 日志记录
    - 训练完成后自动评估（可选）
    """


class Trainer:
    """
    统一训练器 - 适用于所有模型
    
    功能：
    - 训练循环
    - 验证循环
    - 学习率调度
    - 早停
    - 检查点保存
    - 日志记录
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: Optional[Any] = None,
        save_dir: str = "checkpoints",
        early_stopping_patience: int = 10,
        verbose: bool = True,
        auto_evaluate: bool = True,
        test_loader: Optional[DataLoader] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            loss_fn: 损失函数
            optimizer: 优化器
            device: 设备
            scheduler: 学习率调度器
            save_dir: 检查点保存目录
            early_stopping_patience: 早停耐心值
            verbose: 是否打印详细信息
            auto_evaluate: 训练完成后是否自动评估（包含可视化）
            test_loader: 测试数据加载器（用于自动评估）
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.auto_evaluate = auto_evaluate
        self.test_loader = test_loader
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": []
        }
    
    def train_epoch(self) -> float:
        """
        训练一个 epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失（如果损失函数需要input_data，则传递）
            try:
                loss = self.loss_fn(outputs, targets, inputs)
            except TypeError:
                # 如果损失函数不接受input_data参数，使用普通调用
                loss = self.loss_fn(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """
        验证
        
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # 计算损失（如果损失函数需要input_data，则传递）
                try:
                    loss = self.loss_fn(outputs, targets, inputs)
                except TypeError:
                    # 如果损失函数不接受input_data参数，使用普通调用
                    loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs: int) -> Dict[str, list]:
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            
        Returns:
            训练历史
        """
        if self.verbose:
            print(f"\n开始训练 - 设备: {self.device}")
            print(f"训练样本: {len(self.train_loader.dataset)}")
            print(f"验证样本: {len(self.val_loader.dataset)}")
            print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)
            
            # 打印信息
            epoch_time = time.time() - start_time
            if self.verbose:
                print(f"Epoch {self.current_epoch}/{num_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth")
                if self.verbose:
                    print(f"  → 保存最佳模型 (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"\n早停触发 - {self.early_stopping_patience} 个 epoch 无改善")
                break
            
            # 定期保存检查点
            if (self.current_epoch) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch}.pth")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"训练完成！最佳验证损失: {self.best_val_loss:.6f}")
            print(f"{'='*60}\n")
        
        # 自动评估
        if self.auto_evaluate and self.test_loader is not None:
            self._auto_evaluate()
        
        return self.history
    
    def _auto_evaluate(self) -> None:
        """
        训练完成后自动评估（包含可视化）
        """
        if self.verbose:
            print("开始自动评估...")
        
        # 加载最佳模型
        best_model_path = self.save_dir / "best_model.pth"
        if best_model_path.exists():
            self.model.load_checkpoint(str(best_model_path))
            if self.verbose:
                print(f"已加载最佳模型: {best_model_path}")
        
        # 导入评估器
        from model.evaluator import Evaluator
        
        # 创建评估器
        evaluator = Evaluator(self.model, device=self.device)
        
        # 评估
        if self.verbose:
            print("\n评估模型...")
        metrics = evaluator.evaluate(self.test_loader)
        evaluator.print_metrics(metrics)
        
        # 可视化
        vis_dir = self.save_dir.parent / "visualizations"
        if self.verbose:
            print(f"\n生成可视化...")
        evaluator.visualize_predictions(output_dir=str(vis_dir))
        
        if self.verbose:
            print(f"\n评估完成！")
            print(f"  模型: {best_model_path}")
            print(f"  可视化: {vis_dir}")
    
    def save_checkpoint(self, filename: str) -> None:
        """
        保存检查点
        
        Args:
            filename: 文件名
        """
        checkpoint_path = self.save_dir / filename
        
        self.model.save_checkpoint(
            path=str(checkpoint_path),
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict(),
            metrics={
                "best_val_loss": self.best_val_loss,
                "train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
                "val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
            }
        )
    
    def load_checkpoint(self, filename: str) -> None:
        """
        加载检查点
        
        Args:
            filename: 文件名
        """
        checkpoint_path = self.save_dir / filename
        
        epoch, optimizer_state, metrics = self.model.load_checkpoint(
            path=str(checkpoint_path),
            load_optimizer=True
        )
        
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        
        if metrics is not None:
            self.best_val_loss = metrics.get("best_val_loss", float("inf"))
        
        self.current_epoch = epoch


if __name__ == "__main__":
    # 测试训练器
    print("测试训练器:\n")
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from model.base import VectorModel
    from model.dataset import get_dataloader
    from model.loss import get_loss_fn
    
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
    
    # 准备数据
    train_loader = get_dataloader("train", batch_size=32)
    val_loader = get_dataloader("val", batch_size=32)
    
    # 创建模型和训练器
    model = SimpleMLP()
    loss_fn = get_loss_fn("mse")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        save_dir="tests/temp/checkpoints",
        early_stopping_patience=3,
        verbose=True
    )
    
    # 训练 5 个 epoch
    history = trainer.train(num_epochs=5)
    
    print("\n训练历史:")
    print(f"  训练损失: {history['train_loss']}")
    print(f"  验证损失: {history['val_loss']}")
