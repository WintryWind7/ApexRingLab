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
        coordinate_mode: str = "relative",
        map_filter: Optional[str] = None,
        use_onehot: bool = False
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
            coordinate_mode: 坐标模式 ('absolute' 或 'relative')
            map_filter: 地图过滤器（用于分地图模型验证）
            use_onehot: 是否使用One-Hot地图编码
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
        self.coordinate_mode = coordinate_mode
        self.map_filter = map_filter
        self.use_onehot = use_onehot
        
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
    
    def validate(self) -> tuple[float, float, float]:
        """
        验证
        
        Returns:
            (平均验证损失, 场景1 Ring3误差(px), 场景2 Ring3误差(px))
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # 计算损失
                try:
                    loss = self.loss_fn(outputs, targets, inputs)
                except TypeError:
                    loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # 计算场景误差（使用简化的评估）
        scenario1_error, scenario2_error = self._compute_scenario_errors()
        
        return avg_loss, scenario1_error, scenario2_error
    
    def _compute_scenario_errors(self) -> tuple[float, float]:
        """
        计算两个场景的Ring3圆心误差
        
        Returns:
            (场景1误差(px), 场景2误差(px))
        """
        import json
        from pathlib import Path
        
        # 地图到One-Hot的映射
        MAP_TO_ONEHOT = {
            "mp_rr_district": [1.0, 0.0],
            "mp_rr_tropic": [0.0, 1.0]
        }
        
        # 读取验证集原始数据
        data_dir = Path(__file__).parent.parent / "data" / "use"
        val_data_path = data_dir / "val.json"
        
        with open(val_data_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        
        scenario1_errors = []
        scenario2_errors = []
        grid_size = 16384
        
        with torch.no_grad():
            for item in val_data:
                map_name = item.get("map", "")
                
                # 排除的地图
                if map_name in ["mp_rr_desertlands_hu"]:
                    continue
                
                # 地图过滤（用于分地图模型）
                if self.map_filter and map_name != self.map_filter:
                    continue
                
                rings = item.get("rings", [])
                if len(rings) < 3:
                    continue
                
                # 归一化
                x1, y1, r1 = rings[0]["x"] / grid_size, rings[0]["y"] / grid_size, rings[0]["r"] / grid_size
                x2, y2, r2 = rings[1]["x"] / grid_size, rings[1]["y"] / grid_size, rings[1]["r"] / grid_size
                x3, y3, r3 = rings[2]["x"] / grid_size, rings[2]["y"] / grid_size, rings[2]["r"] / grid_size
                
                ring1 = torch.tensor([x1, y1, r1], dtype=torch.float32).to(self.device)
                x3, y3, r3 = rings[2]["x"] / grid_size, rings[2]["y"] / grid_size, rings[2]["r"] / grid_size
                
                ring1 = torch.tensor([x1, y1, r1], dtype=torch.float32).to(self.device)
                
                # One-Hot编码（如果需要）
                if self.use_onehot:
                    if map_name in MAP_TO_ONEHOT:
                        map_onehot = torch.tensor(MAP_TO_ONEHOT[map_name], dtype=torch.float32).to(self.device)
                    else:
                        # 未知地图，跳过
                        continue
                
                if self.coordinate_mode == "absolute":
                    # 绝对坐标模式
                    ring2_true = torch.tensor([x2, y2, r2], dtype=torch.float32).to(self.device)
                    ring3_true = torch.tensor([x3, y3, r3], dtype=torch.float32)
                    
                    # 场景1：只给Ring1，预测Ring2再预测Ring3
                    if self.use_onehot:
                        input1 = torch.cat([ring1, torch.zeros(3).to(self.device), map_onehot]).unsqueeze(0)
                    else:
                        input1 = torch.cat([ring1, torch.zeros(3).to(self.device)]).unsqueeze(0)
                    
                    ring2_pred = self.model(input1).squeeze(0)
                    
                    if self.use_onehot:
                        input2 = torch.cat([ring1, ring2_pred, map_onehot]).unsqueeze(0)
                    else:
                        input2 = torch.cat([ring1, ring2_pred]).unsqueeze(0)
                    
                    ring3_pred_s1 = self.model(input2).squeeze(0).cpu()
                    
                    # 计算场景1的Ring3误差
                    center_error_s1 = torch.sqrt(((ring3_pred_s1[:2] - ring3_true[:2]) ** 2).sum()).item() * grid_size
                    scenario1_errors.append(center_error_s1)
                    
                    # 场景2：给Ring1+Ring2，预测Ring3
                    if self.use_onehot:
                        input3 = torch.cat([ring1, ring2_true, map_onehot]).unsqueeze(0)
                    else:
                        input3 = torch.cat([ring1, ring2_true]).unsqueeze(0)
                    
                    ring3_pred_s2 = self.model(input3).squeeze(0).cpu()
                    
                    # 计算场景2的Ring3误差
                    center_error_s2 = torch.sqrt(((ring3_pred_s2[:2] - ring3_true[:2]) ** 2).sum()).item() * grid_size
                    scenario2_errors.append(center_error_s2)
                
                elif self.coordinate_mode == "relative":
                    # 相对坐标模式
                    # 场景1：只给Ring1，预测Ring2（相对坐标）再预测Ring3（相对坐标）
                    if self.use_onehot:
                        input1 = torch.cat([ring1, torch.zeros(3).to(self.device), map_onehot]).unsqueeze(0)
                    else:
                        input1 = torch.cat([ring1, torch.zeros(3).to(self.device)]).unsqueeze(0)
                    
                    ring2_pred_rel = self.model(input1).squeeze(0)  # [dx2, dy2, r2]
                    
                    # 转换为绝对坐标（保持在GPU上用于下一步预测）
                    ring2_pred_abs_x = ring1[0].item() + ring2_pred_rel[0].item()
                    ring2_pred_abs_y = ring1[1].item() + ring2_pred_rel[1].item()
                    
                    # 预测Ring3（相对Ring2）
                    if self.use_onehot:
                        input2 = torch.cat([ring1, ring2_pred_rel, map_onehot]).unsqueeze(0)
                    else:
                        input2 = torch.cat([ring1, ring2_pred_rel]).unsqueeze(0)
                    
                    ring3_pred_rel = self.model(input2).squeeze(0)  # [dx3, dy3, r3] 相对Ring2
                    
                    # 转换为绝对坐标
                    ring3_pred_abs_x = ring2_pred_abs_x + ring3_pred_rel[0].item()
                    ring3_pred_abs_y = ring2_pred_abs_y + ring3_pred_rel[1].item()
                    
                    # 计算场景1的Ring3误差
                    center_error_s1 = ((ring3_pred_abs_x - x3) ** 2 + (ring3_pred_abs_y - y3) ** 2) ** 0.5 * grid_size
                    scenario1_errors.append(center_error_s1)
                    
                    # 场景2：给Ring1+Ring2（真实），预测Ring3
                    dx2_true, dy2_true = x2 - x1, y2 - y1
                    ring2_true_rel = torch.tensor([dx2_true, dy2_true, r2], dtype=torch.float32).to(self.device)
                    
                    if self.use_onehot:
                        input3 = torch.cat([ring1, ring2_true_rel, map_onehot]).unsqueeze(0)
                    else:
                        input3 = torch.cat([ring1, ring2_true_rel]).unsqueeze(0)
                    
                    ring3_pred_rel_s2 = self.model(input3).squeeze(0)  # [dx3, dy3, r3] 相对Ring2
                    
                    # 转换为绝对坐标
                    ring3_pred_abs_x_s2 = x2 + ring3_pred_rel_s2[0].item()
                    ring3_pred_abs_y_s2 = y2 + ring3_pred_rel_s2[1].item()
                    
                    # 计算场景2的Ring3误差
                    center_error_s2 = ((ring3_pred_abs_x_s2 - x3) ** 2 + (ring3_pred_abs_y_s2 - y3) ** 2) ** 0.5 * grid_size
                    scenario2_errors.append(center_error_s2)
        
        avg_s1 = sum(scenario1_errors) / len(scenario1_errors) if scenario1_errors else 0
        avg_s2 = sum(scenario2_errors) / len(scenario2_errors) if scenario2_errors else 0
        
        return avg_s1, avg_s2
    
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
        
        # 创建训练日志文件
        log_file = self.save_dir / "training_log.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("Epoch\tTrain\tVal\tR3(S1)\tR3(S2)\tBest\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, scenario1_error, scenario2_error = self.validate()
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)
            
            # 保存最佳模型
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth")
                is_best = True
            else:
                self.patience_counter += 1
            
            # 打印信息
            if self.verbose:
                best_mark = " ✓" if is_best else ""
                print(f"Epoch {self.current_epoch:3d}/{num_epochs}\t"
                      f"Train: {train_loss:.6f}\t"
                      f"Val: {val_loss:.6f}\t"
                      f"R3(S1): {scenario1_error:4.0f}px\t"
                      f"R3(S2): {scenario2_error:4.0f}px"
                      f"{best_mark}")
            
            # 写入日志
            with open(log_file, "a", encoding="utf-8") as f:
                best_mark = "✓" if is_best else ""
                f.write(f"{self.current_epoch:3d}\t"
                       f"{train_loss:.6f}\t"
                       f"{val_loss:.6f}\t"
                       f"{scenario1_error:4.0f}\t"
                       f"{scenario2_error:4.0f}\t"
                       f"{best_mark}\n")
            
            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"\n早停触发 - {self.early_stopping_patience} 个 epoch 无改善")
                break
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"训练完成！最佳验证损失: {self.best_val_loss:.6f}")
            print(f"{'='*60}\n")
        
        return self.history
    
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
