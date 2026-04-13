"""模型基类定义"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class BaseModel(nn.Module, ABC):
    """
    模型基类 - 定义统一接口
    
    所有模型必须继承此类并实现 forward 方法
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 必须由子类实现
        
        Args:
            x: 输入张量，格式由具体模型决定
               - MLP: (batch_size, input_dim)
               - CNN: (batch_size, channels, H, W)
               
        Returns:
            输出张量 (batch_size, output_dim)
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测接口 - 自动处理 eval 模式和 no_grad
        
        Args:
            x: 输入张量
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(
        self, 
        path: str, 
        epoch: int = 0,
        optimizer_state: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ) -> None:
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            epoch: 当前轮次
            optimizer_state: 优化器状态
            metrics: 评估指标
        """
        checkpoint = {
            "model_name": self.model_name,
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
        }
        
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        # 创建目录
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, path)
        # print(f"模型已保存到: {path}")  # 静默保存，避免输出混乱
    
    def load_checkpoint(
        self, 
        path: str, 
        load_optimizer: bool = False
    ) -> Tuple[int, Optional[Dict], Optional[Dict]]:
        """
        加载模型检查点
        
        Args:
            path: 检查点路径
            load_optimizer: 是否加载优化器状态
            
        Returns:
            (epoch, optimizer_state, metrics)
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        # 加载模型权重
        self.load_state_dict(checkpoint["model_state_dict"])
        
        epoch = checkpoint.get("epoch", 0)
        optimizer_state = checkpoint.get("optimizer_state_dict") if load_optimizer else None
        metrics = checkpoint.get("metrics")
        
        print(f"模型已从 {path} 加载 (epoch: {epoch})")
        
        return epoch, optimizer_state, metrics
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取模型配置 - 可选实现
        
        Returns:
            配置字典
        """
        return {
            "model_name": self.model_name,
            "num_parameters": self.get_num_parameters(),
        }
    
    def summary(self) -> None:
        """打印模型摘要"""
        print(f"\n{'='*50}")
        print(f"模型: {self.model_name}")
        print(f"{'='*50}")
        print(f"参数数量: {self.get_num_parameters():,}")
        print(f"{'='*50}\n")


class VectorModel(BaseModel):
    """
    向量输入模型基类 - 适用于 MLP、线性回归等
    
    输入: (batch_size, input_dim)
    输出: (batch_size, output_dim)
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config


class ImageModel(BaseModel):
    """
    图像输入模型基类 - 适用于 CNN 等
    
    输入: (batch_size, channels, H, W)
    输出: (batch_size, output_dim)
    """
    
    def __init__(self, in_channels: int, output_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "output_dim": self.output_dim,
        })
        return config


if __name__ == "__main__":
    # 测试基类
    print("测试模型基类:\n")
    
    # 创建一个简单的 MLP 示例
    class SimpleMLP(VectorModel):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__(input_dim, output_dim)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 测试
    model = SimpleMLP(input_dim=6, hidden_dim=64, output_dim=3)
    model.summary()
    
    # 测试前向传播
    x = torch.randn(4, 6)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # 测试保存/加载
    model.save_checkpoint("tests/temp/test_model.pth", epoch=10)
    epoch, _, _ = model.load_checkpoint("tests/temp/test_model.pth")
    print(f"加载的 epoch: {epoch}")
