"""损失函数定义"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MSELoss(nn.Module):
    """均方误差损失"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3) - [x, y, r]
            target: 目标值 (batch_size, 3) - [x, y, r]
            
        Returns:
            损失值
        """
        return self.mse(pred, target)


class MAELoss(nn.Module):
    """平均绝对误差损失"""
    
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3)
            target: 目标值 (batch_size, 3)
            
        Returns:
            损失值
        """
        return self.mae(pred, target)


class WeightedMSELoss(nn.Module):
    """加权均方误差损失 - 对 x, y, r 使用不同权重"""
    
    def __init__(self, weights: Optional[list] = None):
        """
        Args:
            weights: [w_x, w_y, w_r] 权重列表，默认 [1.0, 1.0, 1.0]
        """
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3)
            target: 目标值 (batch_size, 3)
            
        Returns:
            损失值
        """
        # 将权重移到相同设备
        weights = self.weights.to(pred.device)
        
        # 计算加权 MSE
        squared_diff = (pred - target) ** 2
        weighted_loss = squared_diff * weights
        return weighted_loss.mean()


class CircleLoss(nn.Module):
    """
    圆形损失 - 分别考虑圆心距离和半径差异
    
    loss = alpha * center_distance + beta * radius_diff
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Args:
            alpha: 圆心距离权重
            beta: 半径差异权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3) - [x, y, r]
            target: 目标值 (batch_size, 3) - [x, y, r]
            
        Returns:
            损失值
        """
        # 圆心距离 (欧氏距离)
        center_pred = pred[:, :2]  # (batch_size, 2)
        center_target = target[:, :2]
        center_distance = torch.sqrt(((center_pred - center_target) ** 2).sum(dim=1))
        
        # 半径差异 (绝对值)
        radius_pred = pred[:, 2]  # (batch_size,)
        radius_target = target[:, 2]
        radius_diff = torch.abs(radius_pred - radius_target)
        
        # 加权组合
        loss = self.alpha * center_distance.mean() + self.beta * radius_diff.mean()
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失 - MSE + Circle Loss
    
    loss = lambda1 * MSE + lambda2 * CircleLoss
    """
    
    def __init__(
        self, 
        lambda1: float = 1.0, 
        lambda2: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        """
        Args:
            lambda1: MSE 权重
            lambda2: CircleLoss 权重
            alpha: CircleLoss 中圆心距离权重
            beta: CircleLoss 中半径差异权重
        """
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mse_loss = MSELoss()
        self.circle_loss = CircleLoss(alpha=alpha, beta=beta)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3)
            target: 目标值 (batch_size, 3)
            
        Returns:
            损失值
        """
        mse = self.mse_loss(pred, target)
        circle = self.circle_loss(pred, target)
        return self.lambda1 * mse + self.lambda2 * circle


class ConstrainedLoss(nn.Module):
    """
    带约束的损失函数 - 确保下一级圈完全在上一级圈内
    
    约束条件：distance(center_pred, center_prev) + radius_pred <= radius_prev
    
    loss = base_loss + lambda_constraint * constraint_penalty
    """
    
    def __init__(
        self,
        base_loss: nn.Module = None,
        lambda_constraint: float = 10.0,
        grid_size: int = 16384
    ):
        """
        Args:
            base_loss: 基础损失函数（如MSE），默认使用MSE
            lambda_constraint: 约束惩罚权重
            grid_size: 坐标系大小，用于归一化
        """
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else MSELoss()
        self.lambda_constraint = lambda_constraint
        self.grid_size = grid_size
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        input_data: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3) - [x, y, r]
            target: 目标值 (batch_size, 3) - [x, y, r]
            input_data: 输入数据 (batch_size, 3 或 6)
                - 如果是3维: [x1, y1, r1] - 预测Ring2
                - 如果是6维: [x1, y1, r1, x2, y2, r2] - 预测Ring3
            
        Returns:
            损失值
        """
        # 基础损失
        base = self.base_loss(pred, target)
        
        # 如果没有提供输入数据，只返回基础损失
        if input_data is None:
            return base
        
        # 计算约束惩罚
        constraint_penalty = self._compute_constraint_penalty(pred, input_data)
        
        # 总损失
        total_loss = base + self.lambda_constraint * constraint_penalty
        
        return total_loss
    
    def _compute_constraint_penalty(
        self, 
        pred: torch.Tensor, 
        input_data: torch.Tensor
    ) -> torch.Tensor:
        """
        计算约束惩罚
        
        约束：distance(center_pred, center_prev) + radius_pred <= radius_prev
        惩罚：max(0, distance + radius_pred - radius_prev)^2
        """
        batch_size = pred.shape[0]
        input_dim = input_data.shape[1]
        
        # 提取预测的圆心和半径
        center_pred = pred[:, :2]  # (batch_size, 2)
        radius_pred = pred[:, 2]   # (batch_size,)
        
        # 根据输入维度确定上一级圈的信息
        if input_dim == 3:
            # 输入是Ring1，预测Ring2
            center_prev = input_data[:, :2]
            radius_prev = input_data[:, 2]
        elif input_dim == 6:
            # 输入是Ring1+Ring2，预测Ring3，上一级是Ring2
            center_prev = input_data[:, 3:5]
            radius_prev = input_data[:, 5]
        else:
            # 无法处理的输入维度，返回0惩罚
            return torch.tensor(0.0, device=pred.device)
        
        # 计算圆心距离
        center_distance = torch.sqrt(((center_pred - center_prev) ** 2).sum(dim=1))
        
        # 计算违反约束的程度
        # violation = distance + radius_pred - radius_prev
        # 如果 violation > 0，说明预测的圈超出了上一级圈
        violation = center_distance + radius_pred - radius_prev
        
        # 使用ReLU确保只惩罚违反约束的情况
        penalty = torch.relu(violation) ** 2
        
        return penalty.mean()


def get_loss_fn(loss_type: str = "mse", **kwargs):
    """
    获取损失函数
    
    Args:
        loss_type: 损失函数类型
            - "mse": 均方误差
            - "mae": 平均绝对误差
            - "weighted_mse": 加权均方误差
            - "circle": 圆形损失
            - "combined": 组合损失
            - "constrained": 带约束的损失（推荐用于毒圈预测）
        **kwargs: 损失函数参数
        
    Returns:
        损失函数实例
    """
    loss_dict = {
        "mse": MSELoss,
        "mae": MAELoss,
        "weighted_mse": WeightedMSELoss,
        "circle": CircleLoss,
        "combined": CombinedLoss,
        "constrained": ConstrainedLoss,
    }
    
    if loss_type not in loss_dict:
        raise ValueError(f"未知的损失函数类型: {loss_type}")
    
    return loss_dict[loss_type](**kwargs)


if __name__ == "__main__":
    # 测试损失函数
    print("测试损失函数:\n")
    
    # 模拟数据
    pred = torch.tensor([[0.5, 0.6, 0.3], [0.7, 0.8, 0.2]])
    target = torch.tensor([[0.6, 0.7, 0.35], [0.65, 0.75, 0.25]])
    
    # 测试各种损失
    losses = {
        "MSE": get_loss_fn("mse"),
        "MAE": get_loss_fn("mae"),
        "Weighted MSE": get_loss_fn("weighted_mse", weights=[2.0, 2.0, 1.0]),
        "Circle": get_loss_fn("circle", alpha=1.0, beta=0.5),
        "Combined": get_loss_fn("combined", lambda1=1.0, lambda2=0.5),
    }
    
    for name, loss_fn in losses.items():
        loss = loss_fn(pred, target)
        print(f"{name:15s}: {loss.item():.6f}")
