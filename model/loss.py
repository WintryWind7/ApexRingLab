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


class DirectionalCircleLoss(nn.Module):
    """
    方向圆形损失 - 考虑圆心距离、半径差异和方向一致性
    
    loss = alpha * center_distance + beta * radius_diff + gamma * direction_loss
    
    方向损失使用余弦相似度，鼓励预测方向与真实方向一致
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 1.0, gamma: float = 1.0):
        """
        Args:
            alpha: 圆心距离权重
            beta: 半径差异权重
            gamma: 方向损失权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3) - [dx, dy, r] (相对坐标)
            target: 目标值 (batch_size, 3) - [dx, dy, r] (相对坐标)
            
        Returns:
            损失值
        """
        # 圆心距离 (欧氏距离)
        center_pred = pred[:, :2]  # (batch_size, 2)
        center_target = target[:, :2]
        center_distance = torch.sqrt(((center_pred - center_target) ** 2).sum(dim=1) + 1e-8)
        
        # 半径差异 (绝对值)
        radius_pred = pred[:, 2]  # (batch_size,)
        radius_target = target[:, 2]
        radius_diff = torch.abs(radius_pred - radius_target)
        
        # 方向损失 (余弦相似度)
        # 归一化方向向量
        pred_norm = torch.sqrt((center_pred ** 2).sum(dim=1, keepdim=True) + 1e-8)
        target_norm = torch.sqrt((center_target ** 2).sum(dim=1, keepdim=True) + 1e-8)
        
        pred_direction = center_pred / pred_norm  # (batch_size, 2)
        target_direction = center_target / target_norm  # (batch_size, 2)
        
        # 余弦相似度 (范围[-1, 1]，1表示方向完全一致)
        cos_similarity = (pred_direction * target_direction).sum(dim=1)
        
        # 方向损失 (范围[0, 2]，0表示方向完全一致，2表示完全相反)
        direction_loss = 1 - cos_similarity
        
        # 加权组合
        loss = (
            self.alpha * center_distance.mean() + 
            self.beta * radius_diff.mean() + 
            self.gamma * direction_loss.mean()
        )
        return loss


class EdgeConstraintLoss(nn.Module):
    """
    贴边约束损失 - 鼓励预测的圆心距离接近理想的贴边距离
    
    loss = alpha * center_distance + beta * radius_diff + gamma * edge_constraint
    
    edge_constraint = |predicted_distance - (r_prev - r_current)|
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 1.0, gamma: float = 1.0):
        """
        Args:
            alpha: 圆心距离权重
            beta: 半径差异权重
            gamma: 贴边约束权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        input_data: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3) - [dx, dy, r] (相对坐标)
            target: 目标值 (batch_size, 3) - [dx, dy, r] (相对坐标)
            input_data: 输入数据，需要包含上一级圈的半径
            
        Returns:
            损失值
        """
        # 圆心距离 (欧氏距离)
        center_pred = pred[:, :2]  # (batch_size, 2)
        center_target = target[:, :2]
        center_distance = torch.sqrt(((center_pred - center_target) ** 2).sum(dim=1) + 1e-8)
        
        # 半径差异 (绝对值)
        radius_pred = pred[:, 2]  # (batch_size,)
        radius_target = target[:, 2]
        radius_diff = torch.abs(radius_pred - radius_target)
        
        # 贴边约束
        edge_constraint = torch.tensor(0.0, device=pred.device)
        if input_data is not None:
            # 从input_data提取上一级圈的半径
            # 假设格式：[map1, map2, x1, y1, r1, dx2, dy2, r2, ...]
            if input_data.shape[1] >= 5:
                r_prev = input_data[:, 4]  # r1
                
                # 计算预测的距离
                pred_distance = torch.sqrt((center_pred ** 2).sum(dim=1) + 1e-8)
                
                # 理想的贴边距离（使用真实半径）
                ideal_distance = torch.clamp(r_prev - radius_target, min=0.0)
                
                # 贴边约束：预测距离应接近理想距离
                edge_constraint = torch.abs(pred_distance - ideal_distance).mean()
        
        # 加权组合
        loss = (
            self.alpha * center_distance.mean() + 
            self.beta * radius_diff.mean() + 
            self.gamma * edge_constraint
        )
        return loss


class HuberCircleLoss(nn.Module):
    """
    Huber圆形损失 - 对小误差用L2，大误差用L1
    
    减少方向错误时的巨大惩罚，让模型敢于尝试贴边
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 1.0, delta: float = 1.0):
        """
        Args:
            alpha: 圆心距离权重
            beta: 半径差异权重
            delta: Huber损失的阈值
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3) - [dx, dy, r]
            target: 目标值 (batch_size, 3) - [dx, dy, r]
            
        Returns:
            损失值
        """
        # 圆心距离 (欧氏距离)
        center_pred = pred[:, :2]
        center_target = target[:, :2]
        center_distance = torch.sqrt(((center_pred - center_target) ** 2).sum(dim=1) + 1e-8)
        
        # Huber损失应用于圆心距离
        huber_center = torch.where(
            center_distance <= self.delta,
            0.5 * center_distance ** 2,
            self.delta * (center_distance - 0.5 * self.delta)
        )
        
        # 半径差异 (绝对值)
        radius_pred = pred[:, 2]
        radius_target = target[:, 2]
        radius_diff = torch.abs(radius_pred - radius_target)
        
        # 加权组合
        loss = self.alpha * huber_center.mean() + self.beta * radius_diff.mean()
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


class DistanceConstrainedLoss(nn.Module):
    """
    带距离约束的损失函数 - 鼓励模型学习贴边刷圈
    
    loss = base_loss + lambda_distance * distance_penalty
    
    distance_penalty惩罚那些预测距离与真实距离差异大的样本
    """
    
    def __init__(
        self,
        base_loss: nn.Module = None,
        lambda_distance: float = 1.0
    ):
        """
        Args:
            base_loss: 基础损失函数（如MSE），默认使用MSE
            lambda_distance: 距离约束权重
        """
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else MSELoss()
        self.lambda_distance = lambda_distance
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        input_data: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3) - [dx, dy, r] (相对坐标)
            target: 目标值 (batch_size, 3) - [dx, dy, r] (相对坐标)
            input_data: 输入数据 (batch_size, input_dim)
                需要包含上一级圈的半径信息来计算normalized_distance
            
        Returns:
            损失值
        """
        # 基础损失
        base = self.base_loss(pred, target)
        
        # 如果没有提供输入数据，只返回基础损失
        if input_data is None:
            return base
        
        # 计算距离约束惩罚
        distance_penalty = self._compute_distance_penalty(pred, target, input_data)
        
        # 总损失
        total_loss = base + self.lambda_distance * distance_penalty
        
        return total_loss
    
    def _compute_distance_penalty(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        input_data: torch.Tensor
    ) -> torch.Tensor:
        """
        计算距离约束惩罚
        
        惩罚预测的normalized_distance与真实的normalized_distance的差异
        normalized_distance = sqrt(dx² + dy²) / (r_prev - r_current)
        
        关键：使用真实半径计算，避免影响半径学习
        """
        # 提取预测和目标的相对坐标
        dx_pred, dy_pred = pred[:, 0], pred[:, 1]
        dx_target, dy_target, r_target = target[:, 0], target[:, 1], target[:, 2]
        
        # 从input_data中提取上一级圈的半径
        # 假设输入格式：[map1, map2, x1, y1, r1, dx2, dy2, r2, ...]
        input_dim = input_data.shape[1]
        
        # 根据输入维度判断
        if input_dim >= 5:
            r_prev = input_data[:, 4]  # r1的位置
            
            if input_dim >= 8:
                r_prev = input_data[:, 7]  # r2的位置
        else:
            return torch.tensor(0.0, device=pred.device)
        
        # 使用真实半径计算最大距离
        max_distance = torch.clamp(r_prev - r_target, min=0.01)
        
        # 计算预测的normalized_distance（基于真实半径）
        distance_pred = torch.sqrt(dx_pred**2 + dy_pred**2 + 1e-8)
        normalized_distance_pred = torch.clamp(distance_pred / max_distance, max=10.0)
        
        # 计算真实的normalized_distance（基于真实半径）
        distance_target = torch.sqrt(dx_target**2 + dy_target**2 + 1e-8)
        normalized_distance_target = torch.clamp(distance_target / max_distance, max=10.0)
        
        # 计算差异（MSE）- 只约束位置，不影响半径
        penalty = ((normalized_distance_pred - normalized_distance_target) ** 2).mean()
        
        return penalty


class RelativeLoss(nn.Module):
    """
    相对位置损失 - 同时优化圆心距离、角度和距离比例
    
    loss = alpha * center_distance + beta * angle_loss + gamma * distance_ratio_loss
    
    角度损失使用sin(angle_diff)，使得0度和180度（同一直线）损失都为0，
    90度（垂直）损失最大
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        """
        Args:
            alpha: 圆心距离权重
            beta: 角度损失权重
            gamma: 距离比例损失权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch_size, 3) - [dx, dy, r] (相对坐标)
            target: 目标值 (batch_size, 3) - [dx, dy, r] (相对坐标)
            
        Returns:
            损失值
        """
        dx_pred, dy_pred = pred[:, 0], pred[:, 1]
        dx_true, dy_true = target[:, 0], target[:, 1]
        r_pred = pred[:, 2]
        r_true = target[:, 2]
        
        # 1. 圆心距离损失（欧氏距离）
        center_distance = torch.sqrt((dx_pred - dx_true)**2 + (dy_pred - dy_true)**2 + 1e-8)
        center_loss = center_distance.mean()
        
        # 2. 角度损失（使用sin，0度和180度损失都为0）
        # 计算角度
        angle_pred = torch.atan2(dy_pred, dx_pred)
        angle_true = torch.atan2(dy_true, dx_true)
        
        # 角度差
        angle_diff = angle_pred - angle_true
        
        # 使用sin：sin(0°)=0, sin(90°)=1, sin(180°)=0
        # 这样0度和180度（同一直线）损失都为0，90度（垂直）损失最大
        angle_loss = torch.abs(torch.sin(angle_diff)).mean()
        
        # 3. 距离比例损失
        dist_pred = torch.sqrt(dx_pred**2 + dy_pred**2 + 1e-8)
        dist_true = torch.sqrt(dx_true**2 + dy_true**2 + 1e-8)
        
        # 距离比例
        ratio = dist_pred / dist_true
        # 比例误差：|ratio - 1|
        distance_ratio_loss = torch.abs(ratio - 1.0).mean()
        
        # 4. 半径损失
        radius_loss = torch.abs(r_pred - r_true).mean()
        
        # 组合损失
        total_loss = (
            self.alpha * center_loss + 
            self.beta * angle_loss + 
            self.gamma * distance_ratio_loss +
            radius_loss  # 半径损失始终包含
        )
        
        return total_loss


def get_loss_fn(loss_type: str = "mse", **kwargs):
    """
    获取损失函数
    
    Args:
        loss_type: 损失函数类型
            - "mse": 均方误差
            - "mae": 平均绝对误差
            - "weighted_mse": 加权均方误差
            - "circle": 圆形损失
            - "directional_circle": 方向圆形损失（CircleLoss + 方向损失）
            - "edge_constraint": 贴边约束损失（CircleLoss + 贴边约束）
            - "huber_circle": Huber圆形损失（对大误差更宽容）
            - "combined": 组合损失
            - "constrained": 带约束的损失（确保圈在上一级圈内）
            - "distance_constrained": 带距离约束的损失（鼓励学习贴边刷圈）
        **kwargs: 损失函数参数
        
    Returns:
        损失函数实例
    """
    loss_dict = {
        "mse": MSELoss,
        "mae": MAELoss,
        "weighted_mse": WeightedMSELoss,
        "circle": CircleLoss,
        "directional_circle": DirectionalCircleLoss,
        "edge_constraint": EdgeConstraintLoss,
        "huber_circle": HuberCircleLoss,
        "combined": CombinedLoss,
        "constrained": ConstrainedLoss,
        "distance_constrained": DistanceConstrainedLoss,
        "relative": RelativeLoss,
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
