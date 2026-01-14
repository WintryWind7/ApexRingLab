"""
将可视化结果拼接成大图

生成2张拼图：
- 场景1（ring2_3）：所有地图的图片拼在一起
- 场景2（ring3）：所有地图的图片拼在一起
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List


def merge_visualizations(vis_dir: str = "experiments/mlp_baseline/visualizations") -> None:
    """
    将可视化结果拼接成大图
    
    生成2张拼图：
    - scenario1_merged.png: 场景1（预测Ring2+Ring3）
    - scenario2_merged.png: 场景2（预测Ring3）
    
    Args:
        vis_dir: 可视化目录
    """
    vis_path = Path(vis_dir)
    ring2_3_dir = vis_path / "ring2_3"
    ring3_dir = vis_path / "ring3"
    
    if not ring2_3_dir.exists() or not ring3_dir.exists():
        print("可视化目录不存在")
        return
    
    # 获取所有图片（排序确保顺序一致）
    ring2_3_files = sorted(ring2_3_dir.glob("*.png"))
    ring3_files = sorted(ring3_dir.glob("*.png"))
    
    print(f"\n找到 {len(ring2_3_files)} 张场景1图片")
    print(f"找到 {len(ring3_files)} 张场景2图片")
    
    # 读取图片
    ring2_3_imgs = [cv2.imread(str(f)) for f in ring2_3_files]
    ring3_imgs = [cv2.imread(str(f)) for f in ring3_files]
    
    # 过滤掉读取失败的图片
    ring2_3_imgs = [img for img in ring2_3_imgs if img is not None]
    ring3_imgs = [img for img in ring3_imgs if img is not None]
    
    if not ring2_3_imgs or not ring3_imgs:
        print("没有有效的图片")
        return
    
    # 创建输出目录
    output_dir = vis_path / "merged"
    output_dir.mkdir(exist_ok=True)
    
    # 拼接场景1
    print(f"\n拼接场景1（{len(ring2_3_imgs)} 张图片）...")
    scenario1_merged = merge_images_grid(ring2_3_imgs, cols=5)
    if scenario1_merged is not None:
        output_file = output_dir / "scenario1_merged.png"
        cv2.imwrite(str(output_file), scenario1_merged)
        print(f"✓ 场景1保存到: {output_file}")
    
    # 拼接场景2
    print(f"\n拼接场景2（{len(ring3_imgs)} 张图片）...")
    scenario2_merged = merge_images_grid(ring3_imgs, cols=5)
    if scenario2_merged is not None:
        output_file = output_dir / "scenario2_merged.png"
        cv2.imwrite(str(output_file), scenario2_merged)
        print(f"✓ 场景2保存到: {output_file}")
    
    print(f"\n拼图完成！保存到: {output_dir}")


def merge_images_grid(imgs: List[np.ndarray], cols: int = 5) -> np.ndarray:
    """
    将图片拼接成网格（自动计算行数）
    
    Args:
        imgs: 图片列表
        cols: 每行的列数
        
    Returns:
        拼接后的图片
    """
    if not imgs:
        return None
    
    # 计算行数
    rows = (len(imgs) + cols - 1) // cols
    
    # 获取图片尺寸（假设所有图片尺寸相同）
    h, w = imgs[0].shape[:2]
    
    # 如果图片数量不足填满网格，用黑色图片填充
    while len(imgs) < rows * cols:
        imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    # 按行拼接
    row_imgs = []
    for i in range(rows):
        row_start = i * cols
        row_end = min(row_start + cols, len(imgs))
        row = np.hstack(imgs[row_start:row_end])
        row_imgs.append(row)
    
    # 拼接所有行
    merged = np.vstack(row_imgs)
    
    return merged


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        vis_dir = sys.argv[1]
    else:
        # 默认使用baseline的可视化目录
        vis_dir = "experiments/mlp_baseline/visualizations"
    
    merge_visualizations(vis_dir)
