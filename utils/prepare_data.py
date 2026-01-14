"""数据准备工具 - 分割训练/验证/测试集"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import re
import cv2
import numpy as np

# ==================== 配置区域 ====================
# 获取项目根目录（prepare_data.py在utils/目录下，需要向上一级）
PROJECT_ROOT = Path(__file__).parent.parent

# 输入文件
INPUT_FILE = PROJECT_ROOT / "data" / "crawler" / "public_rings.jsonl"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "data" / "use"

# 数据集分割比例
TRAIN_RATIO = 0.7  # 训练集 70%
VAL_RATIO = 0.15   # 验证集 15%
TEST_RATIO = 0.15  # 测试集 15%

# 随机种子（保证可复现）
RANDOM_SEED = 42

# 地图名标准化规则
# 移除这些后缀，统一为 ALGS 标准地图名
MAP_SUFFIXES_TO_REMOVE = [
    r"_island",
    r"_mu\d+",
    r"_landscape",
]

# 地图文件路径映射
MAP_FILES = {
    "mp_rr_desertlands_hu": PROJECT_ROOT / "data" / "map" / "mp_rr_desertlands_hu.png",
    "mp_rr_district": PROJECT_ROOT / "data" / "map" / "mp_rr_district.png",
    "mp_rr_tropic": PROJECT_ROOT / "data" / "map" / "mp_rr_tropic_island_mu2.png",
}

# 地图名简称映射
MAP_SHORT_NAMES = {
    "mp_rr_desertlands_hu": "desertlands",
    "mp_rr_district": "district",
    "mp_rr_tropic": "tropic",
}

# 测试样本配置
TEST_SAMPLES_PER_MAP = 5  # 每个地图抽取的测试样本数
GRID_SIZE = 16384  # 坐标系大小
# ==================================================


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    加载 JSONL 文件
    
    Args:
        file_path: JSONL 文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def normalize_map_name(map_name: str) -> str:
    """
    标准化地图名，移除 PUBLIC 模式的后缀
    
    Args:
        map_name: 原始地图名
        
    Returns:
        标准化后的地图名
    """
    normalized = map_name
    for suffix_pattern in MAP_SUFFIXES_TO_REMOVE:
        normalized = re.sub(suffix_pattern, "", normalized)
    return normalized


def normalize_dataset(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    标准化数据集中的地图名
    
    Args:
        data: 原始数据
        
    Returns:
        标准化后的数据
    """
    normalized_data = []
    map_changes = {}
    
    for item in data:
        item_copy = item.copy()
        original_map = item_copy.get("map", "")
        normalized_map = normalize_map_name(original_map)
        
        if original_map != normalized_map:
            map_changes[original_map] = normalized_map
        
        item_copy["map"] = normalized_map
        normalized_data.append(item_copy)
    
    # 打印地图名变更
    if map_changes:
        print(f"\n地图名标准化:")
        for original, normalized in map_changes.items():
            print(f"  {original} → {normalized}")
    
    return normalized_data


def analyze_ring_distribution(data: List[Dict[str, Any]]) -> None:
    """
    分析各地图的毒圈数量分布和半径分布
    
    Args:
        data: 数据列表
    """
    from collections import defaultdict
    
    # 统计每个地图的毒圈数量分布
    map_ring_counts = defaultdict(lambda: defaultdict(int))
    
    # 统计每个地图每级毒圈的半径分布
    map_ring_radius = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for item in data:
        map_name = item.get("map", "unknown")
        rings = item.get("rings", [])
        ring_count = len(rings)
        map_ring_counts[map_name][ring_count] += 1
        
        # 记录每级毒圈的半径
        for i, ring in enumerate(rings, 1):
            radius = ring.get("r", 0)
            map_ring_radius[map_name][i][radius] += 1
    
    # 打印统计结果
    print(f"\n各地图毒圈数量分布:")
    for map_name in sorted(map_ring_counts.keys()):
        print(f"\n  {map_name}:")
        ring_dist = map_ring_counts[map_name]
        total = sum(ring_dist.values())
        for ring_count in sorted(ring_dist.keys()):
            count = ring_dist[ring_count]
            percentage = count / total * 100
            print(f"    {ring_count} 圈: {count} 条 ({percentage:.1f}%)")
        print(f"    总计: {total} 条")
    
    # 打印半径分布
    print(f"\n各地图各级毒圈半径分布:")
    for map_name in sorted(map_ring_radius.keys()):
        print(f"\n  {map_name}:")
        for ring_level in sorted(map_ring_radius[map_name].keys()):
            radius_dist = map_ring_radius[map_name][ring_level]
            total_count = sum(radius_dist.values())
            
            print(f"    Ring {ring_level}:")
            # 按数量降序排列
            sorted_radii = sorted(radius_dist.items(), key=lambda x: x[1], reverse=True)
            for radius, count in sorted_radii:
                percentage = count / total_count * 100
                status = "✓" if percentage > 99 else "✗"
                print(f"      r = {radius}: {count} 条 ({percentage:.1f}%) {status}")


def get_valid_radii(data: List[Dict[str, Any]]) -> Dict[str, Dict[int, int]]:
    """
    获取每个地图每级毒圈的有效半径（出现频率最高的）
    
    Args:
        data: 数据列表
        
    Returns:
        {map_name: {ring_level: valid_radius}}
    """
    from collections import defaultdict
    
    # 统计每个地图每级毒圈的半径分布
    map_ring_radius = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for item in data:
        map_name = item.get("map", "unknown")
        rings = item.get("rings", [])
        
        for i, ring in enumerate(rings, 1):
            radius = ring.get("r", 0)
            map_ring_radius[map_name][i][radius] += 1
    
    # 找出每个地图每级毒圈的有效半径（最常见的）
    valid_radii = {}
    for map_name, ring_levels in map_ring_radius.items():
        valid_radii[map_name] = {}
        for ring_level, radius_dist in ring_levels.items():
            # 找出出现次数最多的半径
            valid_radius = max(radius_dist.items(), key=lambda x: x[1])[0]
            valid_radii[map_name][ring_level] = valid_radius
    
    return valid_radii


def filter_anomalies(data: List[Dict[str, Any]], valid_radii: Dict[str, Dict[int, int]]) -> List[Dict[str, Any]]:
    """
    过滤异常数据（半径不符合标准的数据）
    
    Args:
        data: 原始数据
        valid_radii: 有效半径字典
        
    Returns:
        过滤后的数据
    """
    filtered_data = []
    removed_count = 0
    
    for item in data:
        map_name = item.get("map", "unknown")
        rings = item.get("rings", [])
        
        # 检查每级毒圈的半径是否有效
        is_valid = True
        for i, ring in enumerate(rings, 1):
            radius = ring.get("r", 0)
            expected_radius = valid_radii.get(map_name, {}).get(i)
            
            if expected_radius and radius != expected_radius:
                is_valid = False
                break
        
        if is_valid:
            filtered_data.append(item)
        else:
            removed_count += 1
    
    print(f"\n数据过滤:")
    print(f"  原始数据: {len(data)} 条")
    print(f"  移除异常: {removed_count} 条")
    print(f"  保留数据: {len(filtered_data)} 条")
    
    return filtered_data


def draw_rings_on_map(
    map_path: Path,
    rings: List[Dict[str, Any]],
    output_path: Path,
    grid_size: int = GRID_SIZE
) -> None:
    """
    在地图上绘制毒圈
    
    Args:
        map_path: 地图文件路径
        rings: 毒圈数据列表
        output_path: 输出图片路径
        grid_size: 坐标系大小
    """
    # 读取地图
    map_img = cv2.imread(str(map_path))
    if map_img is None:
        print(f"  ✗ 无法读取地图: {map_path}")
        return
    
    # 计算坐标转换比例
    scale = map_img.shape[0] / grid_size
    
    # 绘制每个毒圈（白色轮廓）
    for ring in rings:
        x_pixel = int(ring["x"] * scale)
        y_pixel = int(ring["y"] * scale)
        r_pixel = int(ring["r"] * scale)
        
        # 绘制白色圆圈
        cv2.circle(map_img, (x_pixel, y_pixel), r_pixel, (255, 255, 255), 2)
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), map_img)


def generate_test_samples(
    test_data: List[Dict[str, Any]],
    output_dir: Path,
    samples_per_map: int = TEST_SAMPLES_PER_MAP,
    random_seed: int = RANDOM_SEED
) -> None:
    """
    从测试集中生成可视化样本
    
    Args:
        test_data: 测试集数据
        output_dir: 输出目录
        samples_per_map: 每个地图抽取的样本数
        random_seed: 随机种子
    """
    from collections import defaultdict
    
    # 按地图分组，保留原始索引
    map_data = defaultdict(list)
    for idx, item in enumerate(test_data):
        map_name = item.get("map", "unknown")
        map_data[map_name].append((idx, item))
    
    # 创建输出目录
    test_rings_dir = output_dir / "test_rings"
    
    # 清空旧文件
    if test_rings_dir.exists():
        import shutil
        shutil.rmtree(test_rings_dir)
    
    test_rings_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n生成测试样本:")
    print(f"  输出目录: {test_rings_dir}")
    
    random.seed(random_seed)
    
    for map_name in sorted(map_data.keys()):
        items_with_idx = map_data[map_name]
        
        # 随机抽取样本（保留索引）
        sample_count = min(samples_per_map, len(items_with_idx))
        samples = random.sample(items_with_idx, sample_count)
        
        print(f"\n  {map_name}: 抽取 {sample_count} 个样本")
        
        # 获取地图文件路径
        map_path = MAP_FILES.get(map_name)
        if not map_path:
            print(f"    ✗ 未找到地图文件配置")
            continue
        
        if not map_path.exists():
            print(f"    ✗ 地图文件不存在: {map_path}")
            continue
        
        # 获取地图简称
        short_name = MAP_SHORT_NAMES.get(map_name, map_name)
        
        # 保存每个样本
        for original_idx, sample in samples:
            # 保存 JSON（使用完整地图名）
            json_filename = f"{map_name}_{original_idx}.json"
            json_path = test_rings_dir / json_filename
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
            
            # 绘制地图
            img_filename = f"{map_name}_{original_idx}.png"
            img_path = test_rings_dir / img_filename
            draw_rings_on_map(map_path, sample.get("rings", []), img_path)
            
            print(f"    ✓ {json_filename} + {img_filename}")
    
    print(f"\n✓ 测试样本生成完成")


def save_json(data: List[Dict[str, Any]], file_path: Path) -> None:
    """
    保存为 JSON 文件
    
    Args:
        data: 数据列表
        file_path: 输出文件路径
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_dataset(
    data: List[Dict[str, Any]], 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    分割数据集
    
    Args:
        data: 原始数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        
    Returns:
        (训练集, 验证集, 测试集)
    """
    # 检查比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1"
    
    # 打乱数据
    random.seed(random_seed)
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)
    
    # 计算分割点
    total = len(data_shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 分割
    train_data = data_shuffled[:train_end]
    val_data = data_shuffled[train_end:val_end]
    test_data = data_shuffled[val_end:]
    
    return train_data, val_data, test_data


def prepare_dataset(
    input_file: Path = INPUT_FILE,
    output_dir: Path = OUTPUT_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED
) -> None:
    """
    准备数据集：加载、分割、保存
    
    Args:
        input_file: 输入 JSONL 文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    # 检查输入文件
    if not input_file.exists():
        print(f"✗ 输入文件不存在: {input_file}")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空旧文件
    for old_file in output_dir.glob("*.json"):
        old_file.unlink()
    
    # 加载数据
    print(f"正在加载数据: {input_file}")
    data = load_jsonl(input_file)
    print(f"✓ 加载完成，共 {len(data)} 条数据")
    
    # 标准化地图名
    data = normalize_dataset(data)
    
    # 分析毒圈分布
    analyze_ring_distribution(data)
    
    # 获取有效半径并过滤异常数据
    valid_radii = get_valid_radii(data)
    data = filter_anomalies(data, valid_radii)
    
    # 分割数据集
    print(f"\n正在分割数据集...")
    print(f"训练集: {train_ratio*100:.0f}%, 验证集: {val_ratio*100:.0f}%, 测试集: {test_ratio*100:.0f}%")
    train_data, val_data, test_data = split_dataset(
        data, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    print(f"✓ 分割完成")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    # 保存数据集
    print(f"\n正在保存数据集到: {output_dir}")
    save_json(train_data, output_dir / "train.json")
    save_json(val_data, output_dir / "val.json")
    save_json(test_data, output_dir / "test.json")
    print(f"✓ 保存完成")
    
    # 生成测试样本
    generate_test_samples(test_data, output_dir)


if __name__ == "__main__":
    prepare_dataset()
