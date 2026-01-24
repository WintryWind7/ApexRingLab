"""毒圈可视化服务器 - FastAPI"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import json
import importlib.util
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    import torch
except ImportError:
    print("警告: 未安装 torch，模型加载功能将不可用")
    torch = None

# 配置
DATA_DIR = PROJECT_ROOT / "data"
MAP_DIR = DATA_DIR / "map"
USE_DIR = DATA_DIR / "use"
SERVER_DIR = DATA_DIR / "server"
FULL_JSON = USE_DIR / "full.json"
MODELS_CONFIG = SERVER_DIR / "models_config.json"

# 地图文件映射
MAP_FILES = {
    "mp_rr_district": "mp_rr_district.png",
    "mp_rr_tropic": "mp_rr_tropic_island_mu2.png",
}

# 创建 FastAPI 应用
app = FastAPI(title="毒圈可视化工具")

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（地图图片）
app.mount("/maps", StaticFiles(directory=str(MAP_DIR)), name="maps")

# 全局变量：缓存数据集和模型
dataset_cache: Optional[List[Dict[str, Any]]] = None
models_cache: Dict[str, Any] = {}
models_config_cache: Optional[Dict[str, Any]] = None


def load_models_config() -> Dict[str, Any]:
    """加载模型配置"""
    global models_config_cache
    if models_config_cache is None:
        if MODELS_CONFIG.exists():
            with open(MODELS_CONFIG, "r", encoding="utf-8") as f:
                models_config_cache = json.load(f)
        else:
            models_config_cache = {"models": {}}
    return models_config_cache


def save_models_config(config: Dict[str, Any]) -> None:
    """保存模型配置"""
    global models_config_cache
    models_config_cache = config
    SERVER_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_dataset() -> List[Dict[str, Any]]:
    """加载数据集"""
    global dataset_cache
    if dataset_cache is None:
        with open(FULL_JSON, "r", encoding="utf-8") as f:
            dataset_cache = json.load(f)
    return dataset_cache


def load_model(model_name: str) -> Any:
    """
    动态加载模型的 Predictor
    
    Args:
        model_name: 模型名称，如 "mlp_baseline", "mlp_deep", "gan"
    """
    global models_cache
    
    if model_name in models_cache:
        return models_cache[model_name]
    
    try:
        import importlib.util
        import torch
        
        exp_dir = PROJECT_ROOT / "experiments" / model_name
        predictor_file = exp_dir / "predictor.py"
        checkpoints_dir = exp_dir / "checkpoints"
        
        if not predictor_file.exists():
            print(f"Predictor 文件不存在: {predictor_file}")
            return None
        
        # 1. 动态导入 predictor 模块
        spec = importlib.util.spec_from_file_location(f"{model_name}.predictor", predictor_file)
        if spec is None or spec.loader is None:
            print(f"无法加载模块: {predictor_file}")
            return None
        
        predictor_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = predictor_module
        
        # 临时添加实验目录到 sys.path（用于导入模型文件）
        exp_dir_str = str(exp_dir)
        if exp_dir_str not in sys.path:
            sys.path.insert(0, exp_dir_str)
        
        spec.loader.exec_module(predictor_module)
        
        # 2. 查找 Predictor 类
        predictor_class = None
        predictor_class_name = None
        
        for name in dir(predictor_module):
            if name.startswith('_') or name == 'Predictor':
                continue
            obj = getattr(predictor_module, name)
            
            if isinstance(obj, type) and hasattr(obj, 'predict') and callable(getattr(obj, 'predict')):
                predictor_class = obj
                predictor_class_name = name
                break
        
        if predictor_class is None:
            print(f"未找到 Predictor 类")
            return None
        
        print(f"✓ 找到 Predictor 类: {predictor_class_name}")
        
        # 3. 查找并加载 checkpoint
        checkpoint_files = list(checkpoints_dir.rglob("best_*.pth"))
        if not checkpoint_files:
            print(f"未找到 checkpoint 文件: {checkpoints_dir}")
            return None
        
        checkpoint_path = checkpoint_files[0]
        print(f"✓ 加载 checkpoint: {checkpoint_path.name}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # 4. 根据不同模型类型加载模型
        device = "cpu"  # 使用 CPU 进行推理
        
        if model_name == "mlp_baseline":
            from experiments.mlp_baseline.mlp_baseline import MLPBaseline
            model = MLPBaseline()
            model.load_state_dict(checkpoint["model_state_dict"])
            predictor = predictor_class(model, device=device)
            
        elif model_name == "mlp_deep":
            # mlp_deep 有两个模型：MLPDeep 和 MLPVeryDeep
            # 根据 checkpoint 路径判断
            if "MLPDeep" in str(checkpoint_path):
                from experiments.mlp_deep.mlp_deep import MLPDeep
                model = MLPDeep()
            else:
                from experiments.mlp_deep.mlp_deep import MLPVeryDeep
                model = MLPVeryDeep()
            model.load_state_dict(checkpoint["model_state_dict"])
            predictor = predictor_class(model, device=device)
            
        elif model_name == "gan":
            from experiments.gan.gan_model import Generator
            model = Generator(input_dim=8, noise_dim=16, output_dim=3)
            model.load_state_dict(checkpoint["generator_state_dict"])
            predictor = predictor_class(model, device=device)
            
        else:
            print(f"未知的模型类型: {model_name}")
            return None
        
        print(f"✓ 模型加载成功: {model_name}")
        
        models_cache[model_name] = predictor
        return predictor
        
    except Exception as e:
        print(f"加载模型失败 {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.get("/")
async def root():
    """返回主页"""
    html_file = SERVER_DIR / "ring_viewer.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "请创建 data/server/ring_viewer.html"}


@app.get("/data/use/full.json")
async def get_full_json():
    """返回完整数据集"""
    if not FULL_JSON.exists():
        raise HTTPException(status_code=404, detail="数据集文件不存在")
    return FileResponse(FULL_JSON)


@app.get("/api/dataset/info")
async def get_dataset_info():
    """获取数据集信息"""
    data = load_dataset()
    
    # 统计地图分布
    map_counts = {}
    for item in data:
        map_name = item.get("map", "unknown")
        map_counts[map_name] = map_counts.get(map_name, 0) + 1
    
    return {
        "total": len(data),
        "maps": map_counts
    }


@app.get("/api/dataset/item/{index}")
async def get_dataset_item(index: int):
    """
    获取指定索引的数据
    
    Args:
        index: 数据索引（0-based）
    """
    data = load_dataset()
    
    if index < 0 or index >= len(data):
        raise HTTPException(status_code=404, detail="索引超出范围")
    
    return data[index]


@app.get("/api/models/list")
async def list_models():
    """列出可用的模型"""
    config = load_models_config()
    models = []
    
    # 检查各个实验目录
    experiments_dir = PROJECT_ROOT / "experiments"
    
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        model_id = exp_dir.name
        
        # 检查是否有 predictor.py 和 checkpoints
        predictor_file = exp_dir / "predictor.py"
        checkpoints_dir = exp_dir / "checkpoints"
        
        if predictor_file.exists() and checkpoints_dir.exists():
            # 检查是否有模型文件
            has_model = False
            for checkpoint_file in checkpoints_dir.rglob("*.pth"):
                has_model = True
                break
            
            if has_model:
                # 检查配置中是否存在
                if model_id not in config["models"]:
                    # 第一次发现，默认启用
                    config["models"][model_id] = {
                        "enabled": True,
                        "name": model_id.replace("_", " ").title()
                    }
                    save_models_config(config)
                
                # 只返回启用的模型
                if config["models"][model_id].get("enabled", True):
                    models.append({
                        "id": model_id,
                        "name": config["models"][model_id].get("name", model_id.replace("_", " ").title()),
                        "path": str(exp_dir)
                    })
    
    return {"models": models}


@app.post("/api/models/toggle")
async def toggle_model(request: Dict[str, Any]):
    """
    切换模型启用状态
    
    Request body:
    {
        "model_id": "mlp_baseline",
        "enabled": false
    }
    """
    model_id = request.get("model_id")
    enabled = request.get("enabled")
    
    if not model_id or enabled is None:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    
    config = load_models_config()
    
    if model_id not in config["models"]:
        raise HTTPException(status_code=404, detail=f"模型未找到: {model_id}")
    
    config["models"][model_id]["enabled"] = enabled
    save_models_config(config)
    
    return {"success": True, "model_id": model_id, "enabled": enabled}


@app.post("/api/predict")
async def predict(request: Dict[str, Any]):
    """
    模型预测
    
    Request body:
    {
        "model": "mlp_baseline",
        "map": "mp_rr_district",
        "ring1": {"x": 1234, "y": 5678, "r": 4930},
        "ring2": {"x": 2345, "y": 6789, "r": 2419},  // 可选
        "mode": "ring1_to_ring2_ring3" 或 "ring1_ring2_to_ring3"
    }
    
    Response:
    {
        "ring2": {"x": ..., "y": ..., "r": ...},  // mode=ring1_to_ring2_ring3 时返回
        "ring3": {"x": ..., "y": ..., "r": ...}
    }
    """
    model_name = request.get("model")
    map_name = request.get("map")
    ring1 = request.get("ring1")
    ring2 = request.get("ring2")
    mode = request.get("mode", "ring1_to_ring2_ring3")
    
    if not model_name or not map_name or not ring1:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    
    # 加载模型
    predictor = load_model(model_name)
    if predictor is None:
        raise HTTPException(status_code=404, detail=f"模型未找到: {model_name}")
    
    try:
        if mode == "ring1_to_ring2_ring3":
            # 只提供 Ring1，预测 Ring2 和 Ring3
            pred_ring2, pred_ring3 = predictor.predict(map_name, ring1)
            return {
                "ring2": pred_ring2,
                "ring3": pred_ring3
            }
        elif mode == "ring1_ring2_to_ring3":
            # 提供 Ring1 和 Ring2，预测 Ring3
            if not ring2:
                raise HTTPException(status_code=400, detail="mode=ring1_ring2_to_ring3 需要提供 ring2")
            _, pred_ring3 = predictor.predict(map_name, ring1, ring2)
            return {
                "ring3": pred_ring3
            }
        else:
            raise HTTPException(status_code=400, detail=f"未知的 mode: {mode}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.get("/api/maps/{map_name}")
async def get_map_image(map_name: str):
    """获取地图图片"""
    if map_name not in MAP_FILES:
        raise HTTPException(status_code=404, detail=f"地图未找到: {map_name}")
    
    map_file = MAP_DIR / MAP_FILES[map_name]
    if not map_file.exists():
        raise HTTPException(status_code=404, detail=f"地图文件不存在: {map_file}")
    
    return FileResponse(map_file, media_type="image/png")


if __name__ == "__main__":
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"启动服务器...")
    print(f"访问: http://127.0.0.1:8000")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
