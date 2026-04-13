"""Web 版毒圈可视化工具"""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.models.mlp_deep.mlp_deep import MLPDeep
from data.models.mlp_deep.predictor import DeepMLPPredictor

# 地图配置（与 Predictor 的 MAP_TO_ONEHOT 保持一致）
MAP_CONFIG = {
    "mp_rr_district": {
        "name": "District",
        "path": "data/map/mp_rr_district.png",
        "rings": [4930, 2419, 1488]
    },
    "mp_rr_tropic": {
        "name": "Tropic Island",
        "path": "data/map/mp_rr_tropic_island_mu2.png",
        "rings": [4894, 2407, 1284]
    }
}

# 模型配置
MODEL_CONFIG = {
    "mlp_deep": {
        "name": "MLP Deep",
        "path": PROJECT_ROOT / "data/models/mlp_deep/mlp_deep.pth",
        "module": "data.models.mlp_deep.mlp_deep",
        "class": "MLPDeep"
    }
}

app = FastAPI(title="Apex Ring Viewer")

# Predictor 缓存
predictors_cache = {}

# 挂载静态文件目录（地图图片）
app.mount("/data", StaticFiles(directory=str(PROJECT_ROOT / "data")), name="data")


# 请求模型
class RingData(BaseModel):
    x: int
    y: int


class PredictRequest(BaseModel):
    model_id: str
    map_id: str
    ring1: RingData
    ring2: Optional[RingData] = None


@app.on_event("startup")
async def load_models():
    """启动时加载模型到内存"""
    print("加载模型...")
    
    for model_id, config in MODEL_CONFIG.items():
        try:
            model = MLPDeep()
            model.load_checkpoint(str(config["path"]))
            model.eval()
            
            # 创建 Predictor
            predictor = DeepMLPPredictor(model, device="cpu")
            predictors_cache[model_id] = predictor
            
            print(f"✓ 模型加载成功: {config['name']}")
        except Exception as e:
            print(f"✗ 模型加载失败 {model_id}: {e}")
    
    print(f"模型加载完成，共 {len(predictors_cache)} 个模型")


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回主页面"""
    html_path = Path(__file__).parent / "ring_viewer.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/maps")
async def get_maps():
    """获取地图配置"""
    return {
        "maps": [
            {
                "id": map_id,
                "name": config["name"],
                "path": f"/{config['path']}",
                "rings": config["rings"]
            }
            for map_id, config in MAP_CONFIG.items()
        ]
    }


@app.get("/api/models")
async def get_models():
    """获取模型配置"""
    return {
        "models": [
            {
                "id": model_id,
                "name": config["name"]
            }
            for model_id, config in MODEL_CONFIG.items()
        ]
    }


@app.post("/api/predict")
async def predict(request: PredictRequest):
    """预测接口"""
    # 检查模型是否存在
    if request.model_id not in predictors_cache:
        raise HTTPException(status_code=404, detail=f"模型不存在: {request.model_id}")
    
    # 检查地图是否存在
    if request.map_id not in MAP_CONFIG:
        raise HTTPException(status_code=404, detail=f"地图不存在: {request.map_id}")
    
    predictor = predictors_cache[request.model_id]
    map_config = MAP_CONFIG[request.map_id]
    
    # 坐标归一化
    COORD_SIZE = 16384
    
    # 准备 Ring1 数据（归一化）
    ring1_data = {
        "x": request.ring1.x / COORD_SIZE,
        "y": request.ring1.y / COORD_SIZE,
        "r": map_config["rings"][0] / COORD_SIZE
    }
    
    # 准备 Ring2 数据（如果有）
    ring2_data = None
    if request.ring2:
        ring2_data = {
            "x": request.ring2.x / COORD_SIZE,
            "y": request.ring2.y / COORD_SIZE,
            "r": map_config["rings"][1] / COORD_SIZE
        }
    
    # 调用 Predictor
    ring2_pred, ring3_pred = predictor.predict(
        map_name=request.map_id,
        ring1_data=ring1_data,
        ring2_data=ring2_data
    )
    
    # 反归一化
    if request.ring2:
        # 只预测了 Ring3
        result = {
            "mode": "ring3",
            "ring3": {
                "x": int(ring3_pred["x"] * COORD_SIZE),
                "y": int(ring3_pred["y"] * COORD_SIZE),
                "r": int(ring3_pred["r"] * COORD_SIZE)
            }
        }
    else:
        # 预测了 Ring2 和 Ring3
        result = {
            "mode": "ring2_ring3",
            "ring2": {
                "x": int(ring2_pred["x"] * COORD_SIZE),
                "y": int(ring2_pred["y"] * COORD_SIZE),
                "r": int(ring2_pred["r"] * COORD_SIZE)
            },
            "ring3": {
                "x": int(ring3_pred["x"] * COORD_SIZE),
                "y": int(ring3_pred["y"] * COORD_SIZE),
                "r": int(ring3_pred["r"] * COORD_SIZE)
            }
        }
    
    return result


if __name__ == "__main__":
    import uvicorn
    
    print("启动 HTTPS 服务器...")
    
    # 使用 Python 生成自签名证书
    from pathlib import Path
    cert_dir = Path(__file__).parent / "certs"
    cert_dir.mkdir(exist_ok=True)
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    
    # 如果证书不存在，生成一个
    if not cert_file.exists():
        print("生成自签名证书...")
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            import datetime
            
            # 生成私钥
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # 生成证书
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.datetime.utcnow()
            ).not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=365)
            ).sign(private_key, hashes.SHA256())
            
            # 保存私钥
            with open(key_file, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # 保存证书
            with open(cert_file, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            print("✓ 证书生成成功")
        except ImportError:
            print("✗ 缺少 cryptography 库")
            print("请安装: pip install cryptography")
            exit(1)
    
    print(f"证书文件: {cert_file}")
    print(f"密钥文件: {key_file}")
    print("访问地址: https://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=str(key_file),
        ssl_certfile=str(cert_file)
    )
