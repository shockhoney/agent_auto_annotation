from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tools 
import uvicorn

app = FastAPI(title="Data Preprocess API", version="1.0.0")

# 定义输入参数模型
class PathRequest(BaseModel):
    path: str = "/app/data/input"

class ProcessRequest(BaseModel):
    source_path: str
    sample_name: str

# 1. 扫描目录接口
@app.post("/scan")
async def scan_endpoint(req: PathRequest):
    try:
        structure = tools.scan_directory(req.path)
        return {"structure": structure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. 3D 标准化接口
@app.post("/process/3d")
async def process_3d_endpoint(req: ProcessRequest):
    # 调用 tools.py 中的 pcd_to_bin 逻辑
    result = tools.convert_3d_to_kitti_style(req.source_path, req.sample_name)
    return {"result": result}

# 3. 2D 标准化接口
@app.post("/process/2d")
async def process_2d_endpoint(req: ProcessRequest):
    # 假设默认后缀为 png
    result = tools.standardize_2d_image(req.source_path, req.sample_name, "png")
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
