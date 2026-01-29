FROM python:3.10-slim

# 1. 设置环境变量，防止生成 pyc 文件且实时打印日志
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. 安装系统底层库（OpenCV 和 Open3D 必需）
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libgomp1 \\
    libusb-1.0-0 \\
    libx11-6 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 升级 pip 并安装 Python 依赖
# 我们直接在这里列出，方便你以后根据需要修改
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir \\
    langchain \\
    langchain-openai \\
    langgraph \\
    pandas \\
    numpy \\
    opencv-python-headless \\
    open3d \\
    pydantic \\
    fastapi \\
    uvicorn

# 4. 设置默认启动命令，运行我们的 API 服务
CMD ["python", "src/api.py"]
