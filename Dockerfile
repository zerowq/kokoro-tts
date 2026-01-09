FROM nvidia/cuda:12.2.2-cudnn9-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# 安装基础系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libsndfile1 \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 创建软连接
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/
ENV PATH="/uv/bin:${PATH}"

WORKDIR /app

# 拷贝依赖定义
COPY pyproject.toml .

# 安装核心依赖 (锁定 onnxruntime-gpu 版本以确保兼容性)
RUN uv pip install --system \
    onnxruntime-gpu==1.17.1 \
    kokoro-onnx \
    torch \
    transformers \
    scipy \
    fastapi \
    uvicorn \
    loguru \
    soundfile \
    numpy<2.0.0

# 拷贝代码
COPY . .

# 暴露端口
EXPOSE 8879

# 启动
CMD ["python", "-m", "src.main"]
