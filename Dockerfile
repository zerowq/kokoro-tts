FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 设置环境变量，强制指定库搜索路径
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# 安装系统依赖和完整的 CUDA 12 数学库
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    libcudnn9-cuda-12 \
    cuda-libraries-12-2 \
    libcublas-12-2 \
    libcurand-12-2 \
    libcusolver-12-2 \
    libcusparse-12-2 \
    ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

# 创建软连接 (使用 -sf 强制覆盖已存在的链接)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python


# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 拷贝依赖定义
COPY pyproject.toml .

# 安装依赖：使用最新的 onnxruntime-gpu 以更好地支持 CUDA 12
RUN uv pip install --system \
    onnxruntime-gpu \
    kokoro-onnx \
    torch \
    transformers \
    scipy \
    fastapi \
    uvicorn \
    loguru \
    soundfile \
    "numpy<2.0.0"

# 拷贝代码
COPY . .

EXPOSE 8879

CMD ["python3", "-m", "src.main"]
