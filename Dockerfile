# 参考 cosyvoice-mms 的成功实践
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 安装 cuDNN 9（这是 ONNX GPU 识别的命门）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量 (完全对齐 cosyvoice 风格)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

WORKDIR /app

# 安装 uv 提速
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 拷贝依赖定义
COPY pyproject.toml .

# 强制安装与 V100/CUDA 12.2 完美兼容的包版本
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
    "numpy<2.0.0"

# 拷贝代码
COPY . .

# 暴露端口
EXPOSE 8879

# 启动命令
CMD ["python3", "-m", "src.main"]
