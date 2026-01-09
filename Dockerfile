FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装基础系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libcudnn9-cuda-12 \
    libsndfile1 \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 创建软连接
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 安装 uv (极致快速的包管理器)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/
ENV PATH="/uv/bin:${PATH}"

WORKDIR /app

# 先拷贝项目元数据，利用缓存层
COPY pyproject.toml .

# 安装核心依赖 (包括 GPU 版 ONNX, Kokoro 和 Torch)
RUN uv pip install --system onnxruntime-gpu kokoro-onnx torch transformers scipy fastapi uvicorn loguru soundfile numpy


# 拷贝代码
COPY . .

# 暴露服务端口
EXPOSE 8879

# 启动命令
CMD ["python", "-m", "src.main"]
