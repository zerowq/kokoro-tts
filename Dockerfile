FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# 安装系统依赖（升级到 Python 3.11）
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    libcudnn9-cuda-12 \
    cuda-libraries-12-2 \
    libcublas-12-2 \
    ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

# 获取 pip (python 3.11 需要手动安装 pip)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 创建软连接
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 拷贝依赖定义
COPY pyproject.toml .

# 安装依赖：锁定旧版本的 kokoro-onnx 以确保在 Python 3.11 下的兼容性
# 并使用 unsafe-best-match 策略让 uv 自动匹配最佳 CUDA 12 的 onnx 运行时
RUN uv pip install --system \
    --index-strategy unsafe-best-match \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    onnxruntime-gpu \
    "kokoro-onnx<0.4.0" \
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
