FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# 1. 强制安装 Python 3.11 (通过 PPA)
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

# 2. 强制设为默认 Python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# 3. 安装 pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 4. 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 5. 显式锁定依赖 (绕过版本冲突)
# 这里我们直接指定需要的包，并让 uv 忽略冲突寻找最佳匹配
RUN uv pip install --system \
    --index-strategy unsafe-best-match \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    onnxruntime-gpu \
    "kokoro-onnx>=0.1.6,<0.4.0" \
    torch \
    transformers \
    scipy \
    fastapi \
    uvicorn \
    loguru \
    soundfile \
    "numpy<2.0.0"

# 6. 拷贝代码
COPY . .

EXPOSE 8879

CMD ["python3", "-m", "src.main"]
