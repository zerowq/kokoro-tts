FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# 1. 强制安装 Python 3.11 (通过 PPA)
RUN DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common wget gnupg2 && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    libcudnn9-cuda-12 \
    cuda-libraries-12-2 \
    libcublas-12-2 \
    cuda-nvrtc-12-2 \
    cuda-nvcc-12-2 \
    ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

# 2. 强制设为默认 Python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# 3. 安装 pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 4. 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /bin/uvx

WORKDIR /app

# 5. 显式锁定依赖 (绕过版本冲突)
# 先安装 torch 和 onnxruntime-gpu
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    onnxruntime-gpu==1.19.2

# 再安装其他依赖
RUN uv pip install --system \
    --index-strategy unsafe-best-match \
    "kokoro-onnx>=0.1.6,<0.4.0" \
    transformers \
    scipy \
    fastapi \
    uvicorn \
    loguru \
    soundfile \
    "numpy<2.0.0"

# 验证 ONNX Runtime GPU 安装
RUN python3 -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers()); assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDA provider not found!'"

# 6. 创建输出目录
RUN mkdir -p /app/output && chmod 777 /app/output

# 7. 拷贝代码和模型（模型已存在，直接复制）
COPY . .

EXPOSE 8879

CMD ["python3", "-m", "src.main"]
