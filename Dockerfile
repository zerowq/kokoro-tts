FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

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
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    ffmpeg libsndfile1 git curl espeak-ng \
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
# 先降级NumPy以兼容onnxruntime-gpu
RUN pip install --no-cache-dir "numpy<2.0.0"

# 先安装 torch (CUDA 12.1版本)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装 onnxruntime-gpu (尝试CUDA 12专用版本，从Microsoft源)
RUN pip install --no-cache-dir onnxruntime-gpu==1.18.0 \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ || \
    pip install --no-cache-dir onnxruntime-gpu==1.17.0

# 验证CUDA支持（构建时就能检测）
RUN python3 << 'PYEOF'
import onnxruntime as ort
providers = ort.get_available_providers()
print('✅ Installed providers:', providers)
if 'CUDAExecutionProvider' not in providers:
    print('⚠️  WARNING: CUDAExecutionProvider not found!')
    print('⚠️  This package does not have CUDA support built-in.')
    print('⚠️  Service will run in CPU mode.')
else:
    print('✅ CUDA support detected!')
PYEOF

# 安装其他依赖
RUN uv pip install --system \
    --index-strategy unsafe-best-match \
    "transformers>=4.35.0,<4.40.0" \
    scipy \
    fastapi \
    uvicorn \
    loguru \
    soundfile \
    "numpy<2.0.0"

# 最后安装kokoro-onnx及其依赖（排除onnxruntime和numpy）
RUN pip install --no-cache-dir \
    --no-deps kokoro-onnx colorlog espeakng-loader phonemizer && \
    pip list | grep -E "kokoro|onnxruntime|numpy|phonemizer" || true

# 6. 创建输出目录
RUN mkdir -p /app/output && chmod 777 /app/output

# 7. 拷贝代码和模型（模型已存在，直接复制）
COPY . .

EXPOSE 8879

CMD ["python3", "-m", "src.main"]
