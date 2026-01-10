FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
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

# 先安装 torch (CUDA 11.8版本)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 安装 onnxruntime-gpu (1.17.1 对 CUDA 11.8 非常稳定)
RUN pip install --no-cache-dir onnxruntime-gpu==1.17.1

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

# 最后安装依赖
# 1. 正常安装 phonemizer-fork 及其依赖 (它不包含 onnxruntime/numpy 冲突)
RUN pip install --no-cache-dir phonemizer-fork colorlog espeakng-loader

# 2. 仅对 kokoro-onnx 使用 --no-deps (为了保护已安装的 onnxruntime-gpu 和 numpy<2.0)
RUN pip install --no-cache-dir --no-deps kokoro-onnx

# 3. 验证所有核心组件是否到位
RUN python3 -c "import phonemizer; import joblib; import dlinfo; print('✅ Backend dependencies verified')" && \
    pip list | grep -E "kokoro|onnxruntime|numpy|phonemizer|joblib|dlinfo" || true

# 6. 创建输出目录
RUN mkdir -p /app/output && chmod 777 /app/output

# 7. 拷贝代码和模型（模型已存在，直接复制）
COPY . .

EXPOSE 8879

CMD ["python3", "-m", "src.main"]



# docker 命令
docker run -d \
  --gpus '"device=1"' \
  -p 8879:8879 \
  -v /home/work/evyd/code/speech/kokoro-tts/output:/app/output \
  --name kokoro-tts-server \
  kokoro-tts-gpu
