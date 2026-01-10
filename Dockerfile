FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# 1. 强制安装 Python 3.11 (通过 PPA)
RUN DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    libcudnn9-cuda-12 \
    cuda-libraries-12-2 \
    libcublas-12-2 \
    libnvinfer8 libnvinfer-plugin8 libnvonnxparsers8 \
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

# 7. 创建输出目录和模型目录
RUN mkdir -p /app/output /app/models/kokoro && \
    chmod 777 /app/output

# 8. 下载模型文件 (如果不存在)
RUN python3 << 'EOF'
import os
from pathlib import Path

model_dir = Path("/app/models/kokoro")
model_dir.mkdir(parents=True, exist_ok=True)

# 检查模型文件是否存在
model_file = model_dir / "kokoro-v1.0.onnx"
voices_file = model_dir / "voices.json"

if not model_file.exists() or not voices_file.exists():
    print("⬇️ Downloading Kokoro models...")
    try:
        # 使用 kokoro-onnx 包的下载功能
        from kokoro_onnx import download_model
        download_model(str(model_dir))
        print("✅ Models downloaded successfully")
    except Exception as e:
        print(f"⚠️ Could not auto-download models: {e}")
        print("   Models will be required at runtime")
else:
    print("✅ Models already present")
EOF

EXPOSE 8879

CMD ["python3", "-m", "src.main"]
