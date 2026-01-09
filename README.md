# Kokoro TTS + MMS 多引擎服务

轻量级文本转语音服务，支持多语言合成。

## 特点

### Kokoro-82M (英文优秀)
- **轻量级**: 仅 82M 参数，快速推理
- **高性能**: ONNX Runtime GPU 加速
- **英文优秀**: 英文合成效果最佳

### Meta MMS-TTS (多语言)
- **多语言**: 支持马来文、印尼文、中文、日文等 10+ 语言
- **灵活部署**: PyTorch 推理，支持 CPU/GPU
- **语言优化**: 针对马来文等语言优化

### 整体优势
- **自动路由**: 根据语言自动选择最优引擎
- **独立项目**: 避免依赖冲突
- **简单 API**: FastAPI 自动生成文档
- **性能测试**: 集成 GPU/CPU 性能对比工具

## 安装

### 1. 克隆项目
```bash
git clone <repository-url> kokoro-tts
cd kokoro-tts
```

### 2️⃣ 安装依赖

**仅 Kokoro (轻量):**
```bash
uv sync
# 或
make install
```

**Kokoro + MMS (多语言):**
```bash
uv sync --group mms
# 或
make install-mms
```

**PIP 备用:**
```bash
pip install -r requirements.txt
# MMS 可选: pip install torch transformers
```

> 💡 不了解 UV？查看 [SETUP.md](SETUP.md) 了解详情

### 3️⃣ 下载模型

**Kokoro 模型** (必须):
```bash
make download
# 或
uv run python scripts/download_models.py
```

**MMS 模型** (可选，用于马来文等):
```bash
# 仅马来文
make download-mms

# 所有语言
make download-mms-all
# 或
uv run python scripts/download_mms_models.py --all
```

**支持的语言列表:**
```bash
uv run python scripts/download_mms_models.py --list
```

**手动下载 Kokoro**:
从 [GitHub releases](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0) 下载到 `models/kokoro/`:
- `kokoro-v1.0.onnx` (310 MB)
- `voices-v1.0.bin` (1.5 MB)

## 使用

### 启动 API 服务
```bash
# 使用脚本
./start.sh

# 或直接使用 uv
uv run python -m src.main

# 或使用 make
make run
```

API 文档将在 `http://localhost:8879/docs` 可用

### 快速测试
```bash
make test
# 或
uv run python scripts/test_simple.py
```

### 性能测试 (GPU/CPU)
```bash
# GPU 模式对比
make benchmark

# CPU 模式对比
make benchmark-cpu

# CPU vs GPU 对比
make benchmark-both

# 自定义参数
uv run python scripts/benchmark_tts.py --kokoro gpu --mms gpu
uv run python scripts/benchmark_tts.py --kokoro cpu --mms cpu
uv run python scripts/benchmark_tts.py --kokoro both --skip-mms
```

**测试输出包括:**
- 模型加载时间
- 模型预热时间
- GPU 显存占用 (峰值)
- 合成速度对比 (字符/秒)
- 生成的音频文件

### 项目管理命令
```bash
make help              # 查看所有命令
make install           # 安装 Kokoro 依赖
make install-mms       # 安装 MMS 依赖
make download          # 下载 Kokoro 模型
make download-mms      # 下载 MMS 马来文模型
make run               # 启动服务
make test              # 运行测试
make benchmark         # GPU 性能测试
make clean             # 清理环境
```

## API 接口

### 1. 合成语音 (自动引擎选择)
```bash
# 英文 (自动使用 Kokoro)
curl -X POST http://localhost:8879/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice": "af_sarah",
    "lang": "en-us",
    "speed": 1.0
  }'

# 马来文 (自动使用 MMS, 需要安装 MMS 依赖)
curl -X POST http://localhost:8879/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Halo dunia",
    "lang": "ms"
  }'
```

### 2. 指定引擎合成
```bash
# 强制使用 Kokoro
curl -X POST http://localhost:8879/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello",
    "engine": "kokoro",
    "voice": "af_sarah"
  }'

# 强制使用 MMS
curl -X POST http://localhost:8879/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Halo",
    "engine": "mms",
    "lang": "ms"
  }'
```

### 3. 流式合成
```bash
curl -X POST http://localhost:8879/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' > output.wav
```

### 4. 健康检查
```bash
curl http://localhost:8879/api/health
```

## 支持的语言和音色

### Kokoro-82M 音色 (英文)
- `af_sarah` - 女性美式英语
- `am_adam` - 男性美式英语
- `bf_emma` - 女性英式英语
- `bm_george` - 男性英式英语
- 更多见 [kokoro-onnx 文档](https://github.com/thewh1teagle/kokoro-onnx)

### Meta MMS-TTS 语言 (10+ 语言)
| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| `ms` | 马来文 | `ja` | 日文 |
| `en` | 英文 | `ko` | 韩文 |
| `id` | 印尼文 | `es` | 西班牙文 |
| `zh` | 中文 | `fr` | 法文 |
| `de` | 德文 | `it` | 意大利文 |

**查看完整列表:**
```bash
uv run python scripts/download_mms_models.py --list
```

## 引擎对比

| 特性 | Kokoro-82M | MMS-TTS |
|------|-----------|---------|
| 模型大小 | 82M | 300M+ |
| 推理库 | ONNX | PyTorch |
| 英文质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 马来文质量 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 多语言 | ❌ | ✅ (10+) |
| GPU 加速 | ✅ (ONNX) | ✅ (PyTorch) |
| 依赖体积 | 小 (~100MB) | 大 (~2GB+) |
| 推荐场景 | 英文优先 | 多语言需求 |

## 项目结构

```
kokoro-tts/
├── src/
│   ├── main.py              # FastAPI 应用 (端口 8879)
│   ├── config.py            # 配置管理
│   ├── core/
│   │   └── service.py       # 多引擎业务逻辑
│   └── engines/
│       ├── kokoro_engine.py # Kokoro 推理引擎 (ONNX)
│       └── mms_engine.py    # MMS 推理引擎 (PyTorch)
├── scripts/
│   ├── download_models.py   # Kokoro 模型下载
│   ├── download_mms_models.py # MMS 模型下载 (可选)
│   ├── test_simple.py       # 快速测试
│   ├── benchmark_tts.py     # GPU/CPU 性能测试
│   └── test_kokoro.py       # 详细测试
├── models/
│   ├── kokoro/              # Kokoro 模型 (Git 忽略)
│   ├── mms-tts-*/           # MMS 模型 (Git 忽略)
│   └── .gitkeep
├── output/                  # 输出音频目录
├── pyproject.toml           # UV 项目配置 (含 mms 依赖组)
├── uv.lock                  # 依赖锁定
├── requirements.txt         # PIP 依赖列表
├── Makefile                 # 项目管理 (含 benchmark 命令)
├── .python-version          # Python 版本 (3.11)
├── .env.example             # 环境配置示例
├── Dockerfile               # Docker 配置
├── README.md                # 详细文档
└── start.sh                 # 启动脚本 (端口 8879)
```

## 配置

编辑 `src/config.py` 修改：
- 模型路径
- 采样率
- 默认音色等

## 依赖说明

### 核心设计 - 最小化依赖冲突

**Kokoro-82M (默认安装):**
- 轻量级 ONNX 推理库
- 不依赖 PyTorch
- 体积小 (~100MB)
- 适合生产环境

**Meta MMS-TTS (可选):**
- 基于 PyTorch 推理
- 大型依赖 (~2GB+)
- 用于多语言需求
- 通过依赖组隔离

这样设计可以：
1. ✅ 独立部署 Kokoro (不需要 PyTorch)
2. ✅ 按需安装 MMS (避免不必要的依赖)
3. ✅ 在同一服务器与其他引擎共存
4. ✅ 灵活选择性能和功能

### 依赖对比

| 依赖 | Kokoro | MMS | 说明 |
|------|--------|-----|------|
| PyTorch | ❌ | ✅ | 仅 MMS 需要 |
| Transformers | ❌ | ✅ | 仅 MMS 需要 |
| ONNX Runtime | ✅ | ❌ | 仅 Kokoro 需要 |
| FastAPI | ✅ | ✅ | Web 框架 |
| soundfile | ✅ | ✅ | 音频处理 |

## UV 项目管理

本项目使用 [UV](https://docs.astral.sh/uv/) 作为依赖管理工具。

### 核心文件
- `pyproject.toml` - 项目配置和依赖定义 (包含 mms 依赖组)
- `uv.lock` - 依赖版本锁定 (必须提交到 Git)
- `.python-version` - Python 版本指定 (3.11)
- `Makefile` - 快捷命令集合

### 常用命令
```bash
# 基础
uv sync                      # 安装默认依赖 (仅 Kokoro)
uv sync --group mms          # 安装 + MMS 多语言支持
uv run python -m src.main    # 运行应用

# 依赖管理
uv add <package>             # 添加新依赖
uv remove <package>          # 移除依赖
uv add --group mms <package> # 添加到 mms 组

# 更新和清理
uv sync --upgrade            # 更新所有依赖
uv pip freeze                # 查看已安装包
```

### 模型文件管理

模型文件在 Git 中被忽略 (`.gitignore`)，需要通过脚本自动下载：

```bash
# Kokoro 模型 (必须)
make download

# MMS 模型 (可选，用于多语言)
make download-mms            # 仅马来文
make download-mms-all        # 所有 10+ 语言

# 验证模型
ls -lh models/
```

下载脚本特性：
- ✅ 自动从 GitHub/Hugging Face 下载
- ✅ 显示下载进度条
- ✅ 验证文件完整性
- ✅ 断点续传支持
- ✅ 智能缓存 (已存在则跳过)

## Docker 部署

**仅 Kokoro (轻量):**
```bash
docker build -t kokoro-tts .
docker run -p 8879:8879 -v $(pwd)/models:/app/models kokoro-tts
```

**包含 MMS (多语言):**
```bash
# 需要修改 Dockerfile 安装 MMS 依赖
# 然后使用 GPU:
docker run -p 8879:8879 \
  -v $(pwd)/models:/app/models \
  --gpus all \
  kokoro-tts
```

**环境变量:**
```bash
# GPU 加速 (Kokoro ONNX)
docker run -e ONNX_PROVIDER=CUDAExecutionProvider \
           -p 8879:8879 \
           kokoro-tts

# API 端口
docker run -e API_PORT=8879 \
           -p 8879:8879 \
           kokoro-tts
```

## 故障排除

### 模型文件缺失
- Kokoro: 确保 `models/kokoro/kokoro-v1.0.onnx` 和 `models/kokoro/voices-v1.0.bin` 存在
  ```bash
  make download
  ```
- MMS: 使用脚本下载
  ```bash
  make download-mms
  ```

### GPU 加速不工作
- **Kokoro (ONNX):**
  ```bash
  export ONNX_PROVIDER=CUDAExecutionProvider
  uv run python -m src.main
  ```
- **MMS (PyTorch):** 确保安装了 torch CUDA 版本
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

### MMS 导入错误
- 安装依赖: `make install-mms` 或 `uv sync --group mms`
- 检查 PyTorch 版本: `python -c "import torch; print(torch.__version__)"`

### 导入路径错误
- 确保从项目根目录运行: `python -m src.main`
- 或使用 make: `make run`

### GPU 显存溢出
- Kokoro: 自动使用 CPU fallback
- MMS: 尝试 CPU 模式或减少批处理

## 性能指标

### Kokoro-82M
- 模型加载: ~0.5-2秒 (ONNX 快速)
- 预热时间: ~0.1-0.5秒
- 推理速度: 50-200 字符/秒 (GPU 更快)
- GPU 显存: ~100-300MB
- CPU 内存: ~200MB

### Meta MMS-TTS
- 模型加载: ~5-15秒 (PyTorch 首次)
- 预热时间: ~1-3秒
- 推理速度: 30-100 字符/秒 (语言相关)
- GPU 显存: ~1-2GB
- CPU 内存: ~1GB+

**使用 benchmark_tts.py 进行详细测试:**
```bash
make benchmark      # GPU 对比
make benchmark-cpu  # CPU 对比
```

## License

MIT
