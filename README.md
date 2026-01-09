# Kokoro TTS

轻量级 Kokoro-82M 文本转语音服务，基于 ONNX 推理加速。

## 特点

- **轻量级**: 仅 82M 参数，快速推理
- **高性能**: ONNX Runtime GPU 加速
- **独立项目**: 避免与其他 TTS 引擎的依赖冲突
- **简单 API**: FastAPI 提供 HTTP 接口
- **流式支持**: 支持流式和非流式合成

## 安装

### 1. 克隆项目
```bash
git clone <repository-url> kokoro-tts
cd kokoro-tts
```

### 2. 安装依赖
```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip (备用)
pip install -r requirements.txt
```

> 💡 不了解 UV？查看 [SETUP.md](SETUP.md) 了解详情

### 3. 下载模型

**自动下载** (推荐):
```bash
make download
# 或
uv run python scripts/download_models.py
```

**手动下载**:
从 [GitHub releases](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0) 下载到 `models/kokoro/`:
- `kokoro-v1.0.onnx`
- `voices.json`

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

API 文档将在 `http://localhost:8080/docs` 可用

### 快速测试
```bash
make test
# 或
uv run python scripts/test_simple.py
```

### 项目管理命令
```bash
make help              # 查看所有命令
make install           # 安装依赖
make download          # 下载模型
make run              # 启动服务
make test             # 运行测试
make clean            # 清理环境
```

## API 接口

### 1. 合成语音 (非流式)
```bash
curl -X POST http://localhost:8080/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice": "af_sarah",
    "lang": "en-us",
    "speed": 1.0
  }'
```

### 2. 流式合成
```bash
curl -X POST http://localhost:8080/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' > output.wav
```

### 3. 健康检查
```bash
curl http://localhost:8080/api/health
```

## 支持的音色

- `af_sarah` - 女性美式英语
- `am_adam` - 男性美式英语
- `bf_emma` - 女性英式英语
- `bm_george` - 男性英式英语
- 更多音色见 [kokoro-onnx 文档](https://github.com/thewh1teagle/kokoro-onnx)

## 项目结构

```
kokoro-tts/
├── src/
│   ├── main.py              # FastAPI 应用
│   ├── config.py            # 配置
│   ├── core/
│   │   └── service.py       # 业务逻辑
│   └── engines/
│       └── kokoro_engine.py # Kokoro 推理引擎
├── scripts/
│   ├── download_models.py   # 模型下载脚本
│   ├── test_simple.py       # 快速测试
│   └── test_kokoro.py       # 详细测试
├── models/
│   └── kokoro/              # 模型文件目录 (自动管理)
├── output/                  # 输出音频目录
├── pyproject.toml           # UV 项目配置
├── uv.lock                  # 依赖锁定
├── requirements.txt         # PIP 依赖列表
├── Makefile                 # 项目管理
├── .python-version          # Python 版本指定
├── README.md                # 详细文档
└── start.sh                 # 启动脚本
```

## 配置

编辑 `src/config.py` 修改：
- 模型路径
- 采样率
- 默认音色等

## 依赖说明

项目的独立性设计：
- **不依赖 PyTorch**: 避免大型深度学习框架
- **不依赖 CosyVoice**: 避免复杂的模型依赖
- **仅依赖 ONNX Runtime**: 轻量级推理库

这样可以在同一服务器上与其他 TTS 引擎并存，互不影响。

## UV 项目管理

本项目使用 [UV](https://docs.astral.sh/uv/) 作为依赖管理和虚拟环境工具。

### 核心文件
- `pyproject.toml` - 项目配置和依赖定义
- `uv.lock` - 依赖版本锁定 (自动生成)
- `.python-version` - Python 版本指定
- `Makefile` - 快捷命令集合

### 常用命令
```bash
uv sync              # 安装/更新依赖并创建虚拟环境
uv run <command>     # 在虚拟环境中运行命令
uv add <package>     # 添加新依赖
uv remove <package>  # 移除依赖
```

### 模型文件管理

模型文件在 Git 中被忽略 (`.gitignore`)，需要通过脚本下载：

```bash
# 自动下载所有模型
make download

# 或直接运行脚本
uv run python scripts/download_models.py

# 验证模型文件
ls -lh models/kokoro/
```

下载脚本会自动：
1. 从 GitHub releases 下载最新模型
2. 验证文件完整性
3. 显示下载进度

## Docker 部署

```bash
docker build -t kokoro-tts .
docker run -p 8080:8080 -v $(pwd)/models:/app/models kokoro-tts
```

## 故障排除

### 模型文件缺失
- 确保 `models/kokoro/kokoro-v1.0.onnx` 和 `models/kokoro/voices.json` 存在

### GPU 加速不工作
- 检查 CUDA 环境变量: `export ONNX_PROVIDER=CUDAExecutionProvider`
- 确保安装了 `onnxruntime-gpu`

### 导入错误
- 确保当前目录在 Python 路径中
- 使用 `python -m src.main` 启动服务

## 性能指标

- 模型加载时间: ~1-2秒
- 推理速度: 实时 (取决于文本长度和 GPU)
- 内存占用: ~200MB (GPU VRAM)

## License

MIT
