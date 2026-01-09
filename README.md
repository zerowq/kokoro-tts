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

# 或使用 pip
pip install -r requirements.txt
```

### 3. 下载模型

从 [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0) 下载：
- `kokoro-v1.0.onnx`
- `voices.json`

放置到 `models/kokoro/` 目录：
```bash
mkdir -p models/kokoro
# 将下载的文件放到上述目录
```

## 使用

### 启动 API 服务
```bash
./start.sh
# 或
python -m src.main
```

API 文档将在 `http://localhost:8080/docs` 可用

### 快速测试
```bash
python scripts/test_simple.py
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
