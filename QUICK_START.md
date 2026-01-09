# 快速开始

## 3 步快速启动

### 1️⃣ 下载模型文件
从 [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0) 下载：
```bash
# 创建目录
mkdir -p models/kokoro

# 下载到 models/kokoro/:
# - kokoro-v1.0.onnx
# - voices.json
```

### 2️⃣ 安装依赖
```bash
pip install -r requirements.txt
# 或使用 uv
uv sync
```

### 3️⃣ 启动服务
```bash
./start.sh
# 或直接运行
python -m src.main
```

✅ 服务运行在 `http://localhost:8080`

## 测试

```bash
# 简单测试
python scripts/test_simple.py

# 查看 API 文档
# 访问 http://localhost:8080/docs
```

## 使用示例

### Python 客户端
```python
import requests

response = requests.post('http://localhost:8080/api/tts', json={
    'text': 'Hello world',
    'voice': 'af_sarah',
    'lang': 'en-us'
})

print(response.json())
# {'success': True, 'audio_url': '/output/xxx.wav'}
```

### cURL
```bash
curl -X POST http://localhost:8080/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

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
│   └── test_simple.py       # 测试脚本
├── models/
│   └── kokoro/              # 模型文件目录 (需要下载)
├── output/                  # 输出音频目录
├── requirements.txt         # 依赖
├── README.md               # 详细文档
└── start.sh                # 启动脚本
```

## 关键特性

- ✅ **独立项目**: 避免与 CosyVoice 的依赖冲突
- ✅ **轻量级**: 仅 82M 参数
- ✅ **GPU 加速**: 支持 ONNX Runtime GPU
- ✅ **简单 API**: FastAPI 自动文档
- ✅ **流式支持**: 实时推理返回

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| 模型文件缺失 | 下载到 `models/kokoro/` |
| GPU 不工作 | 检查 ONNX_PROVIDER 环境变量 |
| 导入错误 | 确保在项目根目录运行 |
| 依赖冲突 | 使用虚拟环境隔离 |

## 更多信息

- [详细 API 文档](README.md)
- [Kokoro-ONNX 官方](https://github.com/thewh1teagle/kokoro-onnx)
