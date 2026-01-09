#!/bin/bash
# Kokoro TTS 启动脚本

set -e

echo "🎤 Kokoro TTS Service"
echo "================================="

# 检查模型文件
if [ ! -f "models/kokoro/kokoro-v1.0.onnx" ] || [ ! -f "models/kokoro/voices.json" ]; then
    echo "❌ 模型文件缺失"
    echo ""
    echo "📥 选择一个选项:"
    echo "   1. uv run python scripts/download_models.py"
    echo "   2. 或手动下载:"
    echo "      https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
    exit 1
fi

echo "✅ 模型文件就绪"
echo "🚀 启动 API 服务: http://localhost:8879"
echo ""
uv run python -m src.main
