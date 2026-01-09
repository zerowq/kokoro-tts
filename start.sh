#!/bin/bash
# Kokoro TTS 启动脚本

set -e

echo "🎤 Starting Kokoro TTS Service..."
echo "================================="

# 检查模型文件
if [ ! -f "models/kokoro/kokoro-v1.0.onnx" ]; then
    echo "❌ Model not found: models/kokoro/kokoro-v1.0.onnx"
    echo "📥 Please download from:"
    echo "   https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
    exit 1
fi

if [ ! -f "models/kokoro/voices.json" ]; then
    echo "❌ Voices not found: models/kokoro/voices.json"
    echo "📥 Please download from:"
    echo "   https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
    exit 1
fi

echo "✅ Models ready"
echo "🚀 Starting API server on http://localhost:8080"
python -m src.main
