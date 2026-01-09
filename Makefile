.PHONY: help install install-mms sync download download-mms test benchmark run clean

help:
	@echo "🎤 Kokoro TTS - UV 项目管理"
	@echo ""
	@echo "【基础命令】"
	@echo "  make install       - 安装依赖 (仅 Kokoro)"
	@echo "  make install-mms   - 安装 MMS 多语言支持 (torch, transformers)"
	@echo "  make download      - 下载 Kokoro 模型"
	@echo "  make download-mms  - 下载 MMS 马来文模型"
	@echo ""
	@echo "【测试和性能】"
	@echo "  make test          - 运行快速测试"
	@echo "  make benchmark     - 性能对比测试 (GPU 模式)"
	@echo "  make benchmark-cpu - 性能对比测试 (CPU 模式)"
	@echo ""
	@echo "【运行和清理】"
	@echo "  make run           - 启动服务"
	@echo "  make clean         - 清理缓存和虚拟环境"
	@echo ""

install:
	@echo "📦 Installing Kokoro dependencies..."
	uv sync

install-mms:
	@echo "📦 Installing MMS dependencies (torch, transformers)..."
	uv sync --group mms

download:
	@echo "📥 Downloading Kokoro models..."
	uv run python scripts/download_models.py

download-mms:
	@echo "📥 Downloading MMS models (Malay)..."
	uv run python scripts/download_mms_models.py --lang ms

download-mms-all:
	@echo "📥 Downloading all MMS models..."
	uv run python scripts/download_mms_models.py --all

test:
	@echo "🧪 Running quick test..."
	uv run python scripts/test_simple.py

benchmark:
	@echo "📊 Performance benchmark (GPU mode)..."
	uv run python scripts/benchmark_tts.py --kokoro gpu --mms gpu

benchmark-cpu:
	@echo "📊 Performance benchmark (CPU mode)..."
	uv run python scripts/benchmark_tts.py --kokoro cpu --mms cpu

benchmark-both:
	@echo "📊 Performance benchmark (CPU vs GPU)..."
	uv run python scripts/benchmark_tts.py --kokoro both --mms gpu

run:
	@echo "🚀 Starting service on port 8879..."
	./start.sh

clean:
	@echo "🧹 Cleaning up..."
	rm -rf .venv __pycache__ *.pyc .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Done"
