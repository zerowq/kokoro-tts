.PHONY: help install sync download test run clean

help:
	@echo "🎤 Kokoro TTS - UV 项目管理"
	@echo ""
	@echo "命令列表:"
	@echo "  make install       - 安装依赖 (uv sync)"
	@echo "  make download      - 下载模型文件"
	@echo "  make test          - 运行测试"
	@echo "  make run           - 启动服务"
	@echo "  make clean         - 清理缓存和虚拟环境"
	@echo ""

install:
	@echo "📦 Installing dependencies..."
	uv sync

download:
	@echo "📥 Downloading models..."
	uv run python scripts/download_models.py

test:
	@echo "🧪 Running tests..."
	uv run python scripts/test_simple.py

run:
	@echo "🚀 Starting Kokoro TTS service..."
	./start.sh

clean:
	@echo "🧹 Cleaning up..."
	rm -rf .venv __pycache__ *.pyc .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Done"
