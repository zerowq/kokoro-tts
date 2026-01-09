#!/usr/bin/env python3
"""
下载 Kokoro 模型文件脚本

使用方法:
  python scripts/download_models.py
"""
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from loguru import logger

# 配置
GITHUB_RELEASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_FILES = {
    "kokoro-v1.0.onnx": f"{GITHUB_RELEASE_URL}/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{GITHUB_RELEASE_URL}/voices-v1.0.bin",
}

ROOT_DIR = Path(__file__).parent.parent.absolute()
MODEL_DIR = ROOT_DIR / "models" / "kokoro"

def ensure_model_dir():
    """确保模型目录存在"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ 模型目录: {MODEL_DIR}")

def download_file(url: str, dest_path: Path, timeout: int = 300) -> bool:
    """下载单个文件"""
    try:
        logger.info(f"📥 下载: {dest_path.name}")
        logger.info(f"   URL: {url}")
        
        # 使用流式下载以显示进度
        def download_with_progress(url, dest):
            total_size = 0
            downloaded = 0
            
            def show_progress(block_num, block_size, total_size_):
                nonlocal total_size, downloaded
                total_size = total_size_
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size * 100)) if total_size > 0 else 0
                bar_length = 30
                filled = int(bar_length * percent / 100)
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"\r   [{bar}] {percent:.1f}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, dest, reporthook=show_progress)
            print()  # 新行
        
        download_with_progress(url, dest_path)
        
        # 验证文件大小
        file_size = dest_path.stat().st_size
        if file_size == 0:
            logger.error(f"❌ 文件为空: {dest_path}")
            return False
        
        logger.info(f"✅ 完成: {dest_path.name} ({file_size / 1024 / 1024:.1f} MB)")
        return True
        
    except urllib.error.URLError as e:
        logger.error(f"❌ 网络错误: {e}")
        logger.error("   请检查网络连接或访问: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0")
        return False
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        return False

def verify_models() -> bool:
    """验证模型文件完整性"""
    logger.info("🔍 验证模型文件...")
    
    for filename in MODEL_FILES.keys():
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            logger.error(f"❌ 缺失: {filename}")
            return False
        
        size_mb = model_path.stat().st_size / 1024 / 1024
        logger.info(f"✅ {filename} ({size_mb:.1f} MB)")
    
    return True

def skip_existing_files() -> bool:
    """检查是否所有文件都已存在"""
    all_exist = all((MODEL_DIR / filename).exists() for filename in MODEL_FILES.keys())
    
    if all_exist:
        logger.info("✅ 模型文件已存在，跳过下载")
        return True
    
    return False

def main():
    """主函数"""
    logger.remove()  # 移除默认处理器
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    
    logger.info("=" * 60)
    logger.info("🎤 Kokoro 模型下载工具")
    logger.info("=" * 60)
    
    # 确保目录存在
    ensure_model_dir()
    
    # 检查文件是否已存在
    if skip_existing_files():
        if verify_models():
            logger.info("✨ 所有模型文件就绪!")
            return True
    
    logger.info("")
    logger.info("📦 开始下载模型文件...")
    logger.info("")
    
    # 下载所有文件
    all_success = True
    for filename, url in MODEL_FILES.items():
        dest_path = MODEL_DIR / filename
        
        # 如果文件已存在，跳过
        if dest_path.exists():
            logger.info(f"⏭️  已存在: {filename}")
            continue
        
        if not download_file(url, dest_path):
            all_success = False
            break
    
    logger.info("")
    
    if not all_success:
        logger.error("❌ 下载失败!")
        logger.error("💡 手动下载:")
        for filename, url in MODEL_FILES.items():
            logger.error(f"   {filename}: {url}")
        return False
    
    # 验证下载的文件
    if verify_models():
        logger.info("")
        logger.info("=" * 60)
        logger.info("✨ 模型下载完成!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("🚀 现在可以启动服务:")
        logger.info("   ./start.sh")
        return True
    
    return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n❌ 下载已取消")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
