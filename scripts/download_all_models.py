#!/usr/bin/env python3
"""
统一的模型下载脚本 - 支持 Kokoro 和 MMS

使用方法:
  python scripts/download_all_models.py              # 下载 Kokoro + MMS 马来文
  python scripts/download_all_models.py --kokoro     # 仅下载 Kokoro
  python scripts/download_all_models.py --mms-all    # Kokoro + 所有 MMS 语言
  python scripts/download_all_models.py --check      # 检查已有模型
"""
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.config import config

# MMS 支持的语言
MMS_LANGUAGES = {
    "ms": "mms-tts-zlm",      # Malay (马来文)
    "en": "mms-tts-eng",      # English
    "id": "mms-tts-ind",      # Indonesian
    "zh": "mms-tts-zho",      # Chinese
    "ja": "mms-tts-jpn",      # Japanese
    "ko": "mms-tts-kor",      # Korean
    "es": "mms-tts-spa",      # Spanish
    "fr": "mms-tts-fra",      # French
    "de": "mms-tts-deu",      # German
    "it": "mms-tts-ita",      # Italian
}

def setup_logging():
    """配置日志"""
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

def download_file(url: str, dest_path: Path, timeout: int = 300) -> bool:
    """下载单个文件"""
    try:
        logger.info(f"📥 下载: {dest_path.name}")
        
        def show_progress(block_num, block_size, total_size_):
            percent = min(100, (block_num * block_size / total_size_ * 100)) if total_size_ > 0 else 0
            bar_length = 30
            filled = int(bar_length * percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r   [{bar}] {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, reporthook=show_progress)
        print()  # 新行
        
        file_size = dest_path.stat().st_size
        if file_size == 0:
            logger.error(f"❌ 文件为空: {dest_path}")
            return False
        
        logger.info(f"✅ 完成: {dest_path.name} ({file_size / 1024 / 1024:.1f} MB)")
        return True
        
    except urllib.error.URLError as e:
        logger.error(f"❌ 网络错误: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        return False

def download_kokoro() -> bool:
    """下载 Kokoro 模型"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎤 下载 Kokoro-82M 模型")
    logger.info("=" * 60)
    
    model_dir = config.MODEL_DIR / "kokoro"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    github_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    files = {
        "kokoro-v1.0.onnx": f"{github_url}/kokoro-v1.0.onnx",
        "voices-v1.0.bin": f"{github_url}/voices-v1.0.bin",
    }
    
    all_success = True
    for filename, url in files.items():
        dest_path = model_dir / filename
        
        if dest_path.exists() and dest_path.stat().st_size > 0:
            logger.info(f"⏭️  已存在: {filename}")
            continue
        
        if not download_file(url, dest_path):
            all_success = False
    
    # 解压 voices-v1.0.bin 和生成 voices.json
    if all_success or (model_dir / "kokoro-v1.0.onnx").exists():
        try:
            import zipfile
            import json
            
            voices_bin = model_dir / "voices-v1.0.bin"
            if voices_bin.exists():
                logger.info("📦 解压 voices-v1.0.bin...")
                with zipfile.ZipFile(voices_bin, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
                logger.info("✅ 解压完成")
                
                # 生成 voices.json (包含 numpy 数据)
                logger.info("📝 生成 voices.json...")
                import numpy as np
                
                voices_dict = {}
                for npy_file in sorted(model_dir.glob('*.npy')):
                    voice_name = npy_file.stem
                    try:
                        data = np.load(npy_file)
                        voices_dict[voice_name] = data.tolist()
                    except Exception as e:
                        logger.warning(f"⚠️  读取 {voice_name} 失败: {e}")
                
                with open(model_dir / "voices.json", 'w') as f:
                    json.dump(voices_dict, f)
                
                logger.info(f"✅ 生成 voices.json ({len(voices_dict)} 个音色)")
        except Exception as e:
            logger.error(f"⚠️  解压或生成 voices.json 失败: {e}")
            # 不算失败，因为模型文件已下载
    
    if all_success:
        logger.info("✅ Kokoro 模型准备完成")
    else:
        logger.error("❌ Kokoro 模型下载失败")
    
    return all_success

def check_mms_model_exists(language_code: str) -> bool:
    """检查 MMS 模型是否存在"""
    if language_code not in MMS_LANGUAGES:
        return False
    
    model_name = MMS_LANGUAGES[language_code]
    local_path = config.MODEL_DIR / model_name
    
    return local_path.exists() and (local_path / "config.json").exists()

def download_mms(languages: list) -> bool:
    """下载 MMS 模型"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("🗣️  下载 Meta MMS-TTS 模型")
    logger.info("=" * 60)
    
    # 检查 transformers
    try:
        import transformers
    except ImportError:
        logger.error("❌ 需要安装 transformers 才能下载 MMS 模型")
        logger.info("")
        logger.info("请先安装 MMS 依赖:")
        logger.info("   make install-mms")
        logger.info("   或: uv sync --group mms")
        logger.info("   或: pip install torch transformers")
        return False
    
    from transformers import VitsModel, AutoTokenizer
    
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    all_success = True
    for lang in languages:
        if lang not in MMS_LANGUAGES:
            logger.warning(f"⚠️  不支持的语言: {lang}")
            continue
        
        model_name = MMS_LANGUAGES[lang]
        local_path = config.MODEL_DIR / model_name
        
        logger.info(f"")
        logger.info(f"📥 下载 {lang.upper()} ({model_name})")
        
        # 检查是否已存在
        if check_mms_model_exists(lang):
            logger.info(f"✅ 模型已存在")
            continue
        
        try:
            logger.info(f"🔄 从 Hugging Face 下载中...")
            model = VitsModel.from_pretrained(f"facebook/{model_name}")
            tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
            
            local_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)
            
            logger.info(f"✅ {lang.upper()} 下载完成")
        except Exception as e:
            logger.error(f"❌ {lang.upper()} 下载失败: {e}")
            all_success = False
    
    return all_success

def check_models() -> None:
    """检查已有模型"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("📂 模型检查")
    logger.info("=" * 60)
    
    # 检查 Kokoro
    logger.info("")
    logger.info("【Kokoro-82M】")
    kokoro_dir = config.MODEL_DIR / "kokoro"
    kokoro_model = kokoro_dir / "kokoro-v1.0.onnx"
    kokoro_voices = kokoro_dir / "voices-v1.0.bin"
    
    if kokoro_model.exists() and kokoro_voices.exists():
        logger.info(f"   ✅ kokoro-v1.0.onnx ({kokoro_model.stat().st_size / 1024 / 1024:.1f} MB)")
        logger.info(f"   ✅ voices-v1.0.bin ({kokoro_voices.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        logger.info(f"   ❌ 缺失 (请运行: python scripts/download_all_models.py --kokoro)")
    
    # 检查 MMS
    logger.info("")
    logger.info("【Meta MMS-TTS】")
    found = 0
    for code, model_name in sorted(MMS_LANGUAGES.items()):
        if check_mms_model_exists(code):
            local_path = config.MODEL_DIR / model_name
            size_mb = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file()) / 1024 / 1024
            logger.info(f"   ✅ {code} ({model_name}) - {size_mb:.1f} MB")
            found += 1
        else:
            logger.info(f"   ❌ {code} ({model_name}) - 缺失")
    
    logger.info("")
    logger.info(f"📊 统计: Kokoro {'✅' if kokoro_model.exists() else '❌'}, MMS {found}/10")

def main():
    """主函数"""
    setup_logging()
    
    import argparse
    parser = argparse.ArgumentParser(
        description="统一的 TTS 模型下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认: Kokoro + MMS 马来文
  python scripts/download_all_models.py
  
  # 仅下载 Kokoro
  python scripts/download_all_models.py --kokoro-only
  
  # Kokoro + 所有 MMS 语言
  python scripts/download_all_models.py --mms-all
  
  # 检查已有模型
  python scripts/download_all_models.py --check
        """
    )
    
    parser.add_argument("--kokoro-only", action="store_true", help="仅下载 Kokoro")
    parser.add_argument("--mms-only", action="store_true", help="仅下载 MMS 马来文")
    parser.add_argument("--mms-all", action="store_true", help="Kokoro + 所有 MMS 语言")
    parser.add_argument("--check", action="store_true", help="仅检查模型")
    parser.add_argument("--lang", nargs="+", help="指定 MMS 语言下载")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🎤 TTS 模型统一下载工具")
    logger.info("=" * 60)
    
    # 检查模式
    if args.check:
        check_models()
        return 0
    
    # 确定下载策略
    download_kokoro_flag = True
    mms_langs = ["ms"]  # 默认只下载马来文
    
    if args.kokoro_only:
        mms_langs = []
    elif args.mms_only:
        download_kokoro_flag = False
    elif args.mms_all:
        mms_langs = list(MMS_LANGUAGES.keys())
    elif args.lang:
        mms_langs = args.lang
    
    # 下载
    success = True
    
    if download_kokoro_flag:
        if not download_kokoro():
            success = False
    
    if mms_langs:
        if not download_mms(mms_langs):
            success = False
    
    # 最终检查
    logger.info("")
    check_models()
    
    logger.info("")
    if success:
        logger.info("✨ 所有模型下载完成!")
        logger.info("🚀 现在可以启动服务: make run")
        return 0
    else:
        logger.error("❌ 部分模型下载失败")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n❌ 下载已取消")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
