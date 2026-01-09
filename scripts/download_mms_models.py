#!/usr/bin/env python3
"""
下载 Meta MMS-TTS 模型脚本

支持多语言: 马来文(ms), 英文(en), 印尼文(id) 等
模型来源: Hugging Face (facebook/mms-tts-*)

使用方法:
  python scripts/download_mms_models.py                 # 只下载马来文
  python scripts/download_mms_models.py --all           # 下载所有支持的语言
  python scripts/download_mms_models.py --lang ms en id # 下载指定语言
  python scripts/download_mms_models.py --check         # 仅检查已有模型
"""
import sys
import argparse
from pathlib import Path
from typing import List
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# MMS 支持的语言
SUPPORTED_LANGUAGES = {
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

def check_model_exists(language_code: str) -> bool:
    """检查模型是否已存在"""
    if language_code not in SUPPORTED_LANGUAGES:
        return False
    
    model_name = SUPPORTED_LANGUAGES[language_code]
    local_path = ROOT_DIR / "models" / model_name
    
    return local_path.exists() and (local_path / "config.json").exists()

def download_language_model(language_code: str) -> bool:
    """
    下载指定语言的 MMS 模型
    
    Args:
        language_code: 语言代码 (如 'ms', 'en')
        
    Returns:
        下载是否成功
    """
    if language_code not in SUPPORTED_LANGUAGES:
        logger.error(f"❌ 不支持的语言: {language_code}")
        logger.info(f"   支持的语言: {list(SUPPORTED_LANGUAGES.keys())}")
        return False
    
    model_name = SUPPORTED_LANGUAGES[language_code]
    huggingface_model = f"facebook/{model_name}"
    local_path = ROOT_DIR / "models" / model_name
    
    logger.info(f"")
    logger.info(f"📥 下载 {language_code.upper()} ({model_name})")
    logger.info(f"   来源: Hugging Face (facebook/{model_name})")
    logger.info(f"   本地: {local_path}")
    
    try:
        # 检查本地是否已存在
        if check_model_exists(language_code):
            logger.info(f"✅ 模型已存在: {local_path}")
            return True
        
        logger.info(f"🔄 下载中...")
        
        # 延迟导入 transformers (仅在需要时导入)
        try:
            from transformers import VitsModel, AutoTokenizer
        except ImportError:
            logger.error(f"❌ 需要安装 transformers 才能下载 MMS 模型")
            logger.info(f"   请运行: make install-mms")
            logger.info(f"   或: uv sync --group mms")
            return False
        
        # 下载模型
        model = VitsModel.from_pretrained(huggingface_model)
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        
        # 保存到本地
        local_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        
        logger.info(f"✅ {language_code.upper()} 下载完成")
        logger.info(f"   保存到: {local_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {language_code.upper()} 下载失败: {e}")
        return False

def main():
    """主函数"""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="下载 Meta MMS-TTS 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只下载马来文 (默认)
  python scripts/download_mms_models.py
  
  # 下载所有支持的语言
  python scripts/download_mms_models.py --all
  
  # 下载指定语言
  python scripts/download_mms_models.py --lang ms en id
  
  # 查看支持的语言
  python scripts/download_mms_models.py --list
        """
    )
    
    parser.add_argument(
        "--lang",
        nargs="+",
        help="要下载的语言代码 (空格分隔)",
        default=["ms"]  # 默认只下载马来文
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有支持的语言"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有支持的语言"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="检查已有模型，仅显示列表"
    )
    
    args = parser.parse_args()
    
    # 列出支持的语言
    if args.list:
        logger.info("📋 Meta MMS-TTS 支持的语言:")
        for code, model in sorted(SUPPORTED_LANGUAGES.items()):
            logger.info(f"   {code:4} -> {model}")
        return 0
    
    # 检查已有模型
    if args.check:
        logger.info("=" * 60)
        logger.info("📂 已有的 MMS 模型检查")
        logger.info("=" * 60)
        
        models_dir = ROOT_DIR / "models"
        models_dir.mkdir(exist_ok=True)
        
        found_count = 0
        missing_count = 0
        
        for code, model_name in sorted(SUPPORTED_LANGUAGES.items()):
            local_path = models_dir / model_name
            exists = check_model_exists(code)
            
            if exists:
                size_mb = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file()) / 1024 / 1024
                logger.info(f"   ✅ {code:4} ({model_name}) - {size_mb:.1f} MB")
                found_count += 1
            else:
                logger.info(f"   ❌ {code:4} ({model_name}) - 缺失")
                missing_count += 1
        
        logger.info("")
        logger.info(f"📊 统计: {found_count} 个已有, {missing_count} 个缺失")
        
        if missing_count > 0:
            logger.info("")
            logger.info("💡 下载缺失的模型:")
            logger.info("   python scripts/download_mms_models.py --lang ms en  # 下载指定语言")
            logger.info("   python scripts/download_mms_models.py --all         # 下载全部")
        
        return 0
    
    # 确定要下载的语言
    if args.all:
        languages_to_download = list(SUPPORTED_LANGUAGES.keys())
    else:
        languages_to_download = args.lang
    
    # 验证语言代码
    invalid_langs = [l for l in languages_to_download if l not in SUPPORTED_LANGUAGES]
    if invalid_langs:
        logger.error(f"❌ 不支持的语言: {invalid_langs}")
        logger.info(f"   支持的语言: {list(SUPPORTED_LANGUAGES.keys())}")
        return 1
    
    logger.info("=" * 60)
    logger.info("🎤 Meta MMS-TTS 模型下载器")
    logger.info("=" * 60)
    logger.info(f"📝 将下载 {len(languages_to_download)} 个语言模型:")
    for lang in languages_to_download:
        logger.info(f"   • {lang.upper()}: {SUPPORTED_LANGUAGES[lang]}")
    logger.info("")
    
    # 创建模型目录
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    logger.info(f"📂 模型目录: {models_dir}")
    logger.info("")
    
    # 下载模型
    success_count = 0
    for language in languages_to_download:
        if download_language_model(language):
            success_count += 1
    
    # 总结
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"📊 下载结果: {success_count}/{len(languages_to_download)} 成功")
    
    if success_count == len(languages_to_download):
        logger.info("✅ 所有模型下载完成!")
        logger.info("")
        logger.info("🚀 现在可以使用 MMS 引擎:")
        logger.info("   from src.engines.mms_engine import MMSEngine")
        logger.info("   engine = MMSEngine('models')")
        logger.info("   engine.synthesize('你好', language='zh')  # 中文")
        logger.info("   engine.synthesize('Halo', language='ms')  # 马来文")
        return 0
    else:
        logger.error(f"❌ {len(languages_to_download) - success_count} 个模型下载失败")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n❌ 下载已取消")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
