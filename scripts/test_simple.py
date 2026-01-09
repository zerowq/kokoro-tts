#!/usr/bin/env python3
"""
简单的 Kokoro 测试脚本
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.engines.kokoro_engine import KokoroEngine
from src.config import config
from loguru import logger

def test_kokoro():
    model_path = str(config.KOKORO_MODEL)
    voices_path = str(config.KOKORO_VOICES)
    
    if not config.KOKORO_MODEL.exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("📥 Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0")
        return False
    
    if not config.KOKORO_VOICES.exists():
        logger.error(f"❌ Voices not found: {voices_path}")
        logger.info("📥 Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0")
        return False
    
    try:
        logger.info("🚀 Loading Kokoro model...")
        engine = KokoroEngine(model_path, voices_path)
        
        test_text = "Hello, this is Kokoro TTS service. It is lightweight and fast."
        output_file = config.OUTPUT_DIR / "test_kokoro.wav"
        
        logger.info(f"🎤 Synthesizing: {test_text}")
        engine.synthesize(test_text, voice="af_sarah", lang="en-us", output_path=str(output_file))
        
        logger.info(f"✅ Success! Audio saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kokoro()
    sys.exit(0 if success else 1)
