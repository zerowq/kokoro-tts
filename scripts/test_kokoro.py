"""
Kokoro-82M 性能与音质验证脚本
基于官方示例: https://github.com/thewh1teagle/kokoro-onnx/blob/main/examples/save.py
"""
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.engines.kokoro_engine import KokoroEngine
from loguru import logger

def benchmark_kokoro():
    # 默认指向正确的文件 (注意: voices 是 .json 不是 .bin)
    model_path = str(ROOT_DIR / "models" / "kokoro" / "kokoro-v1.0.onnx")
    voices_path = str(ROOT_DIR / "models" / "kokoro" / "voices.json")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("💡 请从以下地址下载模型文件:")
        logger.info("   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
        return
    
    if not os.path.exists(voices_path):
        logger.error(f"❌ Voices not found: {voices_path}")
        logger.info("💡 请从以下地址下载音色文件:")
        logger.info("   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
        return
    
    try:
        engine = KokoroEngine(model_path, voices_path)
        
        test_text = "Hello, this is a test of the Kokoro eighty two M model. It is designed to be extremely lightweight and fast for real-time applications."
        output_dir = ROOT_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        output_file = str(output_dir / "test_kokoro.wav")
        
        logger.info("🚀 Starting benchmark...")
        
        # 1. 测试加载速度
        start_load = time.time()
        engine._load_model()
        logger.info(f"📊 Model Load Time: {(time.time() - start_load):.4f}s")
        
        # 2. 执行合成并保存
        start_gen = time.time()
        engine.synthesize(
            text=test_text, 
            voice="af_sarah",  # 官方推荐的音色
            lang="en-us",
            output_path=output_file
        )
        logger.info(f"📊 Total Generation Time: {(time.time() - start_gen):.4f}s")
        
        logger.info(f"✨ Validation completed! Check output at: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    benchmark_kokoro()
