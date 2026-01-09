"""
Kokoro TTS 配置
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()

class Config:
    # 模型路径
    MODEL_DIR = ROOT_DIR / "models"
    KOKORO_MODEL = MODEL_DIR / "kokoro" / "kokoro-v1.0.onnx"
    KOKORO_VOICES = MODEL_DIR / "kokoro" / "voices.json"
    
    # 输出目录
    OUTPUT_DIR = ROOT_DIR / "output"
    
    # 服务配置
    SAMPLE_RATE = 24000
    DEFAULT_VOICE = "af_sarah"
    DEFAULT_LANG = "en-us"
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(exist_ok=True)

config = Config()
