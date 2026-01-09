"""
Kokoro TTS 服务核心逻辑
"""
import os
import hashlib
from pathlib import Path
from typing import Optional, Generator, Dict
import time
import numpy as np
from loguru import logger

from ..config import config
from ..engines.kokoro_engine import KokoroEngine

class KokoroTTSService:
    """Kokoro TTS 服务"""
    
    def __init__(self):
        self._kokoro = None
        self._cache = {}
        
    @property
    def kokoro(self) -> KokoroEngine:
        if self._kokoro is None:
            model_path = str(config.KOKORO_MODEL)
            voices_path = str(config.KOKORO_VOICES)
            self._kokoro = KokoroEngine(model_path, voices_path)
        return self._kokoro
    
    def synthesize(self, text: str, voice: str = "af_sarah", 
                   lang: str = "en-us", speed: float = 1.0,
                   output_path: Optional[str] = None) -> Dict:
        """合成语音并返回结果"""
        try:
            # 简单缓存 (基于文本+音色)
            cache_key = hashlib.md5(f"{text}_{voice}_{lang}_{speed}".encode()).hexdigest()
            
            if cache_key in self._cache:
                logger.info(f"✅ Cache hit for: {text[:30]}...")
                return {
                    "engine": "kokoro",
                    "cached": True,
                    "audio_path": self._cache[cache_key]
                }
            
            # 执行合成
            samples = self.kokoro.synthesize(text, voice, lang, speed, output_path)
            
            if output_path:
                self._cache[cache_key] = output_path
            
            return {
                "engine": "kokoro",
                "cached": False,
                "audio_path": output_path
            }
        except Exception as e:
            logger.error(f"❌ Synthesis failed: {e}")
            raise
    
    def synthesize_stream(self, text: str, voice: str = "af_sarah",
                         lang: str = "en-us", speed: float = 1.0) -> Generator[bytes, None, None]:
        """流式合成语音"""
        try:
            for chunk in self.kokoro.synthesize_stream(text, voice, lang, speed):
                yield chunk
        except Exception as e:
            logger.error(f"❌ Stream synthesis failed: {e}")
            raise
    
    def get_health(self) -> Dict:
        """获取服务健康状态"""
        try:
            # 尝试预加载模型
            _ = self.kokoro
            return {"status": "healthy", "engine": "kokoro"}
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

_service = None

def get_service() -> KokoroTTSService:
    global _service
    if _service is None:
        _service = KokoroTTSService()
    return _service
