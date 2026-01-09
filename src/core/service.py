"""
TTS 服务核心逻辑 (支持多引擎)

支持的引擎:
  - Kokoro-82M (ONNX, 英文优秀)
  - Meta MMS-TTS (PyTorch, 多语言)
"""
import os
import hashlib
from pathlib import Path
from typing import Optional, Generator, Dict, List
import time
import numpy as np
from loguru import logger

from ..config import config
from ..engines.kokoro_engine import KokoroEngine
try:
    from ..engines.mms_engine import MMSEngine
    HAS_MMS = True
except ImportError:
    HAS_MMS = False
    logger.warning("⚠️ MMSEngine 未可用 (需要 transformers, torch 依赖)")

class TTSService:
    """多引擎 TTS 服务"""
    
    def __init__(self):
        self._kokoro = None
        self._mms = None
        self._cache = {}
        
    @property
    def kokoro(self) -> KokoroEngine:
        """Kokoro-82M 引擎 (英文优秀)"""
        if self._kokoro is None:
            model_path = str(config.KOKORO_MODEL)
            voices_path = str(config.KOKORO_VOICES)
            self._kokoro = KokoroEngine(model_path, voices_path)
        return self._kokoro
    
    @property
    def mms(self):
        """Meta MMS-TTS 引擎 (多语言)"""
        if not HAS_MMS:
            raise RuntimeError("❌ MMS 引擎不可用，需要安装 transformers 和 torch")
        
        if self._mms is None:
            model_dir = str(config.MODEL_DIR)
            self._mms = MMSEngine(model_dir)
        return self._mms
    
    def get_available_engines(self) -> Dict[str, bool]:
        """获取可用的引擎列表"""
        return {
            "kokoro": True,  # 总是可用 (ONNX)
            "mms": HAS_MMS
        }
    
    def auto_select_engine(self, language: str) -> str:
        """
        根据语言自动选择引擎
        
        Args:
            language: 语言代码 (如 'en-us', 'ms', 'zh')
            
        Returns:
            引擎名称 ('kokoro' 或 'mms')
        """
        # 提取语言码 (en-us -> en, ms -> ms)
        lang_code = language.split('-')[0] if '-' in language else language
        
        # 优先级: 英文用 Kokoro, 其他用 MMS
        if lang_code == 'en':
            return 'kokoro'
        
        # 其他语言如果 MMS 可用则使用 MMS
        if HAS_MMS and lang_code in ['ms', 'id', 'zh', 'ja', 'ko', 'es', 'fr', 'de', 'it']:
            return 'mms'
        
        # 回退到 Kokoro
        return 'kokoro'
    
    def synthesize(
        self, 
        text: str, 
        voice: str = "af_sarah", 
        lang: str = "en-us", 
        speed: float = 1.0,
        engine: Optional[str] = None,  # 可指定引擎
        output_path: Optional[str] = None
    ) -> Dict:
        """
        合成语音 (自动或指定引擎)
        
        Args:
            text: 要合成的文本
            voice: 音色 (Kokoro 用)
            lang: 语言 (如 'en-us', 'ms')
            speed: 速度 (Kokoro 用)
            engine: 指定引擎 ('kokoro', 'mms', 或 None 自动选择)
            output_path: 输出文件路径
            
        Returns:
            包含引擎信息和路径的字典
        """
        try:
            # 自动选择引擎
            if engine is None:
                engine = self.auto_select_engine(lang)
            
            # 生成缓存键
            cache_key = hashlib.md5(f"{text}_{engine}_{voice}_{lang}_{speed}".encode()).hexdigest()
            
            if cache_key in self._cache:
                logger.info(f"✅ 缓存命中: {text[:30]}... (引擎: {engine})")
                return {
                    "engine": engine,
                    "cached": True,
                    "audio_path": self._cache[cache_key]
                }
            
            # 执行合成
            if engine == 'mms':
                # MMS 合成
                lang_code = lang.split('-')[0] if '-' in lang else lang
                self.mms.synthesize(text, language=lang_code, output_path=output_path)
            else:
                # Kokoro 合成 (默认)
                self.kokoro.synthesize(text, voice, lang, speed, output_path)
            
            if output_path:
                self._cache[cache_key] = output_path
            
            logger.info(f"✅ 合成完成 (引擎: {engine}, 语言: {lang})")
            
            return {
                "engine": engine,
                "cached": False,
                "audio_path": output_path,
                "language": lang
            }
        except Exception as e:
            logger.error(f"❌ 合成失败: {e}")
            raise
    
    def synthesize_stream(self, text: str, voice: str = "af_sarah",
                         lang: str = "en-us", speed: float = 1.0) -> Generator[bytes, None, None]:
        """流式合成语音"""
        try:
            # 自动选择引擎
            engine = self.auto_select_engine(lang)
            
            if engine == 'mms':
                # MMS 目前不支持流式分片，直接返回完整音频流
                lang_code = lang.split('-')[0] if '-' in lang else lang
                # 使用临时文件过渡
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    self.mms.synthesize(text, language=lang_code, output_path=tmp.name)
                    with open(tmp.name, "rb") as f:
                        yield f.read()
            else:
                # Kokoro 流式合成
                for chunk in self.kokoro.synthesize_stream(text, voice, lang, speed):
                    yield chunk
        except Exception as e:
            logger.error(f"❌ Stream synthesis failed: {e}")
            raise

    
    def get_health(self) -> Dict:
        """获取服务健康状态"""
        try:
            health = {
                "status": "healthy",
                "engines": self.get_available_engines()
            }
            
            # 检查 Kokoro
            try:
                _ = self.kokoro
                health["kokoro"] = "ready"
            except Exception as e:
                health["kokoro"] = f"error: {str(e)}"
            
            # 检查 MMS (如果可用)
            if HAS_MMS:
                try:
                    # 不加载，只检查可用性
                    health["mms"] = "available"
                except Exception as e:
                    health["mms"] = f"error: {str(e)}"
            else:
                health["mms"] = "not installed (requires transformers, torch)"
            
            return health
        except Exception as e:
            logger.error(f"❌ 健康检查失败: {e}")
            return {"status": "unhealthy", "error": str(e)}

_service = None

def get_service() -> TTSService:
    global _service
    if _service is None:
        _service = TTSService()
    return _service
