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
import struct
import numpy as np
import concurrent.futures
from loguru import logger
from scipy.signal import resample

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
        # 🚀 全局推理线程池：用于后台并行预取
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
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
                         lang: str = "en-us", speed: float = 1.0,
                         yield_header: bool = True) -> Generator[bytes, None, None]:
        """流式合成语音 (带异步预取流水线，实现极致响应)"""
        import re
        import time
        try:
            # 1. 自动选择引擎
            engine = self.auto_select_engine(lang)
            logger.info(f"📡 [STREAM] Starting pipeline (Engine: {engine}, Header: {yield_header})...")


            # 2. 增强型分段：先按句末标点切
            # 添加了对问号、叹号、省略号的全面支持
            raw_sentences = re.split(r'([。！？…!.?])', text)
            raw_chunks = []
            for i in range(0, len(raw_sentences)-1, 2):
                raw_chunks.append(raw_sentences[i] + raw_sentences[i+1])
            if len(raw_sentences) % 2 == 1 and raw_sentences[-1].strip():
                raw_chunks.append(raw_sentences[-1])
            
            # 3. 再次精细分段：如果单段还是太长 (> 120字符)，按逗号/分号切
            chunks = []
            for chunk in (raw_chunks or [text]):
                chunk = chunk.strip()
                if not chunk: continue
                
                if len(chunk) > 40:
                    sub_parts = re.split(r'([,，;；])', chunk)
                    for j in range(0, len(sub_parts)-1, 2):
                        chunks.append(sub_parts[j] + sub_parts[j+1])
                    if len(sub_parts) % 2 == 1: chunks.append(sub_parts[-1])
                else:
                    chunks.append(chunk)
            
            # 彻底清洗每一段
            chunks = [c.strip() for c in chunks if c.strip()]
            if not chunks: return

            # 4. 发送流式 WAV 头部
            if yield_header:
                wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                    b'RIFF', 0x7FFFFFFF, b'WAVE', b'fmt ', 16, 1, 1,
                    24000, 24000 * 2, 2, 16, b'data', 0x7FFFFFFF)
                yield wav_header


            # 4. 发送流式数据 (回归串行，利用 GPU 极速推理实现首包秒开)
            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                
                start_t = time.time()
                if engine == 'mms':
                    lang_code = lang.split('-')[0] if '-' in lang else lang
                    audio = self.mms.synthesize(chunk, language=lang_code)
                    source_sr = self.mms.get_sample_rate(lang_code)
                    if source_sr != 24000 and len(audio) > 0:
                        num_samples = int(len(audio) * 24000 / source_sr)
                        audio = resample(audio, num_samples)
                else:
                    audio = self.kokoro.synthesize(chunk, voice=voice, lang=lang, speed=speed)
                
                if audio is not None and len(audio) > 0:
                    duration = time.time() - start_t
                    logger.debug(f"   ↳ [STREAM] Chunk {i+1}/{len(chunks)} ready in {duration:.3f}s")
                    pcm_data = (audio * 32767).astype(np.int16)
                    yield pcm_data.tobytes()

        except Exception as e:
            logger.error(f"❌ Pipeline synthesis failed: {e}")
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
