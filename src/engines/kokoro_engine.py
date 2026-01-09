"""
Kokoro-82M TTS 引擎封装 (v1.0 ONNX 高性能版)
基于官方示例: https://github.com/thewh1teagle/kokoro-onnx/blob/main/examples/save.py
"""
import os
import time
import numpy as np
from typing import Optional, Generator
from loguru import logger

class KokoroEngine:
    """Kokoro-82M TTS 引擎，基于 kokoro-onnx 推理库"""
    
    def __init__(self, model_path: str, voices_path: str):
        """
        Args:
            model_path: kokoro-v1.0.onnx 的路径
            voices_path: voices-v1.0.bin 的路径
        """
        self.model_path = model_path
        self.voices_path = voices_path
        self._kokoro = None
        self._loaded = False
        self.sample_rate = 24000

    def _load_model(self):
        if not self._loaded:
            try:
                from kokoro_onnx import Kokoro
                
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
                if not os.path.exists(self.voices_path):
                    raise FileNotFoundError(f"Voices file not found: {self.voices_path}")

                start_time = time.time()
                logger.info(f"🔄 Initializing Kokoro-ONNX v1.0...")
                
                # 使用官方 API 初始化
                self._kokoro = Kokoro(self.model_path, self.voices_path)
                
                self._loaded = True
                elapsed = time.time() - start_time
                logger.info(f"✅ Kokoro-ONNX v1.0 loaded in {elapsed:.4f}s!")
            except Exception as e:
                logger.error(f"❌ Failed to load Kokoro-ONNX: {e}")
                raise
        return self._kokoro

    def synthesize(self, text: str, voice: str = "af_sarah", lang: str = "en-us", 
                   speed: float = 1.0, output_path: Optional[str] = None) -> np.ndarray:
        """
        合成语音 (非流式)
        Args:
            text: 待合成文本
            voice: 音色名称，例如 'af_sarah', 'am_adam'
            lang: 语言代码，例如 'en-us', 'en-gb'
            speed: 语速 (默认 1.0)
            output_path: 可选，保存音频的路径
        Returns:
            np.ndarray: 音频采样数据
        """
        kokoro = self._load_model()
        
        logger.info(f"🎤 [Kokoro-v1.0] Synthesizing: {text[:50]}...")
        start_time = time.time()
        
        try:
            # 使用官方 create() 方法
            samples, sample_rate = kokoro.create(
                text, voice=voice, speed=speed, lang=lang
            )
            self.sample_rate = sample_rate
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️ [Kokoro-v1.0] Synthesis completed in {elapsed:.4f}s")
            
            if output_path:
                import soundfile as sf
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                sf.write(output_path, samples, sample_rate)
                logger.info(f"💾 Saved audio to {output_path}")
                
            return samples
            
        except Exception as e:
            logger.error(f"❌ Kokoro-v1.0 synthesis failed: {e}")
            raise

    def synthesize_stream(self, text: str, voice: str = "af_sarah", lang: str = "en-us",
                          speed: float = 1.0) -> Generator[bytes, None, None]:
        """
        流式合成 (将整段音频分块返回)
        注意：kokoro-onnx 目前不支持原生流式，这里模拟分块返回
        返回 float32 字节流，以适配前端 Float32Array
        """
        samples = self.synthesize(text, voice, lang, speed)
        
        # 确保是 float32 类型
        float_audio = samples.astype(np.float32)
        
        # 分块返回 (每块约 0.5 秒)
        # float32 每个采样占用 4 字节
        chunk_size = self.sample_rate // 2 
        for i in range(0, len(float_audio), chunk_size):
            yield float_audio[i:i + chunk_size].tobytes()
