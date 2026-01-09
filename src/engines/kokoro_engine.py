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
                
                # 📢 强制开启 GPU 加速
                if "ONNX_PROVIDER" not in os.environ:
                    import torch
                    if torch.cuda.is_available():
                        os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
                        logger.info("🚀 GPU detected, enabling CUDAExecutionProvider for Kokoro")
                    else:
                        os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"

                logger.info(f"🔄 Initializing Kokoro-ONNX v1.0 (Provider: {os.environ.get('ONNX_PROVIDER')})...")
                
                # 🛠️ 修复 ValueError: This file contains pickled (object) data 和编码问题
                import json
                original_load = np.load
                original_json_load = json.load
                
                # 猴子补丁：强制允许 pickle，并确保 json 读取使用 utf-8
                np.load = lambda *a, **k: original_load(*a, allow_pickle=True, **k)
                json.load = lambda f, **k: original_json_load(f, **k)
                
                try:
                    # 初始化 (此时 config.py 中 KOKORO_VOICES 指向 voices.json)
                    self._kokoro = Kokoro(self.model_path, self.voices_path)
                finally:
                    # 还原补丁
                    np.load = original_load
                    json.load = original_json_load

                
                # 检查确认最终选用的 Provider
                actual_providers = self._kokoro.sess.get_providers()
                logger.info(f"📊 Actual ONNX Providers: {actual_providers}")


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
        流式合成 (将生成的音频封装为标准 WAV 字节流)
        """
        import io
        import soundfile as sf
        
        samples = self.synthesize(text, voice, lang, speed)
        
        # 将结果写入内存中的 WAV 格式
        buffer = io.BytesIO()
        sf.write(buffer, samples, self.sample_rate, format='WAV')
        buffer.seek(0)
        
        # 吐出字节
        yield buffer.read()

