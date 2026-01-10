"""
Kokoro-82M TTS 引擎封装 (v1.0 ONNX 高性能版)
基于官方示例: https://github.com/thewh1teagle/kokoro-onnx/blob/main/examples/save.py
"""
import os
import time
import numpy as np
import threading
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
        self._lock = threading.RLock() # 🔒 可重入锁，防止预热时死锁
        self.sample_rate = 24000

    def _load_model(self):
        with self._lock: # 确保只有一个线程在跑初始化
            if not self._loaded:
                try:
                    # 📢 重要：espeakng_loader 必须在 phonemizer/kokoro_onnx 之前导入
                    try:
                        import espeakng_loader
                        logger.info("✅ espeakng_loader initialized")
                    except ImportError:
                        logger.warning("⚠️ espeakng_loader not found")
                    
                    from kokoro_onnx import Kokoro
                    start_time = time.time()
                    
                    if not os.path.exists(self.model_path):
                        raise FileNotFoundError(f"Model file not found: {self.model_path}")

                    # 📢 强制开启 GPU 调度
                    import onnxruntime as ort
                    available_providers = ort.get_available_providers()
                    target_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    
                    original_session = ort.InferenceSession
                    def forced_gpu_session(path_or_bytes, sess_options=None, providers=None, **kwargs):
                        actual_providers = [p for p in target_providers if p in available_providers]
                        return original_session(path_or_bytes, sess_options=sess_options, providers=actual_providers, **kwargs)
                    
                    import json
                    original_load = np.load
                    original_json_load = json.load
                    
                    # 注入补丁 (修复了 allow_pickle 重复传参的问题)
                    def safe_np_load(*args, **kwargs):
                        kwargs['allow_pickle'] = True
                        return original_load(*args, **kwargs)

                    ort.InferenceSession = forced_gpu_session
                    np.load = safe_np_load
                    json.load = lambda f, **k: original_json_load(f, **k)
                    
                    try:
                        logger.info(f"🚀 Initializing Kokoro with GPU Providers: {target_providers}")
                        self._kokoro = Kokoro(self.model_path, self.voices_path)
                    finally:
                        ort.InferenceSession = original_session
                        np.load = original_load
                        json.load = original_json_load

                    self._loaded = True
                    logger.info(f"✅ Kokoro-ONNX v1.0 ready in {time.time() - start_time:.4f}s!")
                    
                    # 📢 预热
                    try:
                        logger.info("🔥 Warming up GPU kernels...")
                        self.synthesize("warmup", voice="af_sarah")
                    except Exception as e:
                        logger.warning(f"⚠️ Warmup failed: {e}")

                except Exception as e:
                    logger.error(f"❌ Failed to load Kokoro-ONNX: {e}")
                    raise
        return self._kokoro

    def synthesize(self, text: str, voice: str = "af_sarah", lang: str = "en-us", 
                   speed: float = 1.0, output_path: Optional[str] = None) -> np.ndarray:
        """
        合成语音 (带文本清洗和并发锁)
        """
        kokoro = self._load_model()
        
        # 1. 文本深度清洗 (解决极度复杂的字符导致的崩溃)
        import re
        
        # A. 替换已知会引发行号变化的特殊字符
        text = text.replace('—', '-') 
        text = text.replace('°', ' degrees ')
        
        # B. 移除 Emoji 表情 (Unicode 范围过滤)
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        
        # C. 过滤非法字符：仅保留可打印字符，并移除 Box Drawing 等特殊符号块
        text = "".join(ch for ch in text if ch.isprintable())
        
        # D. 强制单行化，处理空白符
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            logger.warning("⚠️ 文本清洗后为空，跳过合成")
            return np.array([], dtype=np.float32)

        start_time = time.time()
        
        # 2. 线程安全推理 (phonemizer/espeak 在多线程下极不稳定)
        with self._lock:
            try:
                logger.info(f"🎤 [Kokoro-v1.0] Synthesizing: {text[:50]}...")
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

