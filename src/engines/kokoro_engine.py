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
        with self._lock:  # 确保只有一个线程在跑初始化
            if not self._loaded:
                try:
                    # 📢 显式设置 espeakng 路径
                    import espeakng_loader
                    from phonemizer.backend.espeak.wrapper import EspeakWrapper
                    logger.info(f"📍 Espeak Library: {espeakng_loader.get_library_path()}")
                    EspeakWrapper.set_library(espeakng_loader.get_library_path())
                    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
                    
                    from kokoro_onnx import Kokoro
                    import onnxruntime as ort
                    start_time = time.time()
                    
                    # 🚀 极致性能 Session 配置
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    sess_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
                    
                    available_providers = ort.get_available_providers()
                    target_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    actual_providers = [p for p in target_providers if p in available_providers]

                    try:
                        # 我们手动创建 Session 以便注入配置
                        logger.info(f"🚀 Initializing Kokoro Session with: {actual_providers}")
                        self._kokoro = Kokoro(self.model_path, self.voices_path)
                        
                        # 💡 强制刷新为优化后的 Session
                        self._kokoro.sess = ort.InferenceSession(
                            self.model_path, 
                            sess_options=sess_options, 
                            providers=actual_providers
                        )
                    except Exception as e:
                        logger.error(f"❌ Failed to init Kokoro session: {e}")
                        raise

                    self._loaded = True
                    logger.info(f"✅ Kokoro-ONNX v1.0 ready in {time.time() - start_time:.4f}s!")
                    
                    # 📢 预热
                    try:
                        logger.info("🔥 Warming up GPU with complex sentence...")
                        self.synthesize("Optimization confirmed. The system is operating at maximum efficiency on the Tesla V-100 GPU.")
                    except Exception as e:
                        logger.warning(f"⚠️ Warmup failed: {e}")

                except Exception as e:
                    logger.error(f"❌ Failed to load Kokoro-ONNX: {e}")
                    raise
        return self._kokoro

    def synthesize(self, text: str, voice: str = "af_sarah", lang: str = "en-us", 
                   speed: float = 1.0, output_path: Optional[str] = None) -> np.ndarray:
        """
        合成语音 (带精细计时和优化路径)
        """
        kokoro = self._load_model()
        
        # 文本深度清洗
        import re
        text = text.replace('—', '-').replace('°', ' degrees ')
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        text = "".join(ch for ch in text if ch.isprintable())
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return np.array([], dtype=np.float32)

        start_time = time.time()
        
        try:
            # 1. 音素转换阶段 (CPU)
            pho_start = time.time()
            with self._lock:
                voice_style = voice
                if isinstance(voice, str):
                    voice_style = kokoro.get_voice_style(voice)
                # 提取音素
                phonemes = kokoro.tokenizer.phonemize(text, lang=lang)
            pho_duration = time.time() - pho_start

            # 2. 推理阶段 (GPU) - 已由 RLock 保证单引擎安全
            infer_start = time.time()
            with self._lock:
                # 使用 is_phonemes=True 跳过内部转换，trim=False 维持极速
                samples, sample_rate = kokoro.create(
                    phonemes, voice=voice_style, speed=speed, lang=lang, 
                    is_phonemes=True, trim=False
                )
            infer_duration = time.time() - infer_start
            
            self.sample_rate = sample_rate
            total_duration = time.time() - start_time
            logger.info(f"⏱️ [Kokoro] Total: {total_duration:.3f}s | Phonemes: {pho_duration:.3f}s | Infer: {infer_duration:.3f}s")
            
            if output_path:
                import soundfile as sf
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                sf.write(output_path, samples, sample_rate)
                
            return samples
            
        except Exception as e:
            logger.error(f"❌ Kokoro synthesis failed: {e}")
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

