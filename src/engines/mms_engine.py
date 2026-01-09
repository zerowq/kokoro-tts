"""
Meta MMS-TTS 引擎封装

支持马来文、印度尼西亚文等多语言
适合处理 Kokoro 不支持的语言
"""
import os
import torch
import numpy as np
import scipy.io.wavfile as wav
from typing import Optional, Dict
from transformers import VitsModel, AutoTokenizer
from pathlib import Path
from loguru import logger

class MMSEngine:
    """Meta MMS-TTS 多语言引擎"""
    
    # 支持的语言模型映射
    LANGUAGE_MODELS = {
        "en": "mms-tts-eng",      # English
        "ms": "mms-tts-zlm",      # Malay (马来文)
        "id": "mms-tts-ind",      # Indonesian
        "zh": "mms-tts-zho",      # Chinese
        "ja": "mms-tts-jpn",      # Japanese
        "ko": "mms-tts-kor",      # Korean
        "es": "mms-tts-spa",      # Spanish
        "fr": "mms-tts-fra",      # French
        "de": "mms-tts-deu",      # German
        "it": "mms-tts-ita",      # Italian
    }
    
    def __init__(self, model_dir: str, device: Optional[str] = None):
        """
        初始化 MMS 引擎
        
        Args:
            model_dir: 模型存储目录
            device: 计算设备 ('cpu', 'cuda', 或 None 自动选择)
        """
        self.model_dir = Path(model_dir)
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self._models: Dict[str, VitsModel] = {}
        self._tokenizers: Dict[str, AutoTokenizer] = {}
        
        logger.info(f"🎤 MMS Engine initialized on {self.device.upper()}")
    
    def _load_model(self, language: str):
        """
        加载指定语言的模型
        
        Args:
            language: 语言代码 (如 'ms' 马来文, 'en' 英文)
        """
        if language in self._models:
            return  # 已加载，直接返回
        
        model_name = self.LANGUAGE_MODELS.get(language)
        if not model_name:
            raise ValueError(f"❌ Unsupported language: {language}. Supported: {list(self.LANGUAGE_MODELS.keys())}")
        
        try:
            # 优先从本地 model_dir 加载
            local_model_path = self.model_dir / model_name
            
            if local_model_path.exists():
                logger.info(f"📥 Loading local MMS-TTS from {local_model_path}...")
                self._models[language] = VitsModel.from_pretrained(
                    local_model_path, 
                    local_files_only=True
                ).to(self.device)
                self._tokenizers[language] = AutoTokenizer.from_pretrained(
                    local_model_path,
                    local_files_only=True
                )
            else:
                logger.warning(f"⚠️ Local model not found at {local_model_path}")
                logger.info(f"📥 Downloading MMS-TTS from Hugging Face (facebook/{model_name})...")
                self._models[language] = VitsModel.from_pretrained(
                    f"facebook/{model_name}"
                ).to(self.device)
                self._tokenizers[language] = AutoTokenizer.from_pretrained(
                    f"facebook/{model_name}"
                )
            
            logger.info(f"✅ MMS-TTS ({language}) loaded successfully on {self.device.upper()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load MMS model for language {language}: {e}")
            raise
    
    def get_sample_rate(self, language: str = "ms") -> int:
        """获取指定语言的采样率"""
        self._load_model(language)
        return self._models[language].config.sampling_rate
    
    def synthesize(
        self, 
        text: str, 
        language: str = "ms",
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            language: 语言代码 (默认 'ms' 马来文)
            output_path: 可选的输出文件路径
            
        Returns:
            波形数据 (numpy array)
        """
        self._load_model(language)
        model = self._models[language]
        tokenizer = self._tokenizers[language]
        
        # 文本转 token
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        
        # 推理
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # 提取并转换波形
        waveform = output.squeeze().cpu().numpy()
        
        # 保存到文件 (如果指定)
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sample_rate = model.config.sampling_rate
            wav.write(output_path, rate=sample_rate, data=waveform)
            logger.info(f"✅ Audio saved to {output_path}")
        
        return waveform
    
    def get_supported_languages(self) -> Dict[str, str]:
        """获取支持的语言列表"""
        return self.LANGUAGE_MODELS.copy()
    
    def clear_cache(self):
        """清理模型缓存，释放内存"""
        self._models.clear()
        self._tokenizers.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("✅ Model cache cleared")
