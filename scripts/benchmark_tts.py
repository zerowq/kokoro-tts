#!/usr/bin/env python3
"""
Kokoro vs MMS-TTS 性能对比测试脚本 (全内容展示版)
"""
import os
import sys
import time
import gc
import re
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# 医疗场景测试文本
MEDICAL_TEXT = (
    "If you develop a fever, it could indicate an infection or other illness. "
    "Given your existing conditions—type 2 diabetes, hypertension, and diabetic nephropathy—fever may "
    "require prompt attention, as infections can more easily affect blood sugar and kidney function. "
    "Monitor your temperature and watch for symptoms like chills, body aches, or cough. "
    "Stay hydrated and rest. This information is for general health education only and is not "
    "a medical diagnosis. If the fever persists or is high (above 38 degrees Celsius), contact your "
    "healthcare provider immediately."
)

TEST_TEXTS = {
    "en": [
        "Warmup sentence to stabilize CUDA kernels.", 
        "Kokoro TTS provides high-quality speech synthesis with optimized GPU acceleration today.",
        "Artificial intelligence is transforming the way we interact with technology and daily life.",
        MEDICAL_TEXT
    ],
    "ms": [
        "Ayat pemanasan untuk menstabilkan kernel CUDA.",
        "Sistem ini menyediakan sintesis pertuturan berkualiti tinggi dengan pecutan GPU hari ini.",
        "Kecerdasan buatan sedang mengubah cara kita berinteraksi dengan teknologi dalam kehidupan.",
        "Jika anda mengalami demam, hubungi pembekal penjagaan kesihatan anda dengan segera."
    ]
}

def clear_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except: pass
    gc.collect()

def benchmark_kokoro():
    try:
        from src.engines.kokoro_engine import KokoroEngine
        model_path = str(ROOT_DIR / "models" / "kokoro" / "kokoro-v1.0.onnx")
        voices_path = str(ROOT_DIR / "models" / "kokoro" / "voices-v1.0.bin")
        
        clear_gpu_memory()
        engine = KokoroEngine(model_path, voices_path)
        engine._load_model()
        
        results = {"model_name": "Kokoro-82M (ONNX)", "details": []}
        output_dir = ROOT_DIR / "output" / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(TEST_TEXTS["en"]):
            # 分句模拟流式
            sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
            total_start = time.time()
            ttfb = 0
            all_audio = []
            
            for j, s in enumerate(sentences):
                audio_chunk = engine.synthesize(s)
                if j == 0: ttfb = time.time() - total_start
                all_audio.append(audio_chunk)
            
            total_elapsed = time.time() - total_start
            combined_audio = np.concatenate(all_audio)
            duration = len(combined_audio) / 24000
            
            if i > 0: # 剔除冷启动
                results["details"].append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "char_len": len(text),
                    "total": total_elapsed,
                    "ttfb": ttfb,
                    "duration": duration,
                    "rtf": total_elapsed / duration
                })
        return results
    except Exception as e:
        logger.error(f"Kokoro Fail: {e}")
        return None

def benchmark_mms():
    try:
        from src.engines.mms_engine import MMSEngine
        models_dir = str(ROOT_DIR / "models")
        clear_gpu_memory()
        engine = MMSEngine(models_dir, device="cuda")
        engine._load_model("ms")
        
        results = {"model_name": "Meta MMS (CUDA)", "details": []}
        for i, text in enumerate(TEST_TEXTS["ms"]):
            total_start = time.time()
            audio = engine.synthesize(text, language="ms")
            elapsed = time.time() - total_start
            duration = len(audio) / 16000
            
            if i > 0:
                results["details"].append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "char_len": len(text),
                    "total": elapsed,
                    "ttfb": elapsed,
                    "duration": duration,
                    "rtf": elapsed / duration
                })
        return results
    except Exception as e:
        logger.error(f"MMS Fail: {e}")
        return None

def main():
    results = []
    res_k = benchmark_kokoro()
    if res_k: results.append(res_k)
    res_m = benchmark_mms()
    if res_m: results.append(res_m)
    
    print("\n" + "=" * 125)
    print("🚀 Kokoro TTS 生产性能评估报告 (包含详细文本预览)")
    print("=" * 125)
    header = f"   {'引擎':<15} {'总耗时':<8} {'TTFB':<8} {'时长':<8} {'RTF':<8} {'内容预览'}"
    print(header)
    print("   " + "-" * 120)
    for r in results:
        for item in r['details']:
            print(f"   {r['model_name']:<15} {item['total']:<8.2f} {item['ttfb']:<8.2f} {item['duration']:<8.2f} x {item['rtf']:.3f} {item['text']}")
    print("=" * 125 + "\n")

if __name__ == "__main__":
    main()
