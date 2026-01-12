#!/usr/bin/env python3
"""
Kokoro vs MMS-TTS 性能对比测试脚本 (最终专业版)
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

def get_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024, torch.cuda.max_memory_allocated() / 1024 / 1024
    except: pass
    return 0.0, 0.0

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
        for i, text in enumerate(TEST_TEXTS["en"]):
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
            
            if i > 0:
                results["details"].append({
                    "id": f"EN-{i}",
                    "char_len": len(text),
                    "total": total_elapsed,
                    "ttfb": ttfb,
                    "duration": duration,
                    "rtf": total_elapsed / duration
                })
        
        results["mem_curr"], results["mem_peak"] = get_gpu_memory()
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
                    "id": f"MS-{i}",
                    "char_len": len(text),
                    "total": elapsed,
                    "ttfb": elapsed,
                    "duration": duration,
                    "rtf": elapsed / duration
                })
        results["mem_curr"], results["mem_peak"] = get_gpu_memory()
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
    
    print("\n" + "=" * 110)
    print("🚀 Kokoro TTS 生产性能评估报告 (Tesla V100 稳态测试)")
    print("=" * 110)
    
    print("\n[一] 指标定义说明 (Metrics Definition):")
    print(" - TTFB (Time To First Byte): 首音延迟。从请求开始到听到第一个单词的时间（针对用户感知最关键的指标）。")
    print(" - Total (s): 从请求开始到最后一段音频处理完成的总物理时间。")
    print(" - Duration (s): 生成的合成语音总时长。")
    print(" - RTF (Real Time Factor): [Total / Duration]。RTF < 1 代表比真实语速快，数值越小效率越高。")

    print("\n[二] 稳态性能对比表格 (Steady-State Results):")
    header = f"   {'引擎 (Engine)':<20} {'ID':<8} {'字数':<6} {'Total(s)':<10} {'TTFB(s)':<10} {'音频时长':<8} {'RTF':<8}"
    print(header)
    print("   " + "-" * 100)
    for r in results:
        for item in r['details']:
            print(f"   {r['model_name']:<20} {item['id']:<8} {item['char_len']:<6} {item['total']:<10.2f} {item['ttfb']:<10.2f} {item['duration']:<8.2f} x {item['rtf']:.3f}")
    
    print("\n[三] GPU 显存资源占用 (Memory Usage):")
    for r in results:
        print(f"   {r['model_name']:<20} 显存占用: {r['mem_curr']:.1f} MB | 峰值: {r['mem_peak']:.1f} MB")

    print("\n[四] 测试文本内容附录 (Test Content Appendix):")
    print(" - EN-1: " + TEST_TEXTS['en'][1])
    print(" - EN-2: " + TEST_TEXTS['en'][2])
    print(" - EN-3: (Medical Task)\n   " + TEST_TEXTS['en'][3])
    print("\n - MS-1: " + TEST_TEXTS['ms'][1])
    print(" - MS-2: " + TEST_TEXTS['ms'][2])
    print(" - MS-3: " + TEST_TEXTS['ms'][3])
    print("\n" + "=" * 110 + "\n")

if __name__ == "__main__":
    main()
