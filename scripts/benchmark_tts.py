#!/usr/bin/env python3
"""
Kokoro vs MMS-TTS 性能对比测试脚本 (增强版)
测量指标: 
  - RTF (Real Time Factor)
  - TTFB (Time to First Byte/Audio)
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

# 测试文本
TEST_TEXTS = {
    "en": [
        "Hello, this is a short sentence for testing.",
        "The quick brown fox jumps over the lazy dog. This is a medium length sentence to evaluate the quality of speech synthesis.",
        "Artificial intelligence is transforming the way we interact with technology. From voice assistants to autonomous vehicles, AI is becoming an integral part of our daily lives."
    ],
    "ms": [
        "Halo, ini adalah ayat pendek untuk ujian.",
        "Saya adalah asisten AI yang dirancang untuk membantu Anda dengan berbagai tugas. Saya dapat menjawab pertanyaan, memberikan informasi, dan membantu Anda menyelesaikan pekerjaan.",
        "Kecerdasan buatan sedang mengubah cara kita berinteraksi dengan teknologi. Dari asisten suara hingga kendaraan otonom, AI menjadi bagian integral dari kehidupan sehari-hari kita."
    ]
}

def get_gpu_memory_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except: pass
    return -1

def get_peak_gpu_memory_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except: pass
    return -1

def clear_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except: pass
    gc.collect()

def benchmark_kokoro(provider="auto"):
    try:
        from src.engines.kokoro_engine import KokoroEngine
        model_path = str(ROOT_DIR / "models" / "kokoro" / "kokoro-v1.0.onnx")
        voices_path = str(ROOT_DIR / "models" / "kokoro" / "voices-v1.0.bin")
        
        clear_gpu_memory()
        mem_before = get_gpu_memory_mb()
        
        start_load = time.time()
        engine = KokoroEngine(model_path, voices_path)
        engine._load_model()
        load_time = time.time() - start_load
        
        mem_after = get_gpu_memory_mb()
        
        # 预热
        start_warm = time.time()
        engine.synthesize("Warmup.")
        warmup_time = time.time() - start_warm
        
        results = {
            "model_name": f"Kokoro (Auto -> {provider.upper()})",
            "load_time": load_time,
            "warmup_time": warmup_time,
            "gpu_mem_current": mem_after - mem_before if mem_before >= 0 else 0,
            "details": [],
            "output_files": []
        }
        
        output_dir = ROOT_DIR / "output" / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(TEST_TEXTS["en"]):
            # 模拟流式分块 (针对长文本)
            chunks = [c for c in re.split(r'([.!?])', text) if c.strip()]
            
            total_start = time.time()
            ttfb = 0
            all_audio = []
            
            for j, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                chunk_audio = engine.synthesize(chunk)
                if j == 0:
                    ttfb = time.time() - total_start
                all_audio.append(chunk_audio)
            
            total_elapsed = time.time() - total_start


            combined_audio = np.concatenate(all_audio)
            duration = len(combined_audio) / 24000
            
            out_file = output_dir / f"kokoro_{i+1}.wav"
            wav.write(out_file, 24000, (combined_audio * 32767).astype(np.int16))
            
            results["details"].append({
                "char_len": len(text),
                "elapsed": total_elapsed,
                "ttfb": ttfb,
                "duration": duration,
                "rtf": total_elapsed / duration if duration > 0 else 0
            })
            results["output_files"].append(str(out_file))
            
        results["gpu_mem_peak"] = get_peak_gpu_memory_mb()
        return results
    except Exception as e:
        logger.error(f"Kokoro Fail: {e}")
        return None

def benchmark_mms(device="auto"):
    try:
        from src.engines.mms_engine import MMSEngine
        models_dir = str(ROOT_DIR / "models")
        
        clear_gpu_memory()
        mem_before = get_gpu_memory_mb()
        
        start_load = time.time()
        engine = MMSEngine(models_dir, device="cuda" if device=="auto" else device)
        engine._load_model("ms")
        load_time = time.time() - start_load
        
        mem_after = get_gpu_memory_mb()
        
        start_warm = time.time()
        engine.synthesize("Ujian", language="ms")
        warmup_time = time.time() - start_warm
        
        results = {
            "model_name": f"MMS (CUDA)",
            "load_time": load_time,
            "warmup_time": warmup_time,
            "gpu_mem_current": mem_after - mem_before if mem_before >= 0 else 0,
            "details": [],
            "output_files": []
        }
        
        output_dir = ROOT_DIR / "output" / "benchmark"
        
        for i, text in enumerate(TEST_TEXTS["ms"]):
            # 真实计时
            total_start = time.time()
            audio = engine.synthesize(text, language="ms")
            elapsed = time.time() - total_start


            
            duration = len(audio) / 16000
            out_file = output_dir / f"mms_{i+1}.wav"
            wav.write(out_file, 16000, (audio * 32767).astype(np.int16))
            
            results["details"].append({
                "char_len": len(text),
                "elapsed": elapsed,
                "ttfb": elapsed, # MMS 暂不分块，首音即总耗时
                "duration": duration,
                "rtf": elapsed / duration if duration > 0 else 0
            })
            results["output_files"].append(str(out_file))
            
        results["gpu_mem_peak"] = get_peak_gpu_memory_mb()
        return results
    except Exception as e:
        logger.error(f"MMS Fail: {e}")
        return None

def print_comparison(results_list):
    print("\n" + "=" * 90)
    print("📊 TTS 性能对比测试结果 (含首音延迟 TTFB)")
    print("=" * 90)
    
    print("\n🔄 模型加载 / 预热:")
    print(f"   {'模型':<30} {'加载(s)':<10} {'预热(s)':<10}")
    for r in results_list:
        print(f"   {r['model_name']:<30} {r['load_time']:<10.2f} {r['warmup_time']:<10.2f}")

    print("\n⏱️  合成速度对比 (详细报表):")
    print(f"   {'模型':<25} {'字数':<6} {'总耗时(s)':<10} {'TTFB(s)':<10} {'时长(s)':<8} {'RTF':<8}")
    print("   " + "-" * 85)
    for r in results_list:
        for item in r['details']:
            print(f"   {r['model_name']:<25} {item['char_len']:<6} {item['elapsed']:<10.2f} {item['ttfb']:<10.2f} {item['duration']:<8.2f} x {item['rtf']:.3f}")
    
    print("\n💾 GPU 显存:")
    for r in results_list:
        print(f"   {r['model_name']:<30} {r['gpu_mem_current']:.1f} MB / 峰值 {r['gpu_mem_peak']:.1f} MB")
    print("=" * 90 + "\n")

def main():
    results = []
    res_k = benchmark_kokoro("gpu")
    if res_k: results.append(res_k)
    
    res_m = benchmark_mms("cuda")
    if res_m: results.append(res_m)
    
    print_comparison(results)

if __name__ == "__main__":
    main()
