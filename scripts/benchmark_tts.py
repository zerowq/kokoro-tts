#!/usr/bin/env python3
"""
Kokoro vs MMS-TTS 性能对比测试脚本 (生产环境评估版)
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

# 测试文本 (增加到 4 段)
TEST_TEXTS = {
    "en": [
        "Warmup sentence to stabilize CUDA kernels.", # 第1条将被剔除
        "Hello, this is a short sentence for testing.",
        "The quick brown fox jumps over the lazy dog. This is a medium length sentence to evaluate the quality of speech synthesis.",
        "Artificial intelligence is transforming the way we interact with technology. From voice assistants to autonomous vehicles, AI is becoming an integral part of our daily lives."
    ],
    "ms": [
        "Ayat pemanasan untuk menstabilkan kernel CUDA.", # 第1条将被剔除
        "Halo, ini adalah ayat pendek untuk ujian.",
        "Saya adalah asisten AI yang dirancang untuk membantu Anda dengan berbagai tugas. Saya dapat menjawab pertanyaan dan membantu Anda.",
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
        start_load = time.time()
        engine = KokoroEngine(model_path, voices_path)
        engine._load_model()
        load_time = time.time() - start_load
        
        # 预热过程 (内部预热)
        engine.synthesize("Warmup.")
        
        results = {
            "model_name": "Kokoro-82M (ONNX + CUDA)",
            "load_time": load_time,
            "warmup_time": 0, # 这里统一为 0，因为我们关注稳态
            "details": [],
            "output_files": []
        }
        
        output_dir = ROOT_DIR / "output" / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(TEST_TEXTS["en"]):
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
            
            # 只记录索引 > 0 的数据 (剔除第一个冷启动样本)
            if i > 0:
                results["details"].append({
                    "char_len": len(text),
                    "elapsed": total_elapsed,
                    "ttfb": ttfb,
                    "duration": duration,
                    "rtf": total_elapsed / duration if duration > 0 else 0
                })
            
            out_file = output_dir / f"kokoro_steady_{i}.wav"
            wav.write(out_file, 24000, (combined_audio * 32767).astype(np.int16))
            results["output_files"].append(str(out_file))
            
        results["gpu_mem_peak"] = get_peak_gpu_memory_mb()
        results["gpu_mem_current"] = get_gpu_memory_mb()
        return results
    except Exception as e:
        logger.error(f"Kokoro Fail: {e}")
        return None

def benchmark_mms(device="cuda"):
    try:
        from src.engines.mms_engine import MMSEngine
        models_dir = str(ROOT_DIR / "models")
        
        clear_gpu_memory()
        start_load = time.time()
        engine = MMSEngine(models_dir, device=device)
        engine._load_model("ms")
        load_time = time.time() - start_load
        
        results = {
            "model_name": "Meta MMS-TTS (PyTorch + CUDA)",
            "load_time": load_time,
            "warmup_time": 0,
            "details": [],
            "output_files": []
        }
        
        output_dir = ROOT_DIR / "output" / "benchmark"
        
        for i, text in enumerate(TEST_TEXTS["ms"]):
            total_start = time.time()
            audio = engine.synthesize(text, language="ms")
            elapsed = time.time() - total_start
            
            duration = len(audio) / 16000
            
            if i > 0:
                results["details"].append({
                    "char_len": len(text),
                    "elapsed": elapsed,
                    "ttfb": elapsed,
                    "duration": duration,
                    "rtf": elapsed / duration if duration > 0 else 0
                })
            
            out_file = output_dir / f"mms_steady_{i}.wav"
            wav.write(out_file, 16000, (audio * 32767).astype(np.int16))
            results["output_files"].append(str(out_file))
            
        results["gpu_mem_peak"] = get_peak_gpu_memory_mb()
        results["gpu_mem_current"] = get_gpu_memory_mb()
        return results
    except Exception as e:
        logger.error(f"MMS Fail: {e}")
        return None

def print_comparison(results_list):
    print("\n" + "=" * 95)
    print("🚀 Kokoro TTS 生产性能评估报告 (稳态数据)")
    print("=" * 95)
    print("\n[指标定义说明]:")
    print(" - TTFB (Time To First Byte): 首音延迟。指从请求开始到生成第一句音频的时间。此数值越小，用户的主观“秒开”感知越强。")
    print(" - RTF (Real Time Factor): 实时率。计算公式为 [推理时长 / 音频时长]。")
    print("   * RTF < 1: 推理速度快于语速，不会出现卡顿。")
    print("   * RTF < 0.1: 顶级性能，代表 10 秒语音仅需 1 秒合成。")
    print(" - Total (s): 生成整段完整话语所需的总时间。")

    print("\n⏱️  稳态性能数据 (已剔除初次冷启动干扰):")
    header = f"   {'模型':<30} {'字数':<6} {'Total(s)':<10} {'TTFB(s)':<10} {'音频时长(s)':<12} {'RTF':<8}"
    print(header)
    print("   " + "-" * 90)
    for r in results_list:
        for item in r['details']:
            print(f"   {r['model_name']:<30} {item['char_len']:<6} {item['elapsed']:<10.2f} {item['ttfb']:<10.2f} {item['duration']:<12.2f} x {item['rtf']:.3f}")
    
    print("\n💾 GPU 资源占用 (Tesla V100):")
    for r in results_list:
        print(f"   {r['model_name']:<30} 显存占用: {r['gpu_mem_current']:.1f} MB | 峰值: {r['gpu_mem_peak']:.1f} MB")
    print("=" * 95 + "\n")

def main():
    results = []
    res_k = benchmark_kokoro("gpu")
    if res_k: results.append(res_k)
    res_m = benchmark_mms("cuda")
    if res_m: results.append(res_m)
    print_comparison(results)

if __name__ == "__main__":
    main()
