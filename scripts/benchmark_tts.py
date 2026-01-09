#!/usr/bin/env python3
"""
Kokoro vs MMS-TTS 性能对比测试脚本

测试维度:
  • 模型加载时间
  • 推理速度 (CPU vs GPU)
  • GPU 显存占用
  • 音质对比 (手动)

支持:
  • Kokoro-82M (ONNX, CPU/GPU)
  • Meta MMS-TTS (PyTorch, CPU/GPU)
"""
import os
import sys
import time
import gc
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

# 测试文本
TEST_TEXTS = {
    "en": [
        "Hello, this is a short sentence for testing.",
        "The quick brown fox jumps over the lazy dog. This is a medium length sentence to evaluate the quality of speech synthesis.",
        "Artificial intelligence is transforming the way we interact with technology. From voice assistants to autonomous vehicles, AI is becoming an integral part of our daily lives.",
    ],
    "ms": [
        "Halo, ini adalah ayat pendek untuk ujian.",  # 短句
        "Saya adalah asisten AI yang dirancang untuk membantu Anda dengan berbagai tugas. Saya dapat menjawab pertanyaan, memberikan informasi, dan membantu Anda menyelesaikan pekerjaan.",  # 中句
        "Kecerdasan buatan sedang mengubah cara kita berinteraksi dengan teknologi. Dari asisten suara hingga kendaraan otonom, AI menjadi bagian integral dari kehidupan sehari-hari kita. Teknologi ini terus berkembang dan memberikan manfaat luar biasa bagi masyarakat.",  # 长句
    ]
}

def get_gpu_memory_mb():
    """获取当前 GPU 显存使用量 (MB)"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except:
        pass
    return -1

def get_peak_gpu_memory_mb():
    """获取峰值 GPU 显存 (MB)"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except:
        pass
    return -1

def clear_gpu_memory():
    """清理 GPU 缓存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

def benchmark_kokoro(provider="auto"):
    """
    测试 Kokoro-82M
    
    Args:
        provider: ONNX 执行提供者 ("auto", "cpu", "gpu")
    """
    try:
        from src.engines.kokoro_engine import KokoroEngine
    except ImportError:
        logger.error("❌ KokoroEngine 未找到")
        return None
    
    model_path = str(ROOT_DIR / "models" / "kokoro" / "kokoro-v1.0.onnx")
    voices_path = str(ROOT_DIR / "models" / "kokoro" / "voices.json")
    
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        logger.error("❌ Kokoro 模型文件缺失")
        return None
    
    # 设置 ONNX provider
    if provider == "cpu":
        os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
        model_name = "Kokoro-82M (CPU)"
    elif provider == "gpu":
        os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
        model_name = "Kokoro-82M (GPU)"
    else:
        os.environ.pop("ONNX_PROVIDER", None)
        model_name = "Kokoro-82M (Auto)"
    
    results = {
        "model": model_name,
        "load_time": 0,
        "warmup_time": 0,
        "synthesis_times": [],
        "gpu_memory_mb": -1,
        "peak_gpu_memory_mb": -1,
    }
    
    try:
        clear_gpu_memory()
        mem_before = get_gpu_memory_mb()
        
        # 加载模型
        logger.info(f"📥 [Kokoro] 加载模型 ({provider} mode)...")
        start = time.time()
        engine = KokoroEngine(model_path, voices_path)
        engine._load_model()
        results["load_time"] = time.time() - start
        logger.info(f"✅ [Kokoro] 模型加载: {results['load_time']:.2f}s")
        
        mem_after = get_gpu_memory_mb()
        if mem_before >= 0 and mem_after >= 0:
            results["gpu_memory_mb"] = mem_after - mem_before
        
        # 预热
        logger.info("🔥 [Kokoro] 预热...")
        start = time.time()
        engine.synthesize("Warmup test.", voice="af_sarah", lang="en-us")
        results["warmup_time"] = time.time() - start
        logger.info(f"✅ [Kokoro] 预热: {results['warmup_time']:.2f}s")
        
        # 合成测试
        output_dir = ROOT_DIR / "output" / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("⏱️  [Kokoro] 开始合成测试...")
        for i, text in enumerate(TEST_TEXTS["en"]):
            output_file = str(output_dir / f"kokoro_{provider}_{i+1}.wav")
            audio = engine.synthesize(text, voice="af_sarah", lang="en-us", output_path=output_file)
            elapsed = time.time() - start
            
            # 计算音频时长
            duration = len(audio) / 24000  # Kokoro 采样率固定 24k
            
            results["synthesis_times"].append({
                "text_length": len(text),
                "time_seconds": elapsed,
                "duration": duration,
                "output_file": output_file,
            })

            logger.info(f"  ✓ Text {i+1} ({len(text)} chars): {elapsed:.2f}s")
        
        results["peak_gpu_memory_mb"] = get_peak_gpu_memory_mb()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Kokoro 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_mms(device="auto"):
    """
    测试 Meta MMS-TTS
    
    Args:
        device: 计算设备 ("auto", "cpu", "cuda")
    """
    try:
        from src.engines.mms_engine import MMSEngine
    except ImportError:
        logger.error("❌ MMSEngine 未找到")
        return None
    
    models_dir = str(ROOT_DIR / "models")
    
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = f"MMS-TTS (Auto -> {device.upper()})"
    elif device == "cpu":
        model_name = "MMS-TTS (CPU)"
    elif device == "cuda":
        model_name = "MMS-TTS (GPU)"
    else:
        device = "cpu"
        model_name = "MMS-TTS (CPU)"
    
    results = {
        "model": model_name,
        "device": device,
        "load_time": 0,
        "warmup_time": 0,
        "synthesis_times": [],
        "gpu_memory_mb": -1,
        "peak_gpu_memory_mb": -1,
    }
    
    try:
        clear_gpu_memory()
        mem_before = get_gpu_memory_mb()
        
        # 加载模型
        logger.info(f"📥 [MMS] 加载模型 (device={device})...")
        start = time.time()
        engine = MMSEngine(models_dir, device=device)
        engine._load_model("ms")  # 预加载马来文模型
        results["load_time"] = time.time() - start
        logger.info(f"✅ [MMS] 模型加载: {results['load_time']:.2f}s")
        
        mem_after = get_gpu_memory_mb()
        if mem_before >= 0 and mem_after >= 0:
            results["gpu_memory_mb"] = mem_after - mem_before
        
        # 预热
        logger.info("🔥 [MMS] 预热...")
        start = time.time()
        engine.synthesize("Ujian", language="ms")
        results["warmup_time"] = time.time() - start
        logger.info(f"✅ [MMS] 预热: {results['warmup_time']:.2f}s")
        
        # 合成测试 (马来文)
        output_dir = ROOT_DIR / "output" / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("⏱️  [MMS] 开始合成测试 (Malay/马来文)...")
        for i, text in enumerate(TEST_TEXTS["ms"]):
            output_file = str(output_dir / f"mms_{device}_{i+1}.wav")
            audio = engine.synthesize(text, language="ms", output_path=output_file)
            elapsed = time.time() - start
            
            # 计算音频时长
            sample_rate = engine.get_sample_rate("ms")
            duration = len(audio) / sample_rate
            
            results["synthesis_times"].append({
                "text_length": len(text),
                "time_seconds": elapsed,
                "duration": duration,
                "output_file": output_file,
            })

            logger.info(f"  ✓ Text {i+1} ({len(text)} chars): {elapsed:.2f}s")
        
        results["peak_gpu_memory_mb"] = get_peak_gpu_memory_mb()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ MMS 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_comparison(results_list):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("📊 TTS 性能对比测试结果")
    print("=" * 80)
    
    if not results_list:
        print("❌ 没有可用的测试结果")
        return
    
    # 1. 模型加载时间
    print("\n🔄 模型加载时间 (一次性开销):")
    for res in results_list:
        print(f"   {res['model']:<30} {res['load_time']:.2f}s")
    
    # 2. 预热时间
    print("\n🔥 模型预热时间 (首次推理):")
    for res in results_list:
        print(f"   {res['model']:<30} {res.get('warmup_time', 0):.2f}s")
    
    # 3. GPU 显存
    print("\n💾 GPU 显存占用:")
    for res in results_list:
        if res['gpu_memory_mb'] >= 0:
            print(f"   {res['model']:<30} {res['gpu_memory_mb']:.1f} MB (当前)")
        else:
            print(f"   {res['model']:<30} N/A (CPU 模式)")
        
        if res['peak_gpu_memory_mb'] >= 0:
            print(f"   {'   (峰值)':<30} {res['peak_gpu_memory_mb']:.1f} MB")
    
    # 4. 合成速度对比 (详细报表)
    print("\n⏱️  合成速度对比 (详细报表):")
    header = f"   {'模型':<25} {'文本':<6} {'耗时(s)':<8} {'时长(s)':<8} {'速度':<8} {'RTF':<8}"
    print(header)
    print("   " + "-" * len(header))
    
    for res in results_list:
        for item in res['synthesis_times']:
            text_len = item['text_length']
            time_sec = item['time_seconds']
            duration = item['duration']
            
            speed = duration / time_sec if time_sec > 0 else 0
            rtf = time_sec / duration if duration > 0 else 0
            
            print(f"   {res['model']:<25} {text_len:<6} {time_sec:<8.2f} {duration:<8.2f} {speed:<8.1f}x {rtf:<8.3f}")

    
    # 5. 音频文件位置
    print("\n🎵 生成的音频文件:")
    output_dir = ROOT_DIR / "output" / "benchmark"
    print(f"   保存位置: {output_dir}")
    print(f"   文件: ")
    if output_dir.exists():
        for wav_file in sorted(output_dir.glob("*.wav")):
            print(f"      • {wav_file.name}")
    
    print("\n📝 测试完成! 请手动对比音质差异")
    print("=" * 80)

def main():
    """主函数"""
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    
    import argparse
    parser = argparse.ArgumentParser(
        description="TTS 性能对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认: Kokoro Auto + MMS Auto
  python scripts/benchmark_tts.py
  
  # GPU 模式对比
  python scripts/benchmark_tts.py --kokoro gpu --mms gpu
  
  # 仅测试 Kokoro
  python scripts/benchmark_tts.py --kokoro both --skip-mms
  
  # CPU vs GPU 对比
  python scripts/benchmark_tts.py --kokoro both --mms gpu
        """
    )
    
    parser.add_argument(
        "--kokoro",
        choices=["auto", "cpu", "gpu", "both"],
        default="auto",
        help="Kokoro 测试模式"
    )
    parser.add_argument(
        "--mms",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="MMS 测试模式"
    )
    parser.add_argument(
        "--skip-kokoro",
        action="store_true",
        help="跳过 Kokoro 测试"
    )
    parser.add_argument(
        "--skip-mms",
        action="store_true",
        help="跳过 MMS 测试"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 TTS 性能对比测试")
    logger.info("=" * 80)
    
    results_list = []
    
    # 测试 Kokoro
    if not args.skip_kokoro:
        if args.kokoro == "both":
            logger.info("\n--- Kokoro-82M (CPU) ---")
            clear_gpu_memory()
            result = benchmark_kokoro(provider="cpu")
            if result:
                results_list.append(result)
            
            logger.info("\n--- Kokoro-82M (GPU) ---")
            clear_gpu_memory()
            result = benchmark_kokoro(provider="gpu")
            if result:
                results_list.append(result)
        else:
            logger.info(f"\n--- Kokoro-82M ({args.kokoro}) ---")
            clear_gpu_memory()
            result = benchmark_kokoro(provider=args.kokoro)
            if result:
                results_list.append(result)
    
    # 测试 MMS
    if not args.skip_mms:
        logger.info(f"\n--- Meta MMS-TTS ({args.mms}) ---")
        clear_gpu_memory()
        result = benchmark_mms(device=args.mms)
        if result:
            results_list.append(result)
    
    # 打印结果
    print_comparison(results_list)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
