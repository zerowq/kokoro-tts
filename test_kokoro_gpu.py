"""
Kokoro GPU 诊断脚本
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()

print("=" * 70)
print("🔍 Kokoro GPU 诊断")
print("=" * 70)

# 1. 检查 onnxruntime-gpu 是否安装
print("\n1️⃣ 检查 onnxruntime-gpu 安装:")
import importlib.util
spec = importlib.util.find_spec("onnxruntime-gpu")
print(f"   onnxruntime-gpu spec: {spec}")

# 2. 检查 onnxruntime 版本和可用 providers
print("\n2️⃣ 检查 ONNX Runtime:")
import onnxruntime as ort
print(f"   Version: {ort.__version__}")
print(f"   Available providers: {ort.get_available_providers()}")

# 3. 测试不同的 provider 设置方式
print("\n3️⃣ 测试 Kokoro 初始化:")

model_path = str(ROOT_DIR / "models" / "kokoro" / "kokoro-v1.0.onnx")
voices_path = str(ROOT_DIR / "models" / "kokoro" / "voices.json")

# 方式1: 通过环境变量
print("\n   方式1: 设置 ONNX_PROVIDER 环境变量")
os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
from kokoro_onnx import Kokoro
k1 = Kokoro(model_path, voices_path)
print(f"   Session providers: {k1.sess.get_providers()}")

# 方式2: 直接创建 InferenceSession
print("\n   方式2: 直接指定 providers")
sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(f"   Session providers: {sess.get_providers()}")

# 4. 测试实际推理时的 GPU 使用
print("\n4️⃣ 测试推理时 GPU 使用:")
import torch
if torch.cuda.is_available():
    print(f"   推理前 GPU 显存: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")

    # 使用方式1的 Kokoro 实例
    audio = k1.create("Hello world", voice="af_sarah", lang="en-us")

    print(f"   推理后 GPU 显存: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
    print(f"   生成音频长度: {len(audio)} samples")
else:
    print("   ⚠️ CUDA 不可用")

print("\n" + "=" * 70)
