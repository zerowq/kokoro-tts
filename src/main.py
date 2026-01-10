"""
Kokoro TTS FastAPI 主程序
"""
import uvicorn
import uuid
import os
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.absolute()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from src.config import config
from src.core.service import get_service

app = FastAPI(
    title="Kokoro TTS API",
    description="Lightweight Kokoro-82M TTS Service",
    version="0.1.0"
)

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "af_sarah"
    lang: Optional[str] = "en-us"
    speed: Optional[float] = 1.0

class TTSResponse(BaseModel):
    success: bool
    audio_url: str

# 挂载静态文件
static_path = ROOT_DIR / "static"
output_path = config.OUTPUT_DIR

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
if output_path.exists():
    app.mount("/output", StaticFiles(directory=str(output_path)), name="output")

@app.get("/")
async def root():
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"service": "Kokoro TTS", "status": "running"}


@app.get("/api/health")
async def health():
    service = get_service()
    return service.get_health()

@app.post("/api/tts", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    """合成语音并保存为 WAV 文件"""
    try:
        filename = f"{uuid.uuid4()}.wav"
        output_path = config.OUTPUT_DIR / filename
        
        service = get_service()
        service.synthesize(
            text=request.text,
            voice=request.voice,
            lang=request.lang,
            speed=request.speed,
            output_path=str(output_path)
        )
        
        return TTSResponse(
            success=True,
            audio_url=f"/output/{filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tts/stream")
@app.post("/api/tts/stream")
async def synthesize_stream(
    text: Optional[str] = None, 
    voice: Optional[str] = "af_sarah", 
    lang: Optional[str] = "en-us", 
    speed: Optional[float] = 1.0,
    request: Optional[TTSRequest] = None
):
    """流式合成语音 API (兼容多种传参方式)"""
    try:
        # 1. 优先级: POST Body > GET Query
        if request:
            text = request.text
            voice = request.voice
            lang = request.lang
            speed = request.speed
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        service = get_service()
        gen = service.synthesize_stream(
            text=text,
            voice=voice,
            lang=lang,
            speed=speed
        )
        
        # 为了让浏览器直接播放，这里必须返回完整的二进制流 (包含 WAV 头的模拟)
        # 注意：由于 Kokoro 目前是整段生成，我们直接将生成好的结果流式吐出
        return StreamingResponse(gen, media_type="audio/wav")
    except Exception as e:
        logger.error(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎤 Kokoro TTS Service Starting")
    print("=" * 60)
    
    # 🔍 系统环境自检 (仅检查ONNX Runtime，避免torch CUDA初始化问题)
    try:
        import onnxruntime as ort
        import os
        
        print(f"🔍 [DEBUG] ONNX Runtime version: {ort.__version__}")
        print(f"🔍 [DEBUG] LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        print(f"🔍 [DEBUG] CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
        
        providers = ort.get_available_providers()
        print(f"📊 [ONNX] Available Providers: {providers}")
        
        # 尝试获取CUDA设备信息
        try:
            session_options = ort.SessionOptions()
            providers_with_options = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
            print(f"🔍 [DEBUG] Trying to create CUDA session...")
        except Exception as e:
            print(f"⚠️ [DEBUG] CUDA session creation failed: {e}")
        
        if 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers:
            print(f"🚀 [DEVICE] GPU acceleration enabled via ONNX Runtime")
        else:
            print("💡 [DEVICE] Running on CPU")
            print("⚠️ [HINT] If GPU should be available, check:")
            print("   - CUDA libraries are in LD_LIBRARY_PATH")
            print("   - onnxruntime-gpu is built with CUDA support")
            print("   - Docker container has GPU access (--gpus flag)")
    except Exception as e:
        print(f"⚠️ [ONNX] Could not get providers: {e}")

    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8879)

