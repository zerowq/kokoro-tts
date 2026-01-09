"""
Kokoro TTS FastAPI 主程序
"""
import uvicorn
import uuid
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional


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

@app.post("/api/tts/stream")
async def synthesize_stream(request: TTSRequest):
    """流式合成语音"""
    try:
        service = get_service()
        gen = service.synthesize_stream(
            text=request.text,
            voice=request.voice,
            lang=request.lang,
            speed=request.speed
        )
        return StreamingResponse(gen, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("=" * 60)
    print("🎤 Kokoro TTS Service Starting")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8879)
