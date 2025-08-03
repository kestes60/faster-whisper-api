from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import PlainTextResponse
import uvicorn
import tempfile
import os
import subprocess
import torch
from faster_whisper import WhisperModel
from pydantic import BaseModel

app = FastAPI()

# Load FasterWhisper once
model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu")

# ========== Local File Transcription ==========
@app.post("/transcribe", response_class=PlainTextResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        segments, _ = model.transcribe(temp_path)
        transcript = " ".join([segment.text for segment in segments])
        os.remove(temp_path)
        return transcript

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== YouTube Transcription ==========
class YouTubeURL(BaseModel):
    url: str

@app.post("/transcribe-youtube", response_class=PlainTextResponse)
async def transcribe_youtube(data: YouTubeURL):
    try:
        url = data.url

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "audio.m4a")

            result = subprocess.run(
                ["yt-dlp", "-f", "bestaudio", "-o", temp_path, url],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            if result.returncode != 0:
                raise Exception(result.stderr.decode())

            segments, _ = model.transcribe(temp_path)
            transcript = " ".join([segment.text for segment in segments])

            return transcript

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
