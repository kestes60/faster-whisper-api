from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware  # Add CORS support
from youtube_jobs import *
from stt_utils import transcribe_audio_file  # Import the transcription function
from pathlib import Path
import uuid, shutil, os

app = FastAPI()

# Add CORS so your web app can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NEW ENDPOINT: Upload and transcribe audio file
@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and get back the transcription.
    Accepts: mp3, wav, webm, m4a, ogg, etc.
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix or ".webm"
        temp_file = temp_dir / f"{file_id}{file_ext}"
        
        # Write uploaded file to disk
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Transcribe using faster-whisper
        segments, info = transcribe_audio_file(
            temp_file,
            model_name="base",  # Use "tiny" for faster, "small"/"medium" for better quality
            device="cpu",
            compute_type="int8"
        )
        
        # Collect all text from segments
        transcript = " ".join([segment.text.strip() for segment in segments])
        
        # Clean up temp file
        temp_file.unlink()
        
        return {
            "transcript": transcript,
            "language": info.language,
            "language_probability": info.language_probability
        }
        
    except Exception as e:
        # Clean up on error
        if temp_file and temp_file.exists():
            temp_file.unlink()
        return {"error": str(e)}, 500

@app.post("/transcribe-youtube")
async def transcribe_youtube(request: Request):
    data = await request.json()
    url = data.get("url")
    job_id = str(uuid.uuid4())
    update_job_status(job_id, "queued")
    threading.Thread(target=transcribe_youtube_video, args=(job_id, url)).start()
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def check_status(job_id: str):
    return {"job_id": job_id, "status": get_job_status(job_id)}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    status = get_job_status(job_id)
    if status != "done":
        return {"job_id": job_id, "status": status}
    path = f"transcriptions/{job_id}.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return {"job_id": job_id, "transcript": f.read()}
    return {"error": "Transcript not found"}
