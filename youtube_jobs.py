import os, uuid, json, threading
from faster_whisper import WhisperModel
import yt_dlp

TRANSCRIPT_DIR = "transcriptions"
STATUS_FILE = "job_status.json"
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
lock = threading.Lock()

def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)

def update_job_status(job_id, state):
    with lock:
        status = load_status()
        status[job_id] = state
        save_status(status)

def get_job_status(job_id):
    return load_status().get(job_id, "not_found")

def transcribe_youtube_video(job_id, url):
    try:
        update_job_status(job_id, "processing")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{TRANSCRIPT_DIR}/{job_id}.%(ext)s",
            "quiet": True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)

        model = WhisperModel("base", compute_type="int8")
        segments, _ = model.transcribe(file_path)
        transcript = " ".join([seg.text.strip() for seg in segments])

        with open(f"{TRANSCRIPT_DIR}/{job_id}.txt", "w") as f:
            f.write(transcript)

        update_job_status(job_id, "done")

    except Exception as e:
        update_job_status(job_id, f"error: {str(e)}")
