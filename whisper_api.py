from fastapi import FastAPI, UploadFile, File, Request
from youtube_jobs import *
import uuid, shutil, os

app = FastAPI()

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
