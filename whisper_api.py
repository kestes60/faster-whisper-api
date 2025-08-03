"""
FastAPI server for speech-to-text transcription using faster-whisper.
Provides REST API endpoints for transcribing audio files.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from stt_utils import transcribe_audio_file


app = FastAPI(
    title="Whisper API",
    description="Speech-to-text transcription API using faster-whisper",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with the actual frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


class TranscriptionResponse(BaseModel):
    transcript: str
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None


def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key is provided and matches expected value"""
    expected_key = os.getenv("WHISPER_API_KEY")
    if not expected_key or x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Whisper API is running", "version": "1.0.0"}


@app.post("/transcribe", response_model=TranscriptionResponse, dependencies=[Depends(verify_api_key)])
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = "base"
):
    """
    Transcribe an uploaded audio file to text.
    
    Args:
        file: Audio file to transcribe (mp3, wav, m4a, flac, ogg, wma, aac)
        model: Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        JSON response with transcript and language detection info
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Check file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Validate model name
    valid_models = ["tiny", "base", "small", "medium", "large"]
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Valid models: {', '.join(valid_models)}"
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        try:
            # Read and save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            temp_path = Path(temp_file.name)
            
            # Transcribe the audio file
            try:
                segments, info = transcribe_audio_file(temp_path, model_name=model)
                
                # Extract transcript text
                transcript_text = ""
                for segment in segments:
                    transcript_text += segment.text.strip() + " "
                
                transcript_text = transcript_text.strip()
                
                return TranscriptionResponse(
                    transcript=transcript_text,
                    detected_language=info.language,
                    language_probability=info.language_probability
                )
                
            except ImportError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing required dependency: {str(e)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed: {str(e)}"
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"File processing failed: {str(e)}"
            )
        finally:
            # Clean up temporary file
            try:
                temp_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors


@app.get("/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """List available Whisper models"""
    return {
        "models": [
            {"name": "tiny", "description": "Fastest, least accurate"},
            {"name": "base", "description": "Good balance of speed and accuracy"},
            {"name": "small", "description": "Better accuracy, slower"},
            {"name": "medium", "description": "High accuracy, slower"},
            {"name": "large", "description": "Best accuracy, slowest"}
        ]
    }


if __name__ == "__main__":
    # Check if API key is configured
    if not WHISPER_API_KEY:
        print("WARNING: WHISPER_API_KEY environment variable not set. API will be unprotected!")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)