"""
STT (Speech-to-Text) Utilities Module

This module provides reusable speech-to-text functionality using faster-whisper.
Extracted from the YouTube transcriber project for reuse in other applications.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Callable, Generator


class TranscriptionResult:
    """Container for transcription results and metadata"""
    
    def __init__(self, segments, info, video_title: str, url: str, model_name: str):
        self.segments = segments
        self.info = info
        self.video_title = video_title
        self.url = url
        self.model_name = model_name
        self.detected_language = info.language
        self.language_probability = info.language_probability


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS or HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def download_youtube_audio(url: str, output_dir: Path, temp_filename: str = "temp_audio") -> Tuple[Path, Dict[str, Any]]:
    """
    Download audio from YouTube URL
    
    Args:
        url: YouTube URL
        output_dir: Directory to save the audio file
        temp_filename: Base filename for temporary audio file
        
    Returns:
        Tuple of (audio_file_path, video_info)
        
    Raises:
        ImportError: If yt-dlp is not available
        Exception: If download fails
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError(f"Missing required package. Please install: pip install yt-dlp\nError: {e}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp
    audio_path = output_dir / f"{temp_filename}.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(audio_path),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    
    # Download audio
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
        # Find the downloaded audio file
        audio_file = output_dir / f"{temp_filename}.mp3"
        if not audio_file.exists():
            raise FileNotFoundError("Audio file not found after download")
            
        return audio_file, info
        
    except Exception as e:
        raise Exception(f"Failed to download video: {e}")


def transcribe_audio_file(audio_file_path: Path, model_name: str = "base", 
                         device: str = "cpu", compute_type: str = "int8",
                         beam_size: int = 5) -> Tuple[Generator, Any]:
    """
    Transcribe an audio file using faster-whisper
    
    Args:
        audio_file_path: Path to the audio file
        model_name: Whisper model to use (tiny, base, small, medium, large)
        device: Device to use for inference (cpu, cuda)
        compute_type: Computation type (int8, int16, float16, float32)
        beam_size: Beam size for decoding
        
    Returns:
        Tuple of (segments_generator, transcription_info)
        
    Raises:
        ImportError: If faster-whisper is not available
        Exception: If transcription fails
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(f"Missing required package. Please install: pip install faster-whisper\nError: {e}")
    
    # Load faster-whisper model
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        raise Exception(f"Failed to load faster-whisper model: {e}")
    
    # Transcribe
    try:
        segments, info = model.transcribe(str(audio_file_path), beam_size=beam_size)
        return segments, info
    except Exception as e:
        raise Exception(f"Transcription failed: {e}")


def transcribe_youtube_video(url: str, output_dir: Path, model_name: str = "base",
                           include_timestamps: bool = False, 
                           cleanup_audio: bool = True,
                           progress_callback: Optional[Callable[[str], None]] = None) -> TranscriptionResult:
    """
    Complete pipeline: Download YouTube video audio and transcribe it
    
    Args:
        url: YouTube URL
        output_dir: Directory to save transcript and temporary files
        model_name: Whisper model to use (tiny, base, small, medium, large)
        include_timestamps: Whether to include timestamps in output
        cleanup_audio: Whether to delete temporary audio file after transcription
        progress_callback: Optional callback function for progress updates
        
    Returns:
        TranscriptionResult object containing segments, info, and metadata
        
    Raises:
        Exception: If any step of the pipeline fails
    """
    def log(message: str):
        if progress_callback:
            progress_callback(message)
    
    try:
        log(f"Starting transcription for: {url}")
        log(f"Using faster-whisper model: {model_name}")
        log(f"Output directory: {output_dir}")
        log("-" * 50)
        
        # Download audio
        log("Downloading audio from YouTube...")
        audio_file, video_info = download_youtube_audio(url, output_dir)
        
        video_title = video_info.get('title', 'Unknown Video')
        duration = video_info.get('duration', 0)
        
        log(f"Downloaded: {video_title}")
        if duration:
            log(f"Duration: {duration//60}:{duration%60:02d}")
        
        # Transcribe
        log("Transcribing audio...")
        segments, info = transcribe_audio_file(audio_file, model_name)
        
        log(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Create result object
        result = TranscriptionResult(segments, info, video_title, url, model_name)
        
        # Clean up audio file if requested
        if cleanup_audio:
            try:
                audio_file.unlink()
                log("Temporary audio file cleaned up")
            except Exception:
                log("Warning: Failed to clean up temporary audio file")
        
        log("✓ Transcription completed successfully!")
        return result
        
    except Exception as e:
        # Attempt cleanup on error
        if cleanup_audio:
            temp_audio = output_dir / "temp_audio.mp3"
            if temp_audio.exists():
                try:
                    temp_audio.unlink()
                except Exception:
                    pass
        raise e


def save_transcript_to_file(result: TranscriptionResult, output_file: Path, 
                          include_timestamps: bool = False) -> None:
    """
    Save transcription result to a text file
    
    Args:
        result: TranscriptionResult object
        output_file: Path where to save the transcript
        include_timestamps: Whether to include timestamps in the output
        
    Raises:
        Exception: If file writing fails
    """
    try:
        with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
            f.write(f"Transcript for: {result.video_title}\n")
            f.write(f"YouTube URL: {result.url}\n")
            f.write(f"Generated with faster-whisper model: {result.model_name}\n")
            f.write(f"Detected language: {result.detected_language} (probability: {result.language_probability:.2f})\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            if include_timestamps:
                f.write("TRANSCRIPT WITH TIMESTAMPS:\n\n")
                for segment in result.segments:
                    start_time = format_timestamp(segment.start)
                    end_time = format_timestamp(segment.end)
                    f.write(f"[{start_time} - {end_time}] {segment.text.strip()}\n")
            else:
                f.write("TRANSCRIPT:\n\n")
                for segment in result.segments:
                    f.write(segment.text.strip() + " ")
                    
    except Exception as e:
        raise Exception(f"Failed to save transcript: {e}")


def create_safe_filename(title: str, max_length: int = 50) -> str:
    """
    Create a safe filename from video title
    
    Args:
        title: Original video title
        max_length: Maximum length for the filename
        
    Returns:
        Safe filename string
    """
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_title or len(safe_title) < 3:
        safe_title = "transcript"
    
    # Limit filename length
    if len(safe_title) > max_length:
        safe_title = safe_title[:max_length-3] + "..."
        
    return safe_title


def transcribe_youtube_to_file(url: str, output_dir: Path, model_name: str = "base",
                             include_timestamps: bool = False,
                             progress_callback: Optional[Callable[[str], None]] = None) -> Path:
    """
    High-level function: Transcribe YouTube video and save to file
    
    Args:
        url: YouTube URL
        output_dir: Directory to save the transcript
        model_name: Whisper model to use
        include_timestamps: Whether to include timestamps
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to the saved transcript file
        
    Raises:
        Exception: If transcription or file saving fails
    """
    # Transcribe the video
    result = transcribe_youtube_video(url, output_dir, model_name, 
                                    include_timestamps, True, progress_callback)
    
    # Create safe filename and save
    safe_title = create_safe_filename(result.video_title)
    transcript_file = output_dir / f"{safe_title}_transcript.txt"
    
    save_transcript_to_file(result, transcript_file, include_timestamps)
    
    if progress_callback:
        progress_callback(f"✓ Transcript saved to: {transcript_file}")
    
    return transcript_file