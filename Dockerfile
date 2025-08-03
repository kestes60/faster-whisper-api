FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y ffmpeg && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install yt-dlp

CMD ["uvicorn", "whisper_api:app", "--host", "0.0.0.0", "--port", "8000"]
