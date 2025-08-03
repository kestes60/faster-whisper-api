FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg curl wget && \
    pip install --upgrade pip && \
    pip install yt-dlp

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the app
COPY . .

CMD ["uvicorn", "whisper_api:app", "--host", "0.0.0.0", "--port", "8000"]
