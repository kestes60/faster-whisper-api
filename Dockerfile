# Dockerfile
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app files
COPY . .

# Run FastAPI app
CMD ["uvicorn", "whisper_api:app", "--host", "0.0.0.0", "--port", "8000"]
