# ClipFinder MVP - Dockerfile
# Single-stage Python image for FastAPI backend

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies needed for:
# - asyncpg: libpq-dev
# - bcrypt/cryptography: build-essential
# - in-process frame indexing fallback + yt-dlp stream merging: ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy backend application code
COPY backend/ .

# Expose the application port
EXPOSE 8000

# Run uvicorn in production mode
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
