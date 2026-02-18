FROM python:3.12-slim

# System dependencies for OpenCV (headless) and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    fonts-dejavu-core \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .

# Install CPU-only PyTorch first (~200 MB vs ~2 GB for full torch)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Then install everything else
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--pool=solo"]
