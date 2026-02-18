FROM python:3.12-slim

# System dependencies:
#   libglib2.0-0, libgl1  – OpenCV headless
#   fonts-dejavu-core     – label rendering in visualizer
#   ffmpeg                – H.264 re-encoding of annotated video
#   git                   – torch.hub clones the yolov7 repo on first load
#   libglib2.0-dev, libsm6, libxrender1 – OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    fonts-dejavu-core \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than the default CUDA build).
# Override with a GPU-enabled image + index URL for CUDA environments.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch the YOLOv7 source code so the worker doesn't need network access
# at inference time.  Weights are downloaded on first task execution (or
# mounted via YOLOV7_WEIGHTS_PATH) to avoid bloating the image.
RUN python -c "\
import torch; \
torch.hub.load('WongKinYiu/yolov7', 'custom', \
    path_or_model='yolov7.pt', \
    force_reload=False, trust_repo=True, verbose=False)" \
    || true  # weights download may fail in CI; source will be cached

COPY . .

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--pool=solo"]
