# Perception-Aware Traffic Sign Detection

A two-stage ML pipeline for detecting and classifying **25+ types of traffic signs** — far beyond basic stop signs and traffic lights. Uses **YOLOv8** for sign localisation and **OpenAI CLIP** for zero-shot classification into a rich taxonomy of sign categories.

Architecture inspired by [VisionBoard-AI](https://github.com/ouiMalika/VisionBoard-AI): Django REST backend + Celery workers + Redis, containerised with Docker Compose.

## Supported Sign Categories (26 types)

| Priority | Categories |
|----------|-----------|
| **Critical (P1)** | stop, yield, no_entry, pedestrian_crossing, school_zone, railroad_crossing, traffic_light |
| **Warning (P2)** | speed_limit, no_turn, one_way, keep_right_left, road_work, curve_warning, intersection_warning, slippery_road, animal_crossing, merge_warning, roundabout, detour, lane_closure |
| **Guide (P3)** | direction, exit, highway_route |
| **Info (P4)** | distance_marker, parking, information |

## How It Works

```
Image → [YOLOv8 Localisation] → Candidate Regions
                                        ↓
                               [CLIP Zero-Shot Classification]
                                        ↓
                               Category + Confidence Score
```

**Stage 1 — YOLOv8** detects objects in the image and produces bounding boxes for candidate sign regions.

**Stage 2 — CLIP** crops each candidate region and classifies it against 90+ text prompts mapped to 26 sign categories. Multiple prompts per category (e.g., "a red octagonal stop sign", "traffic stop sign") improve zero-shot accuracy.

This approach means **no category-specific training data is needed** — adding a new sign type is just adding text prompts to `worker/sign_taxonomy.py`.

## Quick Start

### Option 1: Docker Compose (recommended)

```bash
# Clone and start the full stack
cp .env.example .env
docker compose up -d

# For CPU-only (no GPU):
docker compose --profile cpu up -d worker-cpu
docker compose up -d postgres redis backend

# Run the API test
pip install requests
python scripts/test_api.py
```

The API will be available at `http://localhost:8000/api/`.

### Option 2: CLI Tool (local testing)

```bash
# Install worker dependencies
cd worker
pip install -r requirements.txt

# Detect signs in a local image
python ../scripts/detect_cli.py --image /path/to/road_photo.jpg --output annotated.jpg

# Detect from a URL
python ../scripts/detect_cli.py --url https://example.com/road.jpg -o result.jpg -v

# List all supported categories
python ../scripts/detect_cli.py --list-categories
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/auth/register/` | No | Register a new user |
| POST | `/api/auth/login/` | No | Login, get auth token |
| GET | `/api/taxonomy/` | No | List all 26 sign categories |
| POST | `/api/detect/` | Yes | Submit images for detection |
| GET | `/api/jobs/` | Yes | List your detection jobs |
| GET | `/api/jobs/<job_id>/` | Yes | Poll job status / get results |
| GET | `/api/signs/` | Yes | List detected signs (filterable) |

### Example: Submit images for detection

```bash
# Register
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demopass123"}' | jq -r '.token')

# Submit detection job
JOB=$(curl -s -X POST http://localhost:8000/api/detect/ \
  -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"image_urls":["https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/STOP_sign.jpg/220px-STOP_sign.jpg"]}' \
  | jq -r '.job_id')

# Poll until complete
curl -s http://localhost:8000/api/jobs/$JOB/ -H "Authorization: Token $TOKEN" | jq

# Filter results by category
curl -s "http://localhost:8000/api/signs/?category=stop" -H "Authorization: Token $TOKEN" | jq
```

## Project Structure

```
perception-aware/
├── backend/                    # Django REST API
│   ├── perception_backend/     # Django project settings
│   ├── sign_detection/         # App: models, views, serializers, URLs
│   ├── manage.py
│   ├── requirements.txt
│   └── Dockerfile
├── worker/                     # Celery ML worker
│   ├── sign_taxonomy.py        # 26 sign categories, 90+ CLIP prompts
│   ├── detector.py             # Two-stage YOLOv8 + CLIP pipeline
│   ├── visualizer.py           # Bounding box annotation drawing
│   ├── tasks.py                # Celery task definitions
│   ├── requirements.txt
│   └── Dockerfile
├── tests/
│   ├── unit/                   # Fast tests (no models needed)
│   └── integration/            # Full pipeline tests (needs models)
├── scripts/
│   ├── detect_cli.py           # CLI tool for local testing
│   └── test_api.py             # API endpoint test script
├── docker-compose.yml          # Full stack orchestration
├── .env.example                # Environment variable template
└── pytest.ini                  # Test configuration
```

## Running Tests

```bash
# Unit tests (no GPU/models required)
pip install pytest
pytest tests/unit/ -v

# Integration tests (requires models downloaded)
INTEGRATION_TESTS=1 pytest tests/integration/ -v
```

## Adding New Sign Types

To add a new sign category, edit `worker/sign_taxonomy.py`:

```python
SIGN_TAXONOMY["new_sign_type"] = {
    "description": "Description of the sign",
    "priority": 2,  # 1=critical, 2=warning, 3=guide, 4=info
    "prompts": [
        "a descriptive prompt for CLIP",
        "another variation of the prompt",
        "a third prompt for better accuracy",
    ],
}
```

No retraining needed — CLIP handles the new prompts via zero-shot classification.

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_SIZE` | `yolov8m` | YOLOv8 model variant (n/s/m/l/x) |
| `DETECTION_CONFIDENCE_THRESHOLD` | `0.25` | YOLO detection threshold |
| `CLASSIFICATION_CONFIDENCE_THRESHOLD` | `0.15` | CLIP classification threshold |
| `MAX_DETECTIONS_PER_IMAGE` | `50` | Max signs detected per image |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | Django 5.0 + Django REST Framework |
| Task Queue | Celery 5.3 + Redis |
| Database | PostgreSQL 16 |
| Object Detection | YOLOv8 (Ultralytics) |
| Classification | OpenAI CLIP (zero-shot) |
| Visualization | Pillow |
| Deployment | Docker Compose |
