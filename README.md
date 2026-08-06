# Parking Lot Occupancy Detection

A computer vision system that detects and counts occupied and vacant parking spaces from images. It is built on top of a YOLO11 object detection model and served through a production ready REST API backed by FastAPI.

---

## What It Does

The system takes an image of a parking lot and returns bounding box predictions for each detected space, along with a confidence score and class label. The model can be swapped between an Ultralytics YOLO backend and an ONNX Runtime backend without changing any other part of the codebase.

---

## Project Structure

```
parking lot occupancy/
├── app/                        FastAPI application
│   ├── main.py                 Application entry point and lifespan handler
│   ├── config.py               All runtime settings sourced from environment variables
│   ├── model.py                Backend abstraction and YOLO / ONNX implementations
│   ├── schemas.py              Request and response data models
│   ├── routers/                API route definitions
│   │   ├── health.py           Health check endpoint
│   │   └── prediction.py       Object detection endpoint
│   ├── services/               Business logic layer
│   ├── middleware/             Request logging middleware
│   └── weights/                Model weight files go here
├── 01_dataset_quality.ipynb    Dataset exploration and quality checks
├── 02_train_yolo11.ipynb       Model training notebook
├── convert_coco_to_yolo.py     Converts COCO annotations to YOLO format
├── data/                       Raw and processed dataset files
├── reports/                    Training reports and evaluation outputs
├── Requirements.txt            All Python dependencies
└── runs/                       YOLO training run outputs
```

---

## API Endpoints

### `POST /predict`

Upload an image and receive a list of detected objects with bounding boxes.

**Query parameters:**

| Parameter        | Type  | Default | Description                              |
|------------------|-------|---------|------------------------------------------|
| `conf_threshold` | float | 0.25    | Minimum confidence score to keep a detection |
| `iou_threshold`  | float | 0.45    | IoU threshold used during non-maximum suppression |

**Accepted image formats:** JPEG, PNG, WebP (up to 10 MB)

**Example response:**

```json
{
  "detections": [
    {
      "x1": 120.4,
      "y1": 85.2,
      "x2": 240.8,
      "y2": 195.6,
      "confidence": 0.91,
      "class_id": 0,
      "class_name": "occupied"
    }
  ]
}
```

### `GET /health`

Returns the current health status of the service. Useful for container orchestration readiness probes.

---

## Getting Started

### Prerequisites

Python 3.10 or higher is required. A CUDA capable GPU is strongly recommended for real time inference, though the service will fall back to CPU automatically.

### Installation

Install all dependencies from the requirements file:

```bash
pip install -r Requirements.txt
```

Place your trained model weights inside the `app/weights/` directory. The default expected filename is `best.engine` for the TensorRT engine backend, but this can be changed through environment variables.

### Configuration

All settings are controlled through environment variables or a `.env` file placed in the `app/` directory. The table below lists the most commonly changed ones.

| Variable                      | Default               | Description                                      |
|-------------------------------|-----------------------|--------------------------------------------------|
| `MODEL_BACKEND`               | `onnx`                | Which backend to use (`ultralytics_yolo` or `onnx`) |
| `MODEL_PATH`                  | `weights/best.engine` | Path to the model weights file                   |
| `DEVICE`                      | `auto`                | Compute device (`auto`, `cpu`, or `cuda`)        |
| `ENVIRONMENT`                 | `development`         | Controls debug docs and log verbosity            |
| `DEFAULT_CONFIDENCE_THRESHOLD`| `0.25`                | Default confidence threshold for detections      |
| `LOG_LEVEL`                   | `INFO`                | Logging verbosity level                          |
| `CORS_ALLOWED_ORIGINS`        | *(empty)*             | Comma separated list of allowed CORS origins     |

### Running the API

From the `app/` directory:

```bash
python main.py
```

The API will be available at `http://localhost:8000`. Interactive documentation is served at `http://localhost:8000/docs` in development mode.

---

## Training

Open and run the notebooks in order:

1. **`01_dataset_quality.ipynb`** — Inspect the dataset, check class distributions, and verify annotation quality.
2. **`02_train_yolo11.ipynb`** — Fine tune a YOLO11 model on the parking lot dataset.

If your annotations are in COCO format, convert them first:

```bash
python convert_coco_to_yolo.py
```

---

## Model Backends

The application supports two interchangeable inference backends. Switch between them by setting the `MODEL_BACKEND` environment variable.

**Ultralytics YOLO** loads a native `.pt` or `.engine` file using the Ultralytics library. This is the simplest option for local development and experimentation.

**ONNX Runtime** loads an exported `.onnx` file and runs inference through ONNX Runtime. This backend supports both CPU and GPU execution providers and is well suited for deployment environments where you want to avoid a full PyTorch dependency.

Adding a new backend requires only writing one new class in `app/model.py` and registering it in the backend factory. No other file in the project needs to change.

---

## Key Dependencies

| Package             | Purpose                                      |
|---------------------|----------------------------------------------|
| `fastapi`           | Web framework for the REST API               |
| `uvicorn`           | ASGI server                                  |
| `ultralytics`       | YOLO model training and inference            |
| `onnxruntime-gpu`   | ONNX model inference with GPU support        |
| `torch`             | Deep learning runtime (CUDA build)           |
| `opencv-python`     | Image decoding and preprocessing             |
| `pydantic-settings` | Environment variable based configuration     |
| `albumentations`    | Data augmentation during training            |
