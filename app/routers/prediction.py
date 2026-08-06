from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from config import Settings
from dependencies import get_backend_instance, get_settings
from model import DetectionBackend
from schemas import ErrorResponse, ObjectDetectionResponse
from services.detector import ObjectDetectionService

router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=ObjectDetectionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format or size limit exceeded"},
        422: {"model": ErrorResponse, "description": "Validation error in parameters"},
        500: {"model": ErrorResponse, "description": "Internal model execution failure"},
        503: {"model": ErrorResponse, "description": "Model backend not initialized"},
    },
    summary="Detect objects in an uploaded image",
    description="Upload a raw image file (JPEG, PNG, WebP) to run object detection inference.",
)
async def predict(
    file: UploadFile = File(..., description="Image file binary content"),
    conf_threshold: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override minimum confidence score threshold (0.0 - 1.0)",
    ),
    iou_threshold: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override NMS IoU threshold (0.0 - 1.0)",
    ),
    backend: DetectionBackend = Depends(get_backend_instance),
    settings: Settings = Depends(get_settings),
) -> ObjectDetectionResponse:
    """Endpoint handling image uploads and returning bounding box detections."""
    image_bytes = await file.read()

    service = ObjectDetectionService(backend=backend, settings=settings)
    return service.detect_objects(
        image_bytes=image_bytes,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )