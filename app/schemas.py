from typing import List, Optional
from pydantic import BaseModel, Field


class DetectionBoundingBox(BaseModel):
    """Normalized bounding box coordinates in pixel space (un-letterboxed)."""

    x1: float = Field(..., description="Top-left X coordinate in pixels", example=120.5)
    y1: float = Field(..., description="Top-left Y coordinate in pixels", example=45.0)
    x2: float = Field(
        ..., description="Bottom-right X coordinate in pixels", example=350.2
    )
    y2: float = Field(
        ..., description="Bottom-right Y coordinate in pixels", example=480.8
    )


class SingleDetectionResponse(BaseModel):
    """Schema representing a single detected object in an image.

    Directly maps to the backend-agnostic canonical `Detection` dataclass.
    """

    box: DetectionBoundingBox = Field(
        ..., description="Bounding box pixel coordinates"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score between 0 and 1", example=0.94
    )
    class_id: int = Field(
        ..., ge=0, description="Zero-indexed class ID from the model", example=0
    )
    class_name: str = Field(
        ..., description="Human-readable class label from model metadata", example="person"
    )


class ObjectDetectionResponse(BaseModel):
    """Top-level response contract for object detection requests."""

    success: bool = Field(
        True, description="Indicates if the inference pipeline executed successfully"
    )
    count: int = Field(
        ..., ge=0, description="Total number of objects detected after filtering", example=3
    )
    inference_time_ms: float = Field(
        ..., ge=0.0, description="Model forward-pass latency in milliseconds", example=24.5
    )
    detections: List[SingleDetectionResponse] = Field(
        default_factory=list, description="List of detected objects"
    )


class ErrorDetails(BaseModel):
    """Standardized error payload format."""

    code: str = Field(..., description="Machine-readable error code", example="INVALID_IMAGE_FORMAT")
    message: str = Field(..., description="Human-readable error description", example="Failed to decode image.")


class ErrorResponse(BaseModel):
    """Top-level error response model."""

    success: bool = Field(False, description="Always False for error responses")
    error: ErrorDetails