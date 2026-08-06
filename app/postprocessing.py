from typing import List

from model import Detection
from schemas import (
    DetectionBoundingBox,
    ObjectDetectionResponse,
    SingleDetectionResponse,
)


def format_detection_response(
    detections: List[Detection],
    inference_time_ms: float,
    min_confidence: float = 0.0,
) -> ObjectDetectionResponse:
    """Converts a list of backend-agnostic Detection objects into an ObjectDetectionResponse.

    Args:
        detections: List of canonical Detection instances output by the backend.
        inference_time_ms: Measured forward-pass model latency in milliseconds.
        min_confidence: Optional additional confidence threshold cutoff.

    Returns:
        ObjectDetectionResponse: Formatted top-level API response payload.
    """
    formatted_detections: List[SingleDetectionResponse] = []

    for det in detections:
        if det.confidence < min_confidence:
            continue

        single_response = SingleDetectionResponse(
            box=DetectionBoundingBox(
                x1=round(float(det.x1), 2),
                y1=round(float(det.y1), 2),
                x2=round(float(det.x2), 2),
                y2=round(float(det.y2), 2),
            ),
            confidence=round(float(det.confidence), 4),
            class_id=int(det.class_id),
            class_name=str(det.class_name),
        )
        formatted_detections.append(single_response)

    return ObjectDetectionResponse(
        success=True,
        count=len(formatted_detections),
        inference_time_ms=round(inference_time_ms, 2),
        detections=formatted_detections,
    )