import logging
from config import Settings
from model import DetectionBackend
from postprocessing import format_detection_response
from preprocessing import validate_and_decode_image
from schemas import ObjectDetectionResponse
from utils import timer

logger = logging.getLogger("api.service.detector")


class ObjectDetectionService:
    """Service layer that orchestrates image validation, model prediction, and response formatting."""

    def __init__(self, backend: DetectionBackend, settings: Settings):
        self.backend = backend
        self.settings = settings

    def detect_objects(
        self,
        image_bytes: bytes,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> ObjectDetectionResponse:
        """Executes the full object detection pipeline on raw upload bytes.

        Args:
            image_bytes: Raw binary bytes uploaded by the client.
            conf_threshold: Optional override for confidence threshold.
            iou_threshold: Optional override for NMS IoU threshold.

        Returns:
            ObjectDetectionResponse: Structured payload containing detections and timing metadata.
        """
        with timer() as get_total_time:
            # Resolve thresholds against application settings defaults
            effective_conf = (
                conf_threshold
                if conf_threshold is not None
                else self.settings.default_confidence_threshold
            )
            effective_iou = (
                iou_threshold
                if iou_threshold is not None
                else self.settings.default_iou_threshold
            )

            # Step 1: Preprocessing (Validate upload & decode into np.ndarray)
            image_array = validate_and_decode_image(image_bytes, self.settings)

            # Step 2: Model Inference (Measure pure forward pass latency)
            with timer() as get_inference_time:
                raw_detections = self.backend.predict(
                    image=image_array,
                    conf_threshold=effective_conf,
                    iou_threshold=effective_iou,
                )
            inference_time_ms = get_inference_time()

            logger.debug(
                f"Inference complete: {len(raw_detections)} objects detected in {inference_time_ms:.2f}ms"
            )

            response = format_detection_response(
            detections=raw_detections,
            inference_time_ms=inference_time_ms,
            min_confidence=effective_conf,
            )

        total_processing_ms = get_total_time()

        logger.info(
            "Processing %.2f ms | Inference %.2f ms",
            total_processing_ms,
            inference_time_ms,
        )

        # Step 3: Postprocessing (Format canonical detections into Pydantic REST schema)
        return response