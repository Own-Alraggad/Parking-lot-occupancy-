"""Detection backend abstraction and the Ultralytics YOLO implementation.

This module defines the model-agnostic contract every detection backend must
satisfy (`DetectionBackend`) and the canonical result type (`Detection`) that
everything downstream -- postprocessing, services, response schemas -- reads.
Nothing outside this module knows what library actually produced a detection.

Swapping the underlying model (YOLO -> RT-DETR -> Faster R-CNN, etc.) means
writing one new class here that implements `DetectionBackend` and adding it
to `get_backend()`'s registry. No other file in the project changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object, in a library-agnostic shape.

    Coordinates are absolute pixel values in the *original* input image's
    coordinate space -- each backend is responsible for un-letterboxing /
    un-scaling its own internal representation before returning results, so
    callers never need to know how a given backend resized or padded.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@runtime_checkable
class DetectionBackend(Protocol):
    """Contract every detection backend must satisfy.

    Deliberately minimal -- three members cover everything the rest of the
    app needs, so implementing a new backend is a small, self-contained
    piece of work rather than a sprawling interface to satisfy. Protocols
    use structural typing: a class satisfies this without inheriting from
    it, as long as it has matching methods/properties.
    """

    def load(self) -> None:
        """Load weights into memory. Called exactly once, at app startup."""
        ...

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Detection]:
        """Run inference on a single decoded image (H, W, 3 uint8, RGB).

        Preprocessing (resize/letterbox/normalize) and postprocessing (NMS,
        rescaling boxes back to the original image size) are the backend's
        own responsibility -- they differ per model family and deliberately
        don't live in shared application code.
        """
        ...

    @property
    def class_names(self) -> dict[int, str]:
        """Mapping of class_id -> human-readable class name."""
        ...


class UltralyticsYoloBackend:
    """DetectionBackend implementation wrapping an Ultralytics YOLO model.

    Satisfies the `DetectionBackend` Protocol structurally (see the
    `isinstance` check in `get_backend()`) -- no inheritance needed.
    """

    def __init__(self, weights_path: Path, device: str, image_size: int) -> None:
        self._weights_path = weights_path
        self._device = device
        self._image_size = image_size
        # None until load() runs. This makes calling predict() before load()
        # fail loudly (RuntimeError below) instead of e.g. silently loading
        # weights on first request, which would defeat "load once at startup".
        self._model = None
        self._resolved_device: str | None = None

    def load(self) -> None:
        """Load the Ultralytics model from disk and move it to the target device.

        Deferred out of __init__ so constructing a backend instance is cheap
        and side-effect-free (useful in tests that build one without
        touching disk/GPU). This is the method the app's lifespan handler
        calls explicitly, exactly once, at startup.
        """
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{self._weights_path}'. "
                "Check the MODEL_PATH setting and confirm the weights file "
                "was actually copied into the deployment."
            )

        # Imported inside the method, not at module level, so importing this
        # module (e.g. for type-checking, or unit tests that substitute a
        # fake backend) never requires ultralytics/torch to be installed.
        from ultralytics import YOLO

        logger.info("Loading Ultralytics model from %s (device=%s)", self._weights_path, self._device)
        self._model = YOLO(str(self._weights_path))
        self._resolved_device = self._resolve_device()
        self._model.to(self._resolved_device)
        logger.info("Model loaded successfully on device=%s", self._resolved_device)

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Detection]:
        if self._model is None:
            raise RuntimeError(
                "UltralyticsYoloBackend.predict() called before load(). "
                "This indicates a startup/dependency-injection wiring bug, "
                "not a normal request-time condition."
            )

        import torch

        with torch.inference_mode():
            results = self._model.predict(
                source=image,
                imgsz=self._image_size,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self._resolved_device,
                verbose=False,
            )

        result = results[0]
        names = result.names  # dict[int, str], already resolved by Ultralytics
        detections: list[Detection] = []

        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), score, class_id in zip(xyxy, conf, cls):
                detections.append(
                    Detection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=float(score),
                        class_id=int(class_id),
                        class_name=str(names.get(int(class_id), class_id)),
                    )
                )

        return detections

    @property
    def class_names(self) -> dict[int, str]:
        if self._model is None:
            raise RuntimeError("class_names accessed before load().")
        return dict(self._model.names)

class OnnxRuntimeBackend(DetectionBackend):
    """ONNX Runtime backend for exported ONNX model execution."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = str(model_path)
        self.device = device
        self._session = None
        self._class_names = {0: "object"}  # Default fallback if metadata isn't embedded

    def load(self) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(self.model_path, providers=providers)

        # Optional: Read embedded class names metadata if available in ONNX model
        meta = self._session.get_modelmeta().custom_metadata_map
        print("Metadata:", meta)

        if "names" in meta:
            import json
            import ast

            raw = meta["names"]

            try:
                names = json.loads(raw)
            except json.JSONDecodeError:
                names = ast.literal_eval(raw)

            self._class_names = {
                int(k): v
                for k, v in names.items()
            }

    def predict(
        self, image: np.ndarray, conf_threshold: float, iou_threshold: float
    ) -> list[Detection]:
        if self._session is None:
            raise RuntimeError("Backend not loaded. Call load() before predict().")

        # 1. Preprocess: Resize, normalize (0-1), CHW conversion, batch dimension
        # (Note: Standard ONNX models require explicit array transformation)
        h_orig, w_orig = image.shape[:2]
        input_tensor = self._preprocess_image(image)

        # 2. Forward pass
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_tensor})

        # 3. Postprocess raw tensor outputs -> canonical Detection list
        return self._parse_onnx_outputs(
            outputs, conf_threshold, iou_threshold, w_orig, h_orig
        )

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(img, (512, 512))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC to CHW
        return np.expand_dims(blob, axis=0)  # Add batch dimension (1, 3, 640, 640)

    def _parse_onnx_outputs(
        self, outputs, conf_thresh, iou_thresh, orig_w, orig_h
    ) -> list[Detection]:
        # Parse model-specific tensor output shape (e.g. YOLOv8 outputs shape [1, 84, 8400])
        # Rescale normalized bounding box coordinates back to orig_w, orig_h
        detections = []
        # ... parse logic & NMS filtering here ...
        return detections
