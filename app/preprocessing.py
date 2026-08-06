import io
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import Settings
from exceptions import InvalidImageError


def validate_and_decode_image(
        image_bytes: bytes,
        settings: Settings,
) -> np.ndarray:
    """Validates uploaded raw binary bytes and decodes them into an RGB NumPy array.

    Args:
        image_bytes: Raw binary content received from the HTTP request upload.
        settings: Application settings for file size and dimension checks.

    Returns:
        np.ndarray: Decoded image array in RGB format with shape (H, W, 3).

    Raises:
        InvalidImageError: If payload is empty, exceeds max size, or fails decoding.
    """
    if not image_bytes:
        raise InvalidImageError("Uploaded file is empty.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise InvalidImageError(
            f"File size exceeds maximum allowed limit of {settings.max_upload_size_mb} MB."
        )

    try:
        # Load binary stream into Pillow
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            # Verify file integrity to detect corrupted files
            pil_img.verify()

        # Re-open stream since verify() mutates the file pointer/state
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            # Standardize color space to 3-channel RGB (handles RGBA, Palette, Grayscale)
            converted_img = pil_img.convert("RGB")

            width, height = converted_img.size
            if width == 0 or height == 0:
                raise InvalidImageError("Image contains invalid zero-dimension bounds.")

            return np.array(converted_img, dtype=np.uint8)

    except (UnidentifiedImageError, OSError, ValueError) as err:
        raise InvalidImageError(
            f"Failed to decode image file. Ensure it is a valid JPEG, PNG, or WebP. Details: {str(err)}"
        ) from err