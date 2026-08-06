from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import ErrorDetails, ErrorResponse


class AppError(Exception):
    """Base exception for application-level errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class InvalidImageError(AppError):
    """Raised when uploaded file is empty, corrupted, or unsupported format."""

    def __init__(
        self,
        message: str = "Invalid image upload. File must be a valid JPEG, PNG, or WebP image.",
    ):
        super().__init__(
            message=message,
            code="INVALID_IMAGE_FORMAT",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class ModelNotLoadedError(AppError):
    """Raised if inference is called before backend initialization/load."""

    def __init__(
        self,
        message: str = "Inference backend is not loaded. Ensure startup lifespan has completed.",
    ):
        super().__init__(
            message=message,
            code="MODEL_NOT_LOADED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class InferenceError(AppError):
    """Raised when model forward pass or backend prediction fails."""

    def __init__(self, message: str = "Failed to execute model inference."):
        super().__init__(
            message=message,
            code="INFERENCE_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Registers standard JSON exception handlers on the FastAPI application instance."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                success=False,
                error=ErrorDetails(code=exc.code, message=exc.message),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Formats FastAPI/Pydantic request input validation errors into canonical ErrorResponse
        errors = exc.errors()
        error_msg = (
            f"Validation error: {errors[0]['msg']}" if errors else "Invalid request body."
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                success=False,
                error=ErrorDetails(code="VALIDATION_ERROR", message=error_msg),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Catches unhandled global exceptions to prevent raw tracebacks from reaching API clients
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                success=False,
                error=ErrorDetails(
                    code="INTERNAL_SERVER_ERROR",
                    message="An unexpected server error occurred.",
                ),
            ).model_dump(),
        )