import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import Settings
from dependencies import cleanup_backend, get_settings, init_backend
from exceptions import register_exception_handlers
from logging_config import setup_logging
from middleware.request_logging import RequestLoggingMiddleware
from routers import health, prediction

logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager controlling application startup and shutdown.

    Loads the model backend once on startup and cleans up memory on shutdown.
    """
    settings = get_settings()

    # Step 1: Configure structured logging
    setup_logging(settings)
    logger.info(f"Starting {settings.app_name} [{settings.environment}]")

    # Step 2: Initialize model backend (loads weights once during startup)
    logger.info(f"Initializing backend: '{settings.model_backend}' on device '{settings.device}'...")
    init_backend(settings)
    logger.info("Model backend loaded and ready for inference.")

    yield  # Application processes requests here

    # Step 3: Shutdown cleanup
    logger.info("Shutting down application and releasing backend resources...")
    cleanup_backend()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Application factory for configuring the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Production-ready, backend-agnostic Object Detection API service.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )

    # Register custom middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Register custom exception handlers
    register_exception_handlers(app)

    # Register API routers
    app.include_router(health.router)
    app.include_router(prediction.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=(settings.environment == "development"),
    )