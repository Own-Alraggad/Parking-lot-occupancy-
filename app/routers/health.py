from typing import Any, Dict

from fastapi import APIRouter, Depends, status

from config import Settings
from dependencies import get_backend_instance, get_settings
from model import DetectionBackend

router = APIRouter(tags=["Health"])


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Root endpoint",
    description="Provides basic service info and version details.",
)
async def root_info(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """Returns basic service metadata."""
    return {
        "service": settings.app_name,
        "status": "online",
        "environment": settings.environment,
        "docs_url": "/docs",
    }


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Endpoint for readiness/liveness probes to verify backend status.",
)
async def health_check(
    backend: DetectionBackend = Depends(get_backend_instance),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Health check endpoint confirming that model backend is loaded."""
    return {
        "status": "healthy",
        "backend": settings.model_backend,
        "device": settings.device,
        "num_classes": len(backend.class_names),
    }