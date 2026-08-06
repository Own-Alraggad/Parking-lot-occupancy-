from functools import lru_cache
from typing import Generator

from config import Settings
from model import DetectionBackend, get_backend

# Global singleton instance for the backend model
_backend_instance: DetectionBackend | None = None


@lru_cache
def get_settings() -> Settings:
    """Returns a cached instance of the application settings."""
    return Settings()


def init_backend(settings: Settings | None = None) -> DetectionBackend:
    """Initializes and loads the model backend during startup.

    Should be called explicitly during FastAPI lifespan startup in main.py.
    """
    global _backend_instance
    if settings is None:
        settings = get_settings()

    _backend_instance = get_backend(settings)
    _backend_instance.load()
    return _backend_instance


def get_backend_instance() -> DetectionBackend:
    """Dependency provider that yields the loaded DetectionBackend singleton.

    Raises:
        ModelNotLoadedError: If accessed before init_backend() was called during startup.
    """
    global _backend_instance
    if _backend_instance is None:
        from exceptions import ModelNotLoadedError

        raise ModelNotLoadedError()
    return _backend_instance


def cleanup_backend() -> None:
    """Resets the global backend instance during application shutdown."""
    global _backend_instance
    _backend_instance = None