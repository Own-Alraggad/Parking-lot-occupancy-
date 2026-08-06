"""Application configuration.

All runtime configuration is centralised here and sourced from environment
variables (optionally via a local .env file). Using a validated settings
object instead of scattered ``os.environ.get(...)`` calls gives us three
things that matter for a production inference service:

1. Fail-fast startup: a malformed or missing value raises immediately when
   the process starts, not the first time a request happens to touch it.
2. Self-documentation: this file is the single place that lists every
   configurable knob, its type, and its default.
3. Testability: tests can construct a ``Settings`` instance directly with
   overrides instead of mutating process-wide environment variables.

``get_settings()`` is cached with ``lru_cache`` so the environment is parsed
once per process and the same instance is reused everywhere via FastAPI's
``Depends(get_settings)`` -- this avoids re-parsing on every request while
still keeping configuration out of module-level global mutable state (it's
a dependency-injected singleton, not a bare global).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings.

    Every field can be overridden by an environment variable of the same
    name (case-insensitive), or by a ``.env`` file in the working directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    app_name: str = "Shelf Product Detector API"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    # Bind host/port are configurable rather than hardcoded so the same
    # image can be run locally, behind a container orchestrator that
    # injects a $PORT, or behind a reverse proxy on a non-default port,
    # without a code change.
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    # Recommended client-facing request timeout. FastAPI/Uvicorn do not
    # enforce a hard request timeout themselves -- this value is intended
    # to be read by the reverse proxy / ASGI server config (documented in
    # the README) and is exposed here so it's declared in one place rather
    # than duplicated across deployment configs.
    request_timeout_seconds: float = Field(default=10.0, gt=0)

    # ------------------------------------------------------------------
    # Model / inference
    # ------------------------------------------------------------------
    model_path: Path = Path("weights/best.engine")

    # "auto" resolves to CUDA if available else CPU (decided in model.py,
    # not here, since torch shouldn't be imported at config-parse time).
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # Registry key selecting which DetectionBackend implementation to load
    # (see model.py's get_backend() factory). Add a new literal value here
    # whenever a new backend class is registered -- this is the single
    # switch that makes the model swappable without touching any other file.
    model_backend: Literal["ultralytics_yolo", "onnx"] = "onnx"

    # Must match (or be compatible with) the imgsz used during training so
    # the model sees the same effective resolution it was fine-tuned at.
    inference_image_size: int = Field(default=512, gt=0)

    default_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    default_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Upload validation / security
    # ------------------------------------------------------------------
    max_upload_size_mb: float = Field(default=10.0, gt=0)
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    # Empty by default (deny all cross-origin requests) -- CORS should be
    # opted into explicitly per deployment, never left wide open by default.
    cors_allowed_origins: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    # Off by default: a pure inference service behind an internal gateway
    # may already be rate-limited upstream. Exposed here so it can be
    # switched on per deployment without a code change.
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = Field(default=60, gt=0)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Structured (JSON) logs are easier to ingest in production log
    # pipelines; human-readable text is easier to read locally.
    log_json: bool = False

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        """Allow CORS origins to be set as a comma-separated string.

        Environment variables are strings by nature. pydantic-settings can
        parse a JSON array string out of the box, but a plain comma-separated
        value (``CORS_ALLOWED_ORIGINS=https://a.com,https://b.com``) is more
        common in .env files and container orchestration configs, so we
        support that form explicitly rather than forcing JSON-array syntax.
        """
        if isinstance(value, str) and value.strip() and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("allowed_mime_types", mode="before")
    @classmethod
    def _parse_mime_types(cls, value: object) -> object:
        """Same comma-separated convenience as CORS origins, see above."""
        if isinstance(value, str) and value.strip() and not value.strip().startswith("["):
            return [mime.strip() for mime in value.split(",") if mime.strip()]
        return value

    @property
    def max_upload_size_bytes(self) -> int:
        """Convenience conversion used by the upload-size guard in preprocessing.py."""
        return int(self.max_upload_size_mb * 1024 * 1024)

    # Deliberately NOT validating that ``model_path`` exists here: config
    # parsing happens at import time, which can occur before weights are
    # mounted/copied into a container (e.g. during a build step or test
    # collection). Existence is checked where it actually matters -- when
    # model.py attempts to load the weights at startup -- so the failure
    # mode there.


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Cached so environment parsing happens exactly once per process; reused
    everywhere via ``Depends(get_settings)``. Tests should bypass this and
    construct ``Settings(...)`` directly with overrides instead of mutating
    environment variables and clearing the cache.
    """
    return Settings()
