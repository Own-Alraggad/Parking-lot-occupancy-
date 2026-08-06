import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log incoming requests, measure total HTTP processing time,
    and attach a unique X-Request-ID header to responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Attach request_id to state so downstream routes/services can access it
        request.state.request_id = request_id

        logger.info(
            f"Incoming request: {request.method} {request.url.path} [ID: {request_id}]"
        )

        try:
            response = await call_next(request)
            process_time_ms = (time.perf_counter() - start_time) * 1000.0

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-MS"] = f"{process_time_ms:.2f}"

            logger.info(
                f"Completed request: {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Duration: {process_time_ms:.2f}ms [ID: {request_id}]"
            )
            return response

        except Exception as exc:
            process_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                f"Unhandled error processing request: {request.method} {request.url.path} "
                f"Duration: {process_time_ms:.2f}ms [ID: {request_id}] - Error: {str(exc)}"
            )
            raise exc