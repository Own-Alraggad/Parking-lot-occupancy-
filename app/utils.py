import time
from contextlib import contextmanager
from typing import Callable, Generator


@contextmanager
def timer() -> Generator[Callable[[], float], None, None]:
    """Context manager to measure execution time in milliseconds.

    Yields a zero-argument function that returns the elapsed time in ms when called.

    Usage:
        with timer() as get_elapsed:
            do_something()
        latency_ms = get_elapsed()
    """
    start_time = time.perf_counter()
    elapsed_fn = lambda: (time.perf_counter() - start_time) * 1000.0
    try:
        yield elapsed_fn
    finally:
        pass


def measure_execution_time(func):
    """Decorator to log or measure function execution duration in milliseconds."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return result, duration_ms
    return wrapper