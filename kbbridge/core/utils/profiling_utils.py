import time
from contextlib import contextmanager


@contextmanager
def profile_stage(stage_name: str, profiling_data: dict, verbose: bool = False):
    """Context manager for profiling stages when verbose mode is enabled

    Args:
        stage_name: Name of the stage being profiled
        profiling_data: Dictionary to store profiling results
        verbose: Whether to enable profiling (no-op if False)

    Yields:
        None

    Example:
        >>> profiling_data = {}
        >>> with profile_stage("data_loading", profiling_data, verbose=True):
        ...     # Do some work
        ...     pass
        >>> print(profiling_data)
        {'data_loading': {'duration_seconds': 0.123, 'duration_ms': 123.0}}
    """
    if not verbose:
        yield
        return

    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        profiling_data[stage_name] = {
            "duration_seconds": round(duration, 3),
            "duration_ms": round(duration * 1000, 1),
        }
