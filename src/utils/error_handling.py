"""
Error handling definitions and retry utilities.
"""
import time
from functools import wraps
from typing import Callable, Any
from .logging_config import logger

class CrossSensorException(Exception):
    """Base exception for cross-sensor translation system."""
    pass

class DatasetAccessError(CrossSensorException):
    """Raised when remote dataset access fails or is denied."""
    pass

class DataValidationError(CrossSensorException):
    """Raised when image pairs fail integrity or alignment checks."""
    pass

class ModelExecutionError(CrossSensorException):
    """Raised when model inference or training encounters an issue."""
    pass

def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 1.0, exceptions=(Exception,)):
    """Decorator to retry flaky remote network requests with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            _tries = retries
            _delay = backoff_in_seconds
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Error in {func.__name__}: {e}. Retrying in {_delay:.1f}s... ({_tries-1} retries left)")
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= 2
            # Final attempt
            return func(*args, **kwargs)
        return wrapper
    return decorator
