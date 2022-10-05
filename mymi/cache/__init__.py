from typing import *

from .cache import Cache

# Create cache
cache = Cache()

def config(**kwargs: dict) -> None:
    """
    effect: configures the cache.
    kwargs:
        path: the path to the cache.
        read_enabled: enables cache read.
        write_enabled: enables cache write.
    """
    # Set cache path.
    path = kwargs.pop('path', None)
    if path is not None:
        cache.path = path

    # Set enabled flags.
    cache.read_enabled = kwargs.pop('read_enabled', True)
    cache.write_enabled = kwargs.pop('write_enabled', True)

    # Set logging flag.
    cache.logging = kwargs.pop('logging', False)

def disable() -> None:
    """
    effect: disables the cache.
    """
    cache.read_enabled = False
    cache.write_enabled = False

def enable() -> None:
    """
    effect: enables the cache.
    """
    cache.read_enabled = True
    cache.write_enabled = True

def read(*args: Sequence[Any], **kwargs: dict) -> Any:
    return cache.read(*args, **kwargs)

def write(*args: Sequence[Any], **kwargs: dict) -> Any:
    return cache.write(*args, **kwargs)

def delete(*args, **kwargs):
    return cache.delete(*args, **kwargs)

def function(*args, **kwargs):
    return cache.function(*args, **kwargs)

def method(*args, **kwargs):
    return cache.method(*args, **kwargs)
