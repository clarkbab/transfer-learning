from colorlog import ColoredFormatter
import logging

LEVEL_MAP = {
    10: 'DEBUG',
    20: 'INFO',
    30: 'WARNING'
}

def config(level: str) -> None:
    level = getattr(logging, level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Logging level '{level}' not valid.")
    log_format = "%(log_color)s%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = ColoredFormatter(log_format, date_format)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logging.basicConfig(handlers=[stream], level=level)

def set_level(level: str) -> None:
    level = getattr(logging, level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Logging level '{level}' not valid.")
    logging.set_level(level)

def info(*args, **kwargs):
    return logging.info(*args, **kwargs)

def level():
    return LEVEL_MAP[logging.root.level]

def warning(*args, **kwargs):
    return logging.warning(*args, **kwargs)

def error(*args, **kwargs):
    return logging.error(*args, **kwargs)

# Default config.
config('info')
