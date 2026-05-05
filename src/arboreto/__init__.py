from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from .algo import genie3, grnboost2

try:
    if isinstance(__package__, str):
        __version__ = version(__package__)
    else:
        __version__ = "unknown"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

logger.disable(__package__)

__all__ = [
    "genie3",
    "grnboost2",
]