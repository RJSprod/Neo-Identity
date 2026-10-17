"""One logger for the extension, shown on Forge's console.

Forge's own loggers get a handler from ``backend.logging``; an extension's does
not, and a logger with no handler prints nothing below WARNING. The handler is
attached once, here, and only when nothing else has configured the logger.
"""

from __future__ import annotations

import logging

NAME = "neo_identity"
PREFIX = "[Neo Identity]"

logger = logging.getLogger(NAME)


def ensure_handler() -> logging.Logger:
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(f"{PREFIX} %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger
