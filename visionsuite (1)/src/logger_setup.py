"""
logger_setup.py
---------------
Centralised logging for VisionSuite.

* A rotating file handler writes detailed logs to ``<output_dir>/logs/visionsuite.log``
  (the file never grows beyond ~1 MB x 3 backups, which caps disk usage).
* The console stays quiet by default because the CLI prints its own
  user-facing messages. Set ``log_level: DEBUG`` in config.yaml to also
  stream log lines to the terminal.

Usage in any module::

    from src.logger_setup import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

ROOT_LOGGER_NAME = "visionsuite"
LOG_FILE_NAME = "visionsuite.log"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

_file_handler = None
_console_handler = None


def _root() -> logging.Logger:
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)   # handlers decide what is actually emitted
    logger.propagate = False
    return logger


def _ensure_console_handler() -> logging.Handler:
    global _console_handler
    if _console_handler is None:
        _console_handler = logging.StreamHandler()
        _console_handler.setFormatter(logging.Formatter(_FORMAT))
        _console_handler.setLevel(logging.CRITICAL)   # quiet by default
        _root().addHandler(_console_handler)
    return _console_handler


def configure_logging(log_dir: str = os.path.join("outputs", "logs"),
                      level: str = "INFO") -> None:
    """(Re)configure handlers. Safe to call more than once."""
    global _file_handler
    level_no = getattr(logging, str(level).upper(), logging.INFO)
    logger = _root()

    # --- rotating file handler ------------------------------------------
    if _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None
    try:
        os.makedirs(log_dir, exist_ok=True)
        _file_handler = RotatingFileHandler(
            os.path.join(log_dir, LOG_FILE_NAME),
            maxBytes=1_000_000, backupCount=3, encoding="utf-8",
        )
        _file_handler.setLevel(level_no)
        _file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(_file_handler)
    except OSError:
        # Logging must never be the reason the tool fails.
        _file_handler = None

    # --- console handler (quiet unless DEBUG) ---------------------------
    console = _ensure_console_handler()
    console.setLevel(logging.DEBUG if level_no == logging.DEBUG else logging.CRITICAL)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the VisionSuite root logger.

    Modules are imported before the config (and therefore the output
    directory) is known, so only a console handler is attached here; the
    file handler is added later by ``configure_logging``.
    """
    _ensure_console_handler()
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
