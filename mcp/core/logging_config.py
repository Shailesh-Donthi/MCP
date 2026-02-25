"""Central logging configuration and helpers for MCP."""

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from mcp.core.config import settings


_configured = False


def configure_logging() -> None:
    """Configure application logging once per process."""
    global _configured
    if _configured:
        return

    level_name = (getattr(settings, "MCP_LOG_LEVEL", None) or os.getenv("MCP_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            if handler.formatter is None:
                handler.setFormatter(logging.Formatter(log_format))

    log_file = str(getattr(settings, "MCP_LOG_FILE", None) or os.getenv("MCP_LOG_FILE", "")).strip()
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = int(
            getattr(settings, "MCP_LOG_MAX_BYTES", None)
            or os.getenv("MCP_LOG_MAX_BYTES", str(10 * 1024 * 1024))
        )
        backup_count = int(
            getattr(settings, "MCP_LOG_BACKUP_COUNT", None)
            or os.getenv("MCP_LOG_BACKUP_COUNT", "5")
        )
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    _configured = True


def log_structured(logger: logging.Logger, level: str, event_name: str, **fields: Any) -> None:
    """Emit a structured JSON log line with an event name and context fields."""
    payload = {"event": event_name, **fields}
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(json.dumps(payload, default=str))
