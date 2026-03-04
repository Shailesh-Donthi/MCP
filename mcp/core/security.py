"""Security helpers used by HTTP API."""

from __future__ import annotations

from typing import Any

import jwt

from mcp.core.config import settings


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except Exception:
        return None
