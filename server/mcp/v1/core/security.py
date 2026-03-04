"""Security helpers used by HTTP API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import jwt

from mcp.core.config import settings


def create_access_token(
    payload: Mapping[str, Any],
    *,
    expires_in_minutes: int = 480,
) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=max(1, int(expires_in_minutes)))
    claims = dict(payload)
    claims.setdefault("iat", int(now.timestamp()))
    claims.setdefault("exp", int(exp.timestamp()))
    return jwt.encode(
        claims,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except Exception:
        return None
