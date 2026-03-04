"""Thin wrappers around Claude/OpenAI chat APIs."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


def _read_dotenv_values() -> Dict[str, str]:
    values: Dict[str, str] = {}
    dotenv_path = Path(".env")
    if not dotenv_path.exists():
        return values
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                values[key] = value
    except Exception as exc:
        logger.warning("Failed reading .env for LLM keys: %s", exc)
    return values


def _get_env_or_dotenv(name: str, default: str = "") -> str:
    direct = os.getenv(name)
    if direct:
        return direct
    return _read_dotenv_values().get(name, default)


def has_llm_api_key() -> bool:
    return bool(_get_env_or_dotenv("ANTHROPIC_API_KEY") or _get_env_or_dotenv("OPENAI_API_KEY"))


def _get_openai_model() -> str:
    # Use a widely-available model by default; allow override via env/.env.
    return _get_env_or_dotenv("OPENAI_MODEL", "gpt-4o-mini")


async def call_claude_api(
    messages: List[Dict[str, str]],
    system: str,
    max_tokens: int = 1024,
) -> Optional[str]:
    anthropic_api_key = _get_env_or_dotenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not set")
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": messages,
                },
            )
            if response.status_code == 200:
                data = response.json()
                return data["content"][0]["text"]
            logger.error(f"Claude API error: {response.status_code} - {response.text}")
            return None
    except Exception as exc:
        logger.exception(f"Claude API call failed: {exc}")
        return None


async def call_openai_api(
    messages: List[Dict[str, str]],
    system: str,
    max_tokens: int = 1024,
) -> Optional[str]:
    openai_api_key = _get_env_or_dotenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not set")
        return None
    openai_model = _get_openai_model()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": openai_model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "system", "content": system}, *messages],
                },
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            # Common setup issue: unsupported or unavailable model name.
            if response.status_code in {400, 404} and openai_model != "gpt-4o-mini":
                fallback_response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "max_tokens": max_tokens,
                        "messages": [{"role": "system", "content": system}, *messages],
                    },
                )
                if fallback_response.status_code == 200:
                    data = fallback_response.json()
                    return data["choices"][0]["message"]["content"]
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return None
    except Exception as exc:
        logger.exception(f"OpenAI API call failed: {exc}")
        return None
