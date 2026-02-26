"""Thin wrappers around Claude/OpenAI/Azure chat APIs."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

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


def _prefers_azure_openai_config() -> bool:
    explicit_azure_endpoint = (
        _get_env_or_dotenv("AZURE_AI_ENDPOINT")
        or _get_env_or_dotenv("AZURE_OPENAI_ENDPOINT")
    )
    if explicit_azure_endpoint:
        return True
    openai_base_url = (_get_env_or_dotenv("OPENAI_BASE_URL") or "").lower()
    return "azure.com" in openai_base_url


def _get_openai_api_key() -> str:
    if _prefers_azure_openai_config():
        return (
            _get_env_or_dotenv("AZURE_AI_API_KEY")
            or _get_env_or_dotenv("AZURE_OPENAI_API_KEY")
        )
    return (
        _get_env_or_dotenv("OPENAI_API_KEY")
        or _get_env_or_dotenv("AZURE_AI_API_KEY")
        or _get_env_or_dotenv("AZURE_OPENAI_API_KEY")
    )


def _normalize_openai_base_url(raw_base_url: str) -> str:
    base_url = (raw_base_url or "").strip().rstrip("/")
    if not base_url:
        return "https://api.openai.com/v1"
    if base_url.endswith("/openai/v1") or base_url.endswith("/v1"):
        return base_url
    # Azure Foundry / Azure OpenAI endpoints need the OpenAI-compatible v1 prefix.
    if "azure.com" in base_url:
        return f"{base_url}/openai/v1"
    return f"{base_url}/v1"


def _get_openai_base_url() -> str:
    if _prefers_azure_openai_config():
        azure_endpoint = (
            _get_env_or_dotenv("AZURE_AI_ENDPOINT")
            or _get_env_or_dotenv("AZURE_OPENAI_ENDPOINT")
        )
        if azure_endpoint:
            return _normalize_openai_base_url(azure_endpoint)

    configured_base = _get_env_or_dotenv("OPENAI_BASE_URL")
    if configured_base:
        return _normalize_openai_base_url(configured_base)

    azure_endpoint = (
        _get_env_or_dotenv("AZURE_AI_ENDPOINT")
        or _get_env_or_dotenv("AZURE_OPENAI_ENDPOINT")
    )
    if azure_endpoint:
        return _normalize_openai_base_url(azure_endpoint)

    return "https://api.openai.com/v1"


def _get_openai_chat_url() -> str:
    return f"{_get_openai_base_url().rstrip('/')}/chat/completions"


def _is_azure_openai_compatible_base_url(base_url: str) -> bool:
    try:
        hostname = (urlparse(base_url).hostname or "").lower()
    except Exception:
        hostname = ""
    return hostname.endswith(".openai.azure.com") or hostname.endswith(".services.ai.azure.com")


def _build_openai_headers(api_key: str, base_url: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    # Azure OpenAI / Foundry key-based auth expects api-key header for REST calls.
    if _is_azure_openai_compatible_base_url(base_url):
        headers["api-key"] = api_key
        return headers
    headers["Authorization"] = f"Bearer {api_key}"
    return headers


def has_llm_api_key() -> bool:
    return bool(_get_env_or_dotenv("ANTHROPIC_API_KEY") or _get_openai_api_key())


def _get_openai_model() -> str:
    # Use a widely-available model by default; allow override via env/.env.
    if _prefers_azure_openai_config():
        return (
            _get_env_or_dotenv("AZURE_AI_MODEL")
            or _get_env_or_dotenv("AZURE_OPENAI_DEPLOYMENT")
            or "gpt-4o-mini"
        )
    return (
        _get_env_or_dotenv("OPENAI_MODEL")
        or _get_env_or_dotenv("AZURE_AI_MODEL")
        or _get_env_or_dotenv("AZURE_OPENAI_DEPLOYMENT")
        or "gpt-4o-mini"
    )


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
    openai_api_key = _get_openai_api_key()
    if not openai_api_key:
        logger.warning("No OpenAI/Azure OpenAI-compatible API key configured")
        return None
    openai_model = _get_openai_model()
    openai_base_url = _get_openai_base_url()
    chat_url = _get_openai_chat_url()
    headers = _build_openai_headers(openai_api_key, openai_base_url)
    is_azure = _is_azure_openai_compatible_base_url(openai_base_url)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                chat_url,
                headers=headers,
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
            if (
                response.status_code in {400, 404}
                and openai_model != "gpt-4o-mini"
                and not is_azure
            ):
                fallback_response = await client.post(
                    chat_url,
                    headers=headers,
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
