"""Thin wrappers around OpenAI/Azure chat APIs."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HTTPX client — reused across all LLM calls to avoid per-request
# connection setup overhead.  Call close_shared_client() on shutdown.
# ---------------------------------------------------------------------------
_shared_client: Optional[httpx.AsyncClient] = None

# ---------------------------------------------------------------------------
# Runtime model override — allows switching LLM provider on-the-fly via API.
# Each entry defines: base_url, api_key, model, is_azure flag.
# ---------------------------------------------------------------------------
_MODEL_PROFILES: Dict[str, Dict[str, str]] = {}
_active_profile: Optional[str] = None  # None = use default env-based config


def register_model_profile(
    profile_id: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    label: str = "",
) -> None:
    _MODEL_PROFILES[profile_id] = {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model": model,
        "label": label or profile_id,
    }


def set_active_profile(profile_id: Optional[str]) -> None:
    global _active_profile
    if profile_id is not None and profile_id not in _MODEL_PROFILES:
        raise ValueError(f"Unknown model profile: {profile_id}")
    _active_profile = profile_id


def get_active_profile() -> Optional[str]:
    return _active_profile


def list_model_profiles() -> Dict[str, Dict[str, str]]:
    """Return profiles with safe metadata (no api keys)."""
    result: Dict[str, Dict[str, str]] = {"default": {"label": "Default (env)", "model": _get_openai_model()}}
    for pid, cfg in _MODEL_PROFILES.items():
        result[pid] = {"label": cfg["label"], "model": cfg["model"]}
    return result


def _get_shared_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _shared_client


async def close_shared_client() -> None:
    """Shutdown hook — call from server lifespan/shutdown."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


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
    return bool(_get_openai_api_key())


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


def _extract_content(response: httpx.Response) -> Optional[str]:
    """Extract content string from a 200 chat completions response."""
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        logger.warning("LLM API returned 200 with empty choices: %s", response.text[:300])
        return None
    content = (choices[0].get("message") or {}).get("content")
    if content is None:
        finish_reason = choices[0].get("finish_reason", "unknown")
        logger.warning("LLM API returned null content (finish_reason=%s): %s", finish_reason, response.text[:300])
    return content


async def call_openai_api(
    messages: List[Dict[str, str]],
    system: str,
    max_tokens: int = 1024,
) -> Optional[str]:
    # Use runtime profile override if active, otherwise fall back to env config
    profile = _MODEL_PROFILES.get(_active_profile or "") if _active_profile else None
    if profile:
        openai_api_key = profile["api_key"]
        openai_model = profile["model"]
        openai_base_url = _normalize_openai_base_url(profile["base_url"])
        chat_url = f"{openai_base_url.rstrip('/')}/chat/completions"
        headers = _build_openai_headers(openai_api_key, openai_base_url)
        is_azure = _is_azure_openai_compatible_base_url(openai_base_url)
    else:
        openai_api_key = _get_openai_api_key()
        if not openai_api_key:
            logger.warning("No OpenAI/Azure OpenAI-compatible API key configured")
            return None
        openai_model = _get_openai_model()
        openai_base_url = _get_openai_base_url()
        chat_url = _get_openai_chat_url()
        headers = _build_openai_headers(openai_api_key, openai_base_url)
        is_azure = _is_azure_openai_compatible_base_url(openai_base_url)
    payload = {
        "model": openai_model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, *messages],
    }
    try:
        client = _get_shared_client()
        for attempt in range(3):
            response = await client.post(chat_url, headers=headers, json=payload)
            if response.status_code == 200:
                content = _extract_content(response)
                if content:
                    return content
                # Empty/null content on 200 — often Azure rate-limiting in disguise
                logger.warning("LLM API returned empty content on 200; retrying after 15s (attempt %d/3)", attempt + 1)
                await asyncio.sleep(15)
                continue
            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 15))
                logger.warning("LLM API rate limited (429); retrying after %ds (attempt %d)", retry_after, attempt + 1)
                await asyncio.sleep(retry_after)
                continue
            # Common setup issue: unsupported or unavailable model name.
            if response.status_code in {400, 404} and openai_model != "gpt-4o-mini" and not is_azure:
                fallback = await client.post(
                    chat_url,
                    headers=headers,
                    json={**payload, "model": "gpt-4o-mini"},
                )
                if fallback.status_code == 200:
                    return _extract_content(fallback)
            logger.error("LLM API error: %s - %s", response.status_code, response.text[:300])
            return None
        logger.error("LLM API rate limit / empty content not resolved after 3 retries")
        return None
    except Exception as exc:
        logger.exception("OpenAI API call failed: %s", exc)
        return None
