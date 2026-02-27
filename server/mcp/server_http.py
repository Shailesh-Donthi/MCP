"""
Enterprise MCP HTTP/SSE Server

Production-ready MCP server with:
- HTTP/SSE transport for remote access
- Optional JWT-based user context
- Rate limiting
- Caching (Redis)
- Audit logging
- Metrics (Prometheus)
- Health checks

Usage:
    python -m app.mcp.server_http
    uvicorn app.mcp.server_http:app --host 0.0.0.0 --port 8090
"""

import asyncio
import copy
import hashlib
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis

from mcp.core.config import settings
from mcp.core.database import connect_to_mongodb, close_mongodb_connection, get_database
from mcp.core.error_catalog import build_error_payload, normalize_error_code
from mcp.core.logging_config import configure_logging, log_structured
from mcp.core.security import decode_access_token
from mcp.config import mcp_settings
from mcp.schemas.context_schema import UserContext
from mcp.handlers.tool_handler import get_tool_handler
from mcp.utils.formatters import generate_natural_language_response
from mcp.utils.output_layer import build_output_payload

configure_logging()
logger = logging.getLogger(__name__)

# Redis client (optional - gracefully degrades if not available)
redis_client: Optional[redis.Redis] = None

# Metrics storage (in production, use prometheus_client)
metrics = {
    "requests_total": 0,
    "requests_by_tool": {},
    "cache_hits": 0,
    "cache_misses": 0,
    "errors_total": 0,
}


# =============================================================================
# Pydantic Models
# =============================================================================

class ToolExecuteRequest(BaseModel):
    """Request to execute a tool"""
    arguments: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Natural language query request"""
    query: str = Field(..., description="Natural language query")
    tool_hint: Optional[str] = Field(None, description="Hint for which tool to use")
    output_format: Optional[str] = Field(
        "auto",
        description="Preferred response format: auto, text, json, tree",
    )
    allow_download: Optional[bool] = Field(
        None,
        description="If true, include downloadable response content",
    )
    session_id: Optional[str] = Field(
        None,
        description="Client conversation/session id for context memory",
    )
    chat_id: Optional[str] = Field(
        None,
        description="Client chat thread id for server-side chat memory",
    )


class ChatMessageRequest(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")
    rawHtml: bool = Field(False, description="Whether content contains trusted HTML")
    tone: str = Field("normal", description="Message tone: normal or error")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")


class ChatCreateRequest(BaseModel):
    chat_id: Optional[str] = Field(None, description="Optional client-provided chat id")
    session_id: Optional[str] = Field(None, description="Optional client-provided session id")


class ChatThreadRequest(BaseModel):
    active_chat_id: Optional[str] = Field(None, description="Client active chat id")


# Lightweight in-memory response history for /api/v1/query follow-ups.
# Bounded to avoid unbounded memory growth from arbitrary session ids.
_MAX_QUERY_HISTORY_SESSIONS = 500
_query_history: "OrderedDict[str, str]" = OrderedDict()

# Server-side chat memory (UI chat threads/messages). In production, move this
# to Redis/DB for persistence across restarts and multi-instance deployments.
_MAX_CHAT_OWNERS = 500
_MAX_CHATS_PER_OWNER = 100
_MAX_MESSAGES_PER_CHAT = 500
_chat_threads_by_owner: "OrderedDict[str, OrderedDict[str, Dict[str, Any]]]" = OrderedDict()


def _query_history_get(session_key: str) -> str:
    if session_key in _query_history:
        _query_history.move_to_end(session_key)
        return _query_history[session_key]
    return ""


def _query_history_set(session_key: str, response_text: str) -> None:
    if not session_key:
        return
    _query_history[session_key] = response_text
    _query_history.move_to_end(session_key)
    while len(_query_history) > _MAX_QUERY_HISTORY_SESSIONS:
        _query_history.popitem(last=False)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _normalize_chat_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    role = "user" if str(payload.get("role", "")).lower() == "user" else "assistant"
    tone = "error" if str(payload.get("tone", "")).lower() == "error" else "normal"
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        timestamp = _now_iso()
    return {
        "role": role,
        "content": str(payload.get("content") or ""),
        "rawHtml": bool(payload.get("rawHtml")),
        "tone": tone,
        "timestamp": timestamp,
    }


def _chat_owner_key(http_request: Request, context: UserContext) -> str:
    if getattr(context, "user_id", None):
        return f"user:{context.user_id}"
    client_id = http_request.headers.get("X-Chat-Client-ID")
    if client_id:
        return f"anon-client:{client_id.strip()[:128]}"
    client_host = getattr(getattr(http_request, "client", None), "host", None) or "unknown"
    return f"anon-ip:{client_host}"


def _ensure_owner_chat_bucket(owner_key: str) -> "OrderedDict[str, Dict[str, Any]]":
    bucket = _chat_threads_by_owner.get(owner_key)
    if bucket is None:
        bucket = OrderedDict()
        _chat_threads_by_owner[owner_key] = bucket
    _chat_threads_by_owner.move_to_end(owner_key)
    while len(_chat_threads_by_owner) > _MAX_CHAT_OWNERS:
        _chat_threads_by_owner.popitem(last=False)
    return bucket


def _create_chat_thread_record(*, chat_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "id": (chat_id or f"chat-{uuid4().hex}"),
        "sessionId": (session_id or f"session-{uuid4().hex}"),
        "createdAt": now,
        "updatedAt": now,
        "messages": [],
        "lastAssistantResult": None,
    }


def _get_chat_thread(owner_key: str, chat_id: str) -> Optional[Dict[str, Any]]:
    bucket = _ensure_owner_chat_bucket(owner_key)
    thread = bucket.get(chat_id)
    if thread is not None:
        bucket.move_to_end(chat_id)
    return thread


def _upsert_chat_thread(
    owner_key: str,
    *,
    chat_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    bucket = _ensure_owner_chat_bucket(owner_key)
    thread = bucket.get(chat_id) if chat_id else None
    if thread is None:
        thread = _create_chat_thread_record(chat_id=chat_id, session_id=session_id)
        bucket[thread["id"]] = thread
    if session_id and not thread.get("sessionId"):
        thread["sessionId"] = session_id
    elif session_id:
        thread["sessionId"] = session_id
    thread["updatedAt"] = thread.get("updatedAt") or _now_iso()
    bucket.move_to_end(thread["id"])
    while len(bucket) > _MAX_CHATS_PER_OWNER:
        bucket.popitem(last=False)
    return thread


def _append_chat_message(owner_key: str, chat_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
    thread = _upsert_chat_thread(owner_key, chat_id=chat_id)
    normalized = _normalize_chat_message(message)
    thread.setdefault("messages", []).append(normalized)
    if len(thread["messages"]) > _MAX_MESSAGES_PER_CHAT:
        thread["messages"] = thread["messages"][-_MAX_MESSAGES_PER_CHAT:]
    thread["updatedAt"] = normalized["timestamp"]
    return thread


def _set_chat_last_assistant_result(owner_key: str, chat_id: str, api_result: Any) -> Optional[Dict[str, Any]]:
    thread = _get_chat_thread(owner_key, chat_id)
    if thread is None:
        return None
    thread["lastAssistantResult"] = copy.deepcopy(api_result)
    thread["updatedAt"] = _now_iso()
    return thread


def _list_chat_threads(owner_key: str) -> List[Dict[str, Any]]:
    bucket = _ensure_owner_chat_bucket(owner_key)
    threads = list(bucket.values())
    threads.sort(key=lambda t: t.get("updatedAt") or t.get("createdAt") or "", reverse=True)
    return [copy.deepcopy(thread) for thread in threads]


def _delete_chat_thread(owner_key: str, chat_id: str) -> bool:
    bucket = _ensure_owner_chat_bucket(owner_key)
    if chat_id in bucket:
        del bucket[chat_id]
        return True
    return False


def _clear_chat_threads(owner_key: str) -> None:
    bucket = _ensure_owner_chat_bucket(owner_key)
    bucket.clear()


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    mongodb: str
    redis: str
    tools_loaded: int


def _truncate_for_log(value: Any, max_length: int = 600) -> Optional[str]:
    """Truncate large values before logging/persisting session events."""
    if value is None:
        return None
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "...[truncated]"


def _safe_json_for_log(payload: Optional[Dict[str, Any]], max_length: int = 1200) -> Optional[Dict[str, Any]]:
    """Best-effort payload sanitization for logs."""
    if not payload:
        return None
    try:
        serialized = json.dumps(payload, default=str)
        if len(serialized) <= max_length:
            return payload
        return {"_truncated_json": serialized[:max_length] + "...[truncated]"}
    except Exception:
        return {"_unserializable": _truncate_for_log(payload, max_length)}


def _build_error_detail(
    message: str,
    *,
    code: str,
    request_id: Optional[str],
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build standardized API error detail with user-readable code and guidance."""
    normalized_code = normalize_error_code(code)
    return build_error_payload(
        normalized_code,
        message=message,
        details=details,
        request_id=request_id,
        legacy_code=code if normalized_code != code else None,
    )


def _build_capability_help_response_text() -> str:
    """Human-readable help text for capability/report prompts."""
    return (
        "I can help with police personnel and unit reporting queries. Try asking:\n\n"
        "- Personnel search: 'What is the mobile number of A Ashok Kumar?'\n"
        "- Personnel by rank/district: 'List all SIs in Chittoor district'\n"
        "- Personnel distribution: 'How many personnel are in Guntur district?'\n"
        "- Unit hierarchy: 'What is the unit hierarchy for Chittoor district?'\n"
        "- Units in district: 'List units in Guntur district'\n"
        "- Village mapping: 'Which villages are mapped to K V Palli PS?'\n"
        "- Transfers: 'Show recent transfers in the last 30 days'\n"
        "- Command/in-charge fallback: 'Who is the SDPO of Kuppam?'\n\n"
        "You can also ask follow-ups like 'next page', 'previous page', or 'show details of the 1st one' after list results."
    )


async def log_session_event(
    *,
    event: str,
    endpoint: str,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    query: Optional[str] = None,
    tool_name: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    success: Optional[bool] = None,
    duration_ms: Optional[int] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Structured session logging for chat/query flows."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "endpoint": endpoint,
        "request_id": request_id,
        "session_id": session_id or "default",
        "user_id": user_id,
        "query": _truncate_for_log(query),
        "tool": tool_name,
        "arguments": _safe_json_for_log(arguments),
        "success": success,
        "duration_ms": duration_ms,
        "error": _truncate_for_log(error),
        "metadata": _safe_json_for_log(metadata),
    }

    log_structured(logger, "info", "session_event", **entry)

    # Best-effort persistence; logging must never fail the request path.
    try:
        db = get_database()
        if db:
            await db["mcp_session_logs"].insert_one(entry)
    except Exception as e:
        log_structured(logger, "warning", "session_log_persist_failed", error=str(e))


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global redis_client

    # Startup
    log_structured(
        logger,
        "info",
        "server_starting",
        server_name=mcp_settings.MCP_SERVER_NAME,
        version=mcp_settings.MCP_SERVER_VERSION,
    )

    # Connect to MongoDB
    await connect_to_mongodb()
    log_structured(logger, "info", "mongodb_connected")

    # Initialize tool handler
    handler = get_tool_handler()
    handler.initialize()
    log_structured(
        logger,
        "info",
        "tools_initialized",
        count=len(handler.get_tool_names()),
    )

    # Connect to Redis (optional)
    redis_url = getattr(settings, 'REDIS_URL', None)
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            await redis_client.ping()
            log_structured(logger, "info", "redis_connected")
        except Exception as e:
            log_structured(
                logger,
                "warning",
                "redis_unavailable",
                error=str(e),
                cache_enabled=False,
            )
            redis_client = None
    else:
        log_structured(logger, "info", "redis_disabled", reason="REDIS_URL not configured")

    yield

    # Shutdown
    log_structured(logger, "info", "server_shutdown")
    await close_mongodb_connection()
    if redis_client:
        await redis_client.close()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="MCP Reporting Server",
    description="Enterprise MCP server for natural language reporting queries",
    version=mcp_settings.MCP_SERVER_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(",") if settings.ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    Attach request ID and emit request-level logs for correlation.
    """
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    started_at = time.time()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.time() - started_at) * 1000)
        metrics["errors_total"] += 1
        logger.exception("Unhandled request exception")
        log_structured(
            logger,
            "error",
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
        )
        raise

    duration_ms = int((time.time() - started_at) * 1000)
    response.headers["X-Request-ID"] = request_id
    log_structured(
        logger,
        "info",
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


# =============================================================================
# Dependencies
# =============================================================================

async def get_current_user_context(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> UserContext:
    """
    Extract user context from JWT token.
    If no Authorization header is provided, default to anonymous state-level
    context (scope can be overridden via MCP_ANON_SCOPE_LEVEL).
    """
    if not authorization:
        anon_scope = str(getattr(settings, "MCP_ANON_SCOPE_LEVEL", "state") or "state").lower()
        if anon_scope not in {"unit", "district", "state"}:
            anon_scope = "state"
        return UserContext(scope_level=anon_scope)

    try:
        # Extract token from "Bearer <token>"
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail=_build_error_detail(
                    "Invalid auth scheme",
                    code="MCP-AUTH-001",
                    request_id=getattr(request.state, "request_id", None),
                ),
            )

        # Decode JWT
        payload = decode_access_token(token)
        if not payload:
            raise HTTPException(
                status_code=401,
                detail=_build_error_detail(
                    "Invalid token",
                    code="MCP-AUTH-002",
                    request_id=getattr(request.state, "request_id", None),
                ),
            )

        # Build user context from token
        return UserContext(
            user_id=payload.get("id") or payload.get("sub"),
            unit_id=payload.get("unitId"),
            unit_ids=[u.get("unitId") for u in payload.get("units", []) if u.get("unitId")],
            district_id=payload.get("districtId"),
            district_ids=[payload.get("districtId")] if payload.get("districtId") else [],
            role_id=payload.get("roleId"),
            scope_level=_determine_scope_level(payload),
        )
    except HTTPException:
        raise
    except Exception as e:
        log_structured(
            logger,
            "warning",
            "auth_failed",
            error=str(e),
            request_id=getattr(request.state, "request_id", None),
        )
        raise HTTPException(
            status_code=401,
            detail=_build_error_detail(
                "Authentication failed",
                code="MCP-AUTH-003",
                request_id=getattr(request.state, "request_id", None),
            ),
        )


def _determine_scope_level(payload: dict) -> str:
    """Determine scope level from JWT payload"""
    # Check for state-level roles (customize based on your role system)
    role_name = payload.get("roleName", "").lower()
    if role_name in ("super admin", "state admin", "dgp"):
        return "state"
    elif role_name in ("sp", "district admin", "range ig"):
        return "district"
    else:
        return "unit"


async def check_rate_limit(
    request: Request,
    context: UserContext = Depends(get_current_user_context),
) -> None:
    """Check rate limit for user"""
    if not redis_client:
        return  # Skip if Redis not available

    user_id = context.user_id or request.client.host
    key = f"mcp:rate:{user_id}"

    try:
        current = await redis_client.incr(key)
        if current == 1:
            await redis_client.expire(key, 60)  # 1 minute window

        limit = 60  # requests per minute
        if current > limit:
            raise HTTPException(
                status_code=429,
                detail=_build_error_detail(
                    f"Rate limit exceeded. Max {limit} requests per minute.",
                    code="MCP-RATE-001",
                    request_id=getattr(request.state, "request_id", None),
                    details={"limit_per_minute": limit},
                ),
            )
    except HTTPException:
        raise
    except Exception as e:
        log_structured(
            logger,
            "warning",
            "rate_limit_check_failed",
            error=str(e),
            user_id=user_id,
            request_id=getattr(request.state, "request_id", None),
        )


# =============================================================================
# Caching Utilities
# =============================================================================

def generate_cache_key(tool_name: str, context: UserContext, args: dict) -> str:
    """Generate cache key for query results"""
    scope_hash = hashlib.md5(
        f"{context.scope_level}:{sorted(context.unit_ids)}:{sorted(context.district_ids)}".encode()
    ).hexdigest()[:8]

    args_hash = hashlib.md5(
        json.dumps(args, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]

    return f"mcp:result:{tool_name}:{scope_hash}:{args_hash}"


async def get_cached_result(cache_key: str) -> Optional[dict]:
    """Get cached result from Redis"""
    if not redis_client:
        return None
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            metrics["cache_hits"] += 1
            return json.loads(cached)
        metrics["cache_misses"] += 1
    except Exception as e:
        log_structured(logger, "warning", "cache_get_failed", error=str(e))
    return None


async def set_cached_result(cache_key: str, result: dict, ttl: int = 300) -> None:
    """Cache result in Redis"""
    if not redis_client:
        return
    try:
        await redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
    except Exception as e:
        log_structured(logger, "warning", "cache_set_failed", error=str(e))


# =============================================================================
# Audit Logging
# =============================================================================

async def log_audit(
    user_id: Optional[str],
    tool_name: str,
    arguments: dict,
    success: bool,
    duration_ms: int,
    result_count: int = 0,
    error: Optional[str] = None,
) -> None:
    """Log audit entry"""
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "tool": tool_name,
        "arguments": arguments,
        "success": success,
        "duration_ms": duration_ms,
        "result_count": result_count,
        "error": error,
    }

    # Log to structured logger
    log_structured(logger, "info", "audit_event", **audit_entry)

    # Optionally store in MongoDB audit collection
    try:
        db = get_database()
        if db:
            await db["mcp_audit_logs"].insert_one(audit_entry)
    except Exception as e:
        log_structured(logger, "warning", "audit_log_persist_failed", error=str(e))


# =============================================================================
# Health & Metrics Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    handler = get_tool_handler()

    # Check MongoDB
    try:
        db = get_database()
        await db.command("ping")
        mongodb_status = "healthy"
    except Exception:
        mongodb_status = "unhealthy"

    # Check Redis
    redis_status = "disabled"
    if redis_client:
        try:
            await redis_client.ping()
            redis_status = "healthy"
        except Exception:
            redis_status = "unhealthy"

    return HealthResponse(
        status="healthy" if mongodb_status == "healthy" else "degraded",
        version=mcp_settings.MCP_SERVER_VERSION,
        mongodb=mongodb_status,
        redis=redis_status,
        tools_loaded=len(handler.get_tool_names()),
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        db = get_database()
        await db.command("ping")
        return {"status": "ready"}
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=_build_error_detail(
                "Not ready",
                code="MCP-READY-001",
                request_id=None,
            ),
        )


@app.get("/api/v1/db-test", tags=["Health"])
async def test_database_permissions(
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
):
    """
    Test database permissions on all collections used by MCP tools.
    Helps diagnose MongoDB Atlas permission issues.
    """
    if not getattr(settings, "MCP_ENABLE_DB_TEST_ENDPOINT", False):
        raise HTTPException(
            status_code=404,
            detail=_build_error_detail(
                "Endpoint not available",
                code="MCP-ENDPOINT-404",
                request_id=getattr(http_request.state, "request_id", None),
            ),
        )

    if getattr(context, "scope_level", None) != "state":
        raise HTTPException(
            status_code=403,
            detail=_build_error_detail(
                "Insufficient scope for database diagnostics",
                code="MCP-AUTH-004",
                request_id=getattr(http_request.state, "request_id", None),
            ),
        )

    from mcp.constants import Collections
    import re

    db = get_database()
    if not db:
        return {"status": "error", "message": "Database not connected"}

    # Get connection info (sanitized)
    uri = settings.MONGODB_URI
    # Extract host from URI (hide password)
    host_match = re.search(r'@([^/]+)', uri)
    host = host_match.group(1) if host_match else "unknown"
    # Extract username from URI
    user_match = re.search(r'://([^:]+):', uri)
    username = user_match.group(1) if user_match else "unknown"

    # Collections used by MCP tools
    test_collections = [
        ("personnel_master", Collections.PERSONNEL_MASTER),
        ("unit_master", Collections.UNIT),
        ("district_master", Collections.DISTRICT),
        ("rank_master", Collections.RANK_MASTER),
        ("unit_villages_master", Collections.UNIT_VILLAGES),
        ("unit_type_master", Collections.UNIT_TYPE),
        ("mandal_master", Collections.MANDAL),
    ]

    results = {}
    all_ok = True
    failed_collections = []

    for name, collection_name in test_collections:
        try:
            # Try to read one document
            doc = await db[collection_name].find_one({}, {"_id": 1})
            results[name] = {
                "status": "OK",
                "collection": collection_name,
                "can_read": True,
                "sample_exists": doc is not None
            }
        except Exception as e:
            all_ok = False
            error_msg = str(e)
            failed_collections.append(collection_name)
            results[name] = {
                "status": "FAILED",
                "collection": collection_name,
                "can_read": False,
                "error": error_msg[:300]
            }

    return {
        "status": "OK" if all_ok else "PERMISSION_ERROR",
        "connection": {
            "database": db.name,
            "host": host,
            "username": username,
        },
        "collections": results,
        "failed_collections": failed_collections,
        "recommendation": None if all_ok else (
            f"MongoDB user '{username}' doesn't have read access to: {', '.join(failed_collections)}. "
            f"Fix in MongoDB Atlas:\n"
            f"1. Go to Database Access\n"
            f"2. Find user '{username}'\n"
            f"3. Click Edit\n"
            f"4. Add role: 'readWrite' on database '{db.name}'\n"
            f"5. Or add role: 'readWriteAnyDatabase' for full access"
        )
    }


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus-style metrics endpoint"""
    handler = get_tool_handler()

    output = []
    output.append(f"# HELP mcp_requests_total Total MCP requests")
    output.append(f"# TYPE mcp_requests_total counter")
    output.append(f'mcp_requests_total {metrics["requests_total"]}')

    output.append(f"# HELP mcp_cache_hits_total Cache hits")
    output.append(f"# TYPE mcp_cache_hits_total counter")
    output.append(f'mcp_cache_hits_total {metrics["cache_hits"]}')

    output.append(f"# HELP mcp_cache_misses_total Cache misses")
    output.append(f"# TYPE mcp_cache_misses_total counter")
    output.append(f'mcp_cache_misses_total {metrics["cache_misses"]}')

    output.append(f"# HELP mcp_tools_loaded Number of tools loaded")
    output.append(f"# TYPE mcp_tools_loaded gauge")
    output.append(f"mcp_tools_loaded {len(handler.get_tool_names())}")

    return "\n".join(output)


# =============================================================================
# MCP Tool Endpoints
# =============================================================================

@app.get("/api/v1/mcp/tools", tags=["MCP Tools"])
async def list_tools(
    _: None = Depends(check_rate_limit),
):
    """List all available MCP tools"""
    handler = get_tool_handler()
    return {
        "success": True,
        "tools": handler.get_tools(),
        "count": len(handler.get_tool_names()),
    }


@app.get("/api/v1/mcp/tools/{tool_name}", tags=["MCP Tools"])
async def get_tool_schema(
    tool_name: str,
    http_request: Request,
    _: None = Depends(check_rate_limit),
):
    """Get schema for a specific tool"""
    handler = get_tool_handler()
    tools = {t["name"]: t for t in handler.get_tools()}

    if tool_name not in tools:
        raise HTTPException(
            status_code=404,
            detail=_build_error_detail(
                f"Tool not found: {tool_name}",
                code="MCP-TOOL-001",
                request_id=getattr(http_request.state, "request_id", None),
                details={"tool": tool_name},
            ),
        )

    return {"success": True, "tool": tools[tool_name]}


@app.post("/api/v1/mcp/tools/{tool_name}/execute", tags=["MCP Tools"])
async def execute_tool(
    tool_name: str,
    request: ToolExecuteRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
):
    """Execute an MCP tool"""
    handler = get_tool_handler()
    start_time = time.time()

    # Update metrics
    metrics["requests_total"] += 1
    metrics["requests_by_tool"][tool_name] = metrics["requests_by_tool"].get(tool_name, 0) + 1

    # Check cache
    cache_key = generate_cache_key(tool_name, context, request.arguments)
    cached = await get_cached_result(cache_key)
    if cached:
        cached["metadata"] = cached.get("metadata", {})
        cached["metadata"]["cached"] = True
        return cached

    # Execute tool
    try:
        result = await handler.execute(tool_name, request.arguments, context)

        duration_ms = int((time.time() - start_time) * 1000)

        # Add metadata
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["execution_time_ms"] = duration_ms
        result["metadata"]["cached"] = False

        # Cache successful results
        if result.get("success", False):
            await set_cached_result(cache_key, result)

        # Audit log
        result_count = 0
        if isinstance(result.get("data"), list):
            result_count = len(result["data"])
        elif isinstance(result.get("data"), dict):
            result_count = result.get("pagination", {}).get("total", 1)

        await log_audit(
            user_id=context.user_id,
            tool_name=tool_name,
            arguments=request.arguments,
            success=result.get("success", False),
            duration_ms=duration_ms,
            result_count=result_count,
        )
        await log_session_event(
            event="tool_execute_completed",
            endpoint="/api/v1/mcp/tools/{tool_name}/execute",
            request_id=getattr(http_request.state, "request_id", None),
            session_id=None,
            user_id=context.user_id,
            query=None,
            tool_name=tool_name,
            arguments=request.arguments,
            success=result.get("success", False),
            duration_ms=duration_ms,
            metadata={"result_count": result_count},
        )

        return result

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        metrics["errors_total"] += 1

        await log_audit(
            user_id=context.user_id,
            tool_name=tool_name,
            arguments=request.arguments,
            success=False,
            duration_ms=duration_ms,
            error=str(e),
        )
        await log_session_event(
            event="tool_execute_failed",
            endpoint="/api/v1/mcp/tools/{tool_name}/execute",
            request_id=getattr(http_request.state, "request_id", None),
            session_id=None,
            user_id=context.user_id,
            query=None,
            tool_name=tool_name,
            arguments=request.arguments,
            success=False,
            duration_ms=duration_ms,
            error=str(e),
        )

        logger.exception(f"Tool execution error: {e}")
        log_structured(
            logger,
            "error",
            "tool_execute_failed",
            tool_name=tool_name,
            request_id=getattr(http_request.state, "request_id", None),
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=_build_error_detail(
                "Tool execution failed",
                code="MCP-TOOL-002",
                request_id=getattr(http_request.state, "request_id", None),
                details={"tool": tool_name},
            ),
        )


# =============================================================================
# Chat Thread Endpoints (Server-Side UI Memory)
# =============================================================================

@app.get("/api/v1/chats", tags=["Chat"])
async def list_chat_threads(
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
):
    owner_key = _chat_owner_key(http_request, context)
    threads = _list_chat_threads(owner_key)
    return {
        "success": True,
        "chats": threads,
        "count": len(threads),
    }


@app.post("/api/v1/chats", tags=["Chat"])
async def create_chat_thread(
    request: ChatCreateRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
):
    owner_key = _chat_owner_key(http_request, context)
    thread = _upsert_chat_thread(
        owner_key,
        chat_id=request.chat_id,
        session_id=request.session_id,
    )
    return {
        "success": True,
        "chat": copy.deepcopy(thread),
    }


@app.delete("/api/v1/chats", tags=["Chat"])
async def clear_chat_threads(
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
    request: Optional[ChatThreadRequest] = None,
):
    owner_key = _chat_owner_key(http_request, context)
    _clear_chat_threads(owner_key)
    fresh = _upsert_chat_thread(owner_key)
    return {
        "success": True,
        "active_chat_id": fresh["id"],
        "chats": [copy.deepcopy(fresh)],
    }


@app.get("/api/v1/chats/{chat_id}", tags=["Chat"])
async def get_chat_thread(
    chat_id: str,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
):
    owner_key = _chat_owner_key(http_request, context)
    thread = _get_chat_thread(owner_key, chat_id)
    if thread is None:
        raise HTTPException(
            status_code=404,
            detail=_build_error_detail(
                f"Chat not found: {chat_id}",
                code="MCP-CHAT-404",
                request_id=getattr(http_request.state, "request_id", None),
                details={"chat_id": chat_id},
            ),
        )
    return {"success": True, "chat": copy.deepcopy(thread)}


@app.delete("/api/v1/chats/{chat_id}", tags=["Chat"])
async def delete_chat_thread(
    chat_id: str,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
):
    owner_key = _chat_owner_key(http_request, context)
    deleted = _delete_chat_thread(owner_key, chat_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=_build_error_detail(
                f"Chat not found: {chat_id}",
                code="MCP-CHAT-404",
                request_id=getattr(http_request.state, "request_id", None),
                details={"chat_id": chat_id},
            ),
        )
    remaining = _list_chat_threads(owner_key)
    if not remaining:
        fresh = _upsert_chat_thread(owner_key)
        remaining = [copy.deepcopy(fresh)]
    return {
        "success": True,
        "chats": remaining,
        "active_chat_id": remaining[0]["id"] if remaining else None,
    }


@app.post("/api/v1/chats/{chat_id}/messages", tags=["Chat"])
async def append_chat_message(
    chat_id: str,
    request: ChatMessageRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
):
    owner_key = _chat_owner_key(http_request, context)
    thread = _append_chat_message(owner_key, chat_id, request.model_dump())
    return {"success": True, "chat": copy.deepcopy(thread)}


# =============================================================================
# Natural Language Query Endpoint
# =============================================================================

# Tool keyword mapping for simple NL routing
TOOL_KEYWORDS = {
    "list_districts": ["districts", "available districts", "district list", "which districts", "all districts"],
    "check_responsible_user": ["responsible user", "responsibleuser", "sho", "in charge", "in-charge", "incharge", "heads", "heading", "head of", "officer in charge", "station house officer"],
    "search_personnel": ["find", "search", "who", "person", "belongs", "which unit", "where is", "locate", "date of birth", "dob", "born", "birthday", "age", "contact", "mobile", "phone", "email", "details of", "info about", "information about"],
    "search_unit": ["find unit", "search station", "where is station", "locate unit"],
    "query_personnel_by_unit": ["personnel in", "staff in", "officers in", "show all", "list personnel"],
    "query_personnel_by_rank": ["rank", "si", "inspector", "constable", "designation", "by rank"],
    "get_unit_hierarchy": ["hierarchy", "heirarchy", "structure", "tree", "organization", "parent", "child"],
    "list_units_in_district": ["units", "stations", "district", "list units"],
    "count_vacancies_by_unit_rank": ["vacancy", "vacancies", "shortage", "strength", "how many"],
    "get_personnel_distribution": ["distribution", "statistics", "breakdown", "count"],
    "query_recent_transfers": ["transfer", "posting", "movement", "moved", "last days"],
    "get_unit_command_history": ["command", "history", "commander", "sho", "in charge"],
    "find_missing_village_mappings": ["village", "mapping", "missing", "coverage", "unmapped"],
    "get_village_coverage": ["village coverage", "jurisdiction", "area covered"],
}


def route_query_to_tool(query: str, hint: Optional[str] = None) -> tuple[str, dict]:
    """
    Smart routing of natural language queries to appropriate tools.
    Uses priority-based pattern matching to ensure accurate routing.
    """
    import re
    query_lower = query.lower().strip()

    # Use hint if provided
    if hint and hint in TOOL_KEYWORDS:
        return hint, {}

    args = {}

    # Ordinal follow-up parser for queries like "info on 1st".
    def extract_ordinal_index(q):
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", q, re.IGNORECASE)
        if m:
            return int(m.group(1))
        words = {
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        }
        for w, idx in words.items():
            if re.search(rf"\b{w}\b", q, re.IGNORECASE):
                return idx
        return None

    # Helper to extract district name from query
    def extract_district(q):
        patterns = [
            r"(?:in|for|of|at|on)\s+([A-Za-z\s]+?)\s+(?:district|dist\.?)",
            r"([A-Za-z]+)\s+(?:district|dist\.?)",
            r"(?:district|dist\.?)\s+([A-Za-z]+)",
        ]
        for p in patterns:
            m = re.search(p, q, re.IGNORECASE)
            if m:
                name = m.group(1).strip().title()
                # Remove "District"/"Dist" if accidentally included
                name = re.sub(r'\s*(?:District|Dist\.?)\s*$', '', name, flags=re.IGNORECASE).strip()
                return name
        return None

    # Helper to extract unit name from query
    def extract_unit(q):
        # Prefer intent-specific captures first to avoid greedy matches like
        # "Which villages are mapped to K V Palli PS" -> "Which villages are mapped to K V Palli".
        specialized_patterns = [
            r"(?:which\s+villages?.*?(?:mapped|assigned|covered)\s+(?:to|in|for|by)|village\s+(?:coverage|mapping)\s+for)\s+([A-Za-z\s]+?\s+(?:ps|police station|station|sdpo|spdo|dpo|range|circle|ups))(?:\?|$)",
            r"(?:where\s+is|search\s+unit|find\s+unit|locate\s+unit)\s+([A-Za-z\s]+?\s+(?:ps|police station|station|sdpo|spdo|dpo|range|circle|ups))(?:\?|$)",
        ]
        for p in specialized_patterns:
            m = re.search(p, q, re.IGNORECASE)
            if m:
                name = re.sub(r"\s+", " ", m.group(1)).strip()
                name = re.sub(r"\b(the|a|an)\b", "", name, flags=re.IGNORECASE).strip()
                if name and len(name) > 1:
                    return name

        patterns = [
            r"(?:in|for|of|at|under)\s+([A-Za-z\s]+?)\s+(?:unit|station|ps|police station|sdpo|spdo|dpo|range|circle)",
            r"([A-Za-z\s]+?)\s+(?:unit|station|ps|police station|sdpo|spdo|dpo|range|circle)(?:\s|$|\?)",
        ]
        for p in patterns:
            m = re.search(p, q, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                name = re.sub(r'\b(the|a|an|all)\b', '', name, flags=re.IGNORECASE).strip()
                name = re.sub(r"^(?:which\s+villages?.*?(?:mapped|assigned|covered)\s+(?:to|in|for|by)|where\s+is|search|find|locate)\s+", "", name, flags=re.IGNORECASE).strip()
                if name and len(name) > 1:
                    return name
        return None

    def extract_role_unit_candidate(q):
        """Extract unit target for role-holder queries like 'SPDO of Kuppam'."""
        m = re.search(
            r"\b(?:who\s+is|what\s+is\s+the\s+name\s+of|name\s+of)?\s*(?:the\s+)?(sho|sdpo|spdo|in[\s-]?charge)\s+of\s+([A-Za-z\s]+?)(?:\s+district)?(?:\?|$)",
            q,
            re.IGNORECASE,
        )
        if not m:
            return None
        role = m.group(1).strip().lower()
        place = re.sub(r"\s+", " ", m.group(2)).strip()
        place = re.sub(r"\b(the|a|an)\b", "", place, flags=re.IGNORECASE).strip()
        if not place:
            return None
        if role in {"sdpo", "spdo"}:
            # Canonicalize typo 'SPDO' to stored form 'SDPO'
            return f"{place.title()} SDPO"
        if role == "sho":
            return f"{place.title()} PS"
        return place.title()

    def extract_rank(q):
        rank_patterns = [
            (r"\bcircle\s+inspectors?\b", "Circle Inspector"),
            (r"\binspector\s+general\s+of\s+police\b", "Inspector General of Police"),
            (r"\binspector\s+of\s+police\b", "Inspector Of Police"),
            (r"\bsub[-\s]?inspectors?\b", "Sub Inspector"),
            (r"\bassistant\s+sub[-\s]?inspectors?\b", "Assistant SubInspector"),
            (r"\bhead\s+constables?\b", "Head Constable"),
            (r"\bpolice\s+constables?\b", "Police Constable"),
            (r"\bconstables?\b", "Constable"),
            (r"\binspectors?\b", "Inspector"),
            (r"\bsi\b", "Sub Inspector"),
            (r"\bsis\b", "Sub Inspector"),
            (r"\basi\b", "Assistant SubInspector"),
            (r"\bhc\b", "Head Constable"),
            (r"\bpc\b", "Police Constable"),
            (r"\bdsp\b", "Deputy Superintendent of Police"),
            (r"\bdysp\b", "Deputy Superintendent of Police"),
            (r"\bsp\b", "Superintendent of Police"),
        ]
        for pattern, rank in rank_patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return rank
        return None

    # PRIORITY -1: ordinal follow-up (requires previous assistant list text in hint)
    if re.search(r"\b(info|information|details?|contact|email|mobile|phone)\b", query_lower):
        idx = extract_ordinal_index(query)
        if idx and hint:
            m = re.search(rf"^\s*{idx}\.\s*(?:User\s*)?(\d{{6,12}})\b", hint, re.IGNORECASE | re.MULTILINE)
            if m:
                return "search_personnel", {"user_id": m.group(1)}

    # ==========================================================================
    # PRIORITY 0: List districts queries
    # ==========================================================================
    if re.search(
        r"(?:which|what|list|show|get|all)\s+(?:districts?|disctricts?)|(?:districts?|disctricts?)\s+(?:available|in\s+the\s+db|in\s+database)|(?:info|information|details?|about)\s+(?:on\s+)?(?:all\s+)?(?:districts|disctricts)\b",
        query_lower,
    ):
        return "list_districts", args

    # ==========================================================================
    # PRIORITY 0b: Help / capability queries
    # ==========================================================================
    if re.search(
        r"\b(what\s+can\s+you\s+help\s+me\s+with|what\s+reports?\s+can\s+i\s+ask\s+for|what\s+can\s+i\s+ask|help(?:\s+me)?)\b",
        query_lower,
    ):
        return "__help__", {}

    # ==========================================================================
    # PRIORITY 1: Missing village mapping queries
    # ==========================================================================
    if re.search(r"(?:missing|unmapped|uncovered|without)\s+village|village(?:s)?\s+(?:not\s+)?(?:mapped|assigned)|no\s+village", query_lower):
        district = extract_district(query)
        if district:
            args["district_name"] = district
        return "find_missing_village_mappings", args

    # ==========================================================================
    # PRIORITY 2: Village coverage queries (specific unit)
    # ==========================================================================
    if re.search(r"village(?:s)?\s+(?:mapped|assigned|covered|in|for|under)|village\s+(?:coverage|mapping)|which\s+villages?", query_lower):
        unit = extract_unit(query)
        if not unit:
            # Try more patterns
            m = re.search(r"(?:mapped|assigned|covered)\s+(?:to|in|for|by)\s+([A-Za-z\s]+?)(?:\?|$)", query, re.IGNORECASE)
            if m:
                unit = m.group(1).strip()
                unit = re.sub(r'\b(the|a|an|unit|station)\b', '', unit, flags=re.IGNORECASE).strip()
        if unit:
            args["unit_name"] = unit
            return "get_village_coverage", args

    # ==========================================================================
    # PRIORITY 3: Unit hierarchy/structure queries
    # ==========================================================================
    if re.search(r"(personnel|officers?|staff)\s+(?:hierarchy|heirarchy)|(?:hierarchy|heirarchy)\s+of\s+(?:personnel|officers?|staff)", query_lower):
        district = extract_district(query)
        if district:
            args["district_name"] = district
        args["group_by"] = "rank"
        return "get_personnel_distribution", args

    # ==========================================================================
    # PRIORITY 3b: Unit hierarchy/structure queries
    # ==========================================================================
    if re.search(r"hierarchy|heirarchy|structure|tree|organization|parent|child\s+unit|sub\s*unit", query_lower):
        district = extract_district(query)
        unit = extract_unit(query)
        # Also try: "Guntur hierarchy" / "hierarchy of Guntur"
        m = re.search(r"(?:hierarchy|heirarchy|structure|tree)\s+(?:of|for)\s+([A-Za-z\s]+?)(?:\?|$)", query, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Clean up - remove common words and "district"
            name = re.sub(r'\b(the|a|an)\b', '', name, flags=re.IGNORECASE).strip()
            name = re.sub(r'\s*district\s*$', '', name, flags=re.IGNORECASE).strip()
            if "district" in query_lower:
                args["district_name"] = name.title()
            else:
                args["root_unit_name"] = name
        elif district:
            args["district_name"] = district
        elif unit:
            args["root_unit_name"] = unit
        return "get_unit_hierarchy", args

    # ==========================================================================
    # PRIORITY 4: Transfer queries
    # ==========================================================================
    if re.search(r"transfer|posting|movement|moved|posted", query_lower):
        district = extract_district(query)
        if district:
            args["district_name"] = district
        # Extract days
        days_match = re.search(r"(\d+)\s*days?", query_lower)
        if days_match:
            args["days"] = int(days_match.group(1))
        else:
            args["days"] = 30  # Default
        # Check if asking about command history
        if re.search(r"command|history|commander|who\s+(?:was|were)\s+(?:the\s+)?(?:sho|in[\s-]?charge)", query_lower):
            unit = extract_unit(query)
            if unit:
                args["unit_name"] = unit
            return "get_unit_command_history", args
        return "query_recent_transfers", args

    # ==========================================================================
    # PRIORITY 4b: Person attribute/detail queries (before SHO/SPDO regex)
    # ==========================================================================
    person_attr_patterns = [
        r"(?:email|e-?mail|mobile|phone|contact|dob|date\s+of\s+birth|birthday|address|blood\s+group)\s+(?:number\s+)?(?:of|for)\s+([A-Za-z\s]+?)(?:\?|$)",
        r"(?:show|get|give|provide)\s+details?\s+(?:of|about)\s+([A-Za-z\s]+?)(?:\?|$)",
    ]
    for p in person_attr_patterns:
        m = re.search(p, query, re.IGNORECASE)
        if m:
            candidate = re.sub(r"\s+", " ", m.group(1)).strip()
            candidate = re.sub(r"\b(the|an)\b", "", candidate, flags=re.IGNORECASE).strip()
            if candidate and not re.search(r"\b(?:sho|sdpo|spdo|dist(?:rict)?|station|ps|unit)\b", candidate, re.IGNORECASE):
                args["name"] = candidate
                return "search_personnel", args

    # ==========================================================================
    # PRIORITY 5: Responsible user / SHO queries
    # ==========================================================================
    if re.search(r"responsible\s*user|responsibleuser|who\s+(?:is|are)\s+(?:the\s+)?\b(?:sho|sdpo|spdo|in[\s-]?charge|head)\b|\b(?:sho|sdpo|spdo|in[\s-]?charge)\b\s+of|name\s+of\s+(?:the\s+)?\b(?:sho|sdpo|spdo)\b", query_lower):
        # Prefer unit-role interpretation first for queries like "Who is the SPDO of Kuppam?"
        role_unit = extract_role_unit_candidate(query)
        if role_unit:
            args["unit_name"] = role_unit
            return "get_unit_command_history", args

        # Extract person name only for queries that start with "is <name> a responsible user..."
        m = re.search(r"^\s*is\s+([A-Za-z\s]+?)\s+(?:a\s+)?(?:responsible|sho|sdpo|spdo|in[\s-]?charge)\b", query, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'\b(the|a|an)\b', '', name, flags=re.IGNORECASE).strip()
            if name:
                args["name"] = name
                return "check_responsible_user", args
        # Extract unit: "who is the SHO of Guntur PS"
        unit = extract_unit(query)
        district = extract_district(query)
        if not unit and district:
            if re.search(r"\b(?:spdo|sdpo)\b", query_lower):
                unit = f"{district} SDPO"
            elif re.search(r"\bsho\b", query_lower):
                unit = f"{district} PS"
        if unit:
            args["unit_name"] = unit
            return "get_unit_command_history", args
        return "check_responsible_user", args

    # ==========================================================================
    # PRIORITY 6: List units queries
    # ==========================================================================
    if re.search(r"(?:list|show|get|display|all)\s+(?:the\s+)?(?:all\s+)?units?|units?\s+(?:in|under|of)", query_lower):
        district = extract_district(query)
        if district:
            args["district_name"] = district
        return "list_units_in_district", args

    # ==========================================================================
    # PRIORITY 7: Personnel count/distribution queries
    # ==========================================================================
    if re.search(r"how\s+many\s+(?:personnel|personell|officers?|staff|people)|(?:count|total|number)\s+(?:of\s+)?(?:personnel|personell|officers?|staff)|(?:personnel|personell)\s+(?:count|total|strength|distribution|breakdown)|strength\s+(?:of|in)|available\s+ranks?|different\s+ranks?|list\s+(?:all\s+)?ranks?|rank\s+distribution|rank\s+wise|ranks?\s+of\s+all", query_lower):
        district = extract_district(query)
        unit = extract_unit(query)
        if district:
            args["district_name"] = district
        elif unit:
            args["unit_name"] = unit
        args["group_by"] = "rank"
        return "get_personnel_distribution", args

    # ==========================================================================
    # PRIORITY 8: Vacancy queries
    # ==========================================================================
    if re.search(r"vacanc|shortage|unfilled|empty\s+post|sanctioned", query_lower):
        district = extract_district(query)
        unit = extract_unit(query)
        if district:
            args["district_name"] = district
        elif unit:
            args["unit_name"] = unit
        return "count_vacancies_by_unit_rank", args

    # ==========================================================================
    # PRIORITY 9: Personnel in unit queries
    # ==========================================================================
    # Rank-detail queries such as "give details of all SI in Chittoor"
    # should route to rank personnel, not unit listing.
    rank_name = extract_rank(query)
    if rank_name and re.search(r"\b(all|details?|info|list|show|give|who|which)\b", query_lower):
        district = extract_district(query)
        if district:
            args["district_name"] = district
        args["rank_name"] = rank_name
        return "query_personnel_by_rank", args

    if re.search(r"(?:personnel|personell|officers?|staff|people)\s+(?:in|at|under|of)|(?:who|list|show)\s+(?:is|are|all)\s+(?:in|at|the)", query_lower):
        district = extract_district(query)
        unit = extract_unit(query)
        if unit:
            args["unit_name"] = unit
        elif district:
            args["district_name"] = district
        # Check if asking by rank
        rank_name = extract_rank(query)
        if rank_name:
            args["rank_name"] = rank_name
            return "query_personnel_by_rank", args
        # query_personnel_by_unit requires unit_id/unit_name; district-only phrasing
        # should fall back to a district-level personnel overview to avoid contract mismatch.
        if district and not unit:
            return "get_personnel_distribution", {"district_name": district, "group_by": "rank"}
        return "query_personnel_by_unit", args

    # ==========================================================================
    # PRIORITY 10: Personnel by rank queries
    # ==========================================================================
    if re.search(r"(?:all|list|show|get)\s+(?:the\s+)?(?:si|asi|hc|pc|inspector|constable|circle\s+inspector|dsp|sp)s?\b|by\s+rank|\brank\s+wise", query_lower):
        district = extract_district(query)
        if district:
            args["district_name"] = district
        rank_name = extract_rank(query)
        if rank_name:
            args["rank_name"] = rank_name
        return "query_personnel_by_rank", args

    # ==========================================================================
    # PRIORITY 11: Search unit queries
    # ==========================================================================
    if re.search(
        r"(?:find|search|locate)\s+(?:the\s+)?(?:unit|station)|(?:unit|station)\s+(?:named?|called)|where\s+is\s+.+\b(?:ps|police station|station|sdpo|spdo|dpo|range|circle|ups)\b",
        query_lower,
    ):
        unit = None
        # Prefer command-specific parsing for "Search unit X" and "Where is X PS?"
        m_direct = re.search(
            r"(?:find|search|locate)\s+(?:the\s+)?(?:unit|station)\s+([A-Za-z\s]+?)(?:\?|$)",
            query,
            re.IGNORECASE,
        )
        if m_direct:
            unit = m_direct.group(1).strip()
        if not unit:
            m_where = re.search(
                r"where\s+is\s+([A-Za-z\s]+?\s+(?:ps|police station|station|sdpo|spdo|dpo|range|circle|ups))(?:\?|$)",
                query,
                re.IGNORECASE,
            )
            if m_where:
                unit = m_where.group(1).strip()
        if not unit:
            unit = extract_unit(query)
        if not unit:
            m = re.search(r"(?:find|search|locate|where\s+is)\s+(?:the\s+)?([A-Za-z\s]+?)(?:\s+unit|\s+station|\?|$)", query, re.IGNORECASE)
            if m:
                unit = m.group(1).strip()
        if unit:
            unit = re.sub(r"^(?:search|find|locate)\s+", "", unit, flags=re.IGNORECASE).strip()
            args["name"] = unit
        return "search_unit", args

    # ==========================================================================
    # PRIORITY 12: Person-specific queries (MUST HAVE clear person indicators)
    # ==========================================================================
    # Only route to search_personnel if there's a clear indication it's about a person
    person_indicators = [
        r"(?:email|mobile|phone|contact|dob|date\s+of\s+birth|birthday|address|blood\s+group)\s+(?:of|for)\s+([A-Za-z\s]+)",
        r"([A-Za-z\s]+?)(?:'s)\s+(?:email|mobile|phone|dob|birthday|address|rank|designation)",
        r"(?:tell\s+me\s+about|details\s+(?:of|about)|info\s+(?:about|on)|information\s+(?:about|on))\s+(?:person\s+)?([A-Za-z\s]+?)(?:\?|$)",
        r"(?:who\s+is|find\s+person|search\s+person|locate\s+person)\s+([A-Za-z\s]+)",
        r"which\s+unit\s+(?:does|is)\s+([A-Za-z\s]+?)\s+(?:belong|work|in)",
        r"([A-Za-z\s]+?)\s+(?:belongs?\s+to|works?\s+(?:in|at)|is\s+(?:in|at))\s+which",
        r"what\s+is\s+(?:the\s+)?([A-Za-z\s]+?)\s+(?:designation|rank|post|position)",
        r"(?:designation|rank|position)\s+(?:of|for)\s+([A-Za-z\s]+)",
    ]

    for pattern in person_indicators:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Clean up - remove common non-name words
            name = re.sub(r'\b(the|an|is|are|was|were|of|for|in|at|to|person|officer|personnel)\b', '', name, flags=re.IGNORECASE).strip()
            # Validate it looks like a person name (not unit/district related words)
            invalid_words = ["unit", "station", "district", "dist", "ps", "police", "hierarchy", "transfer", "vacancy", "all", "list", "show", "get"]
            if name and len(name) > 1 and not any(w in name.lower() for w in invalid_words):
                args["name"] = name
                return "search_personnel", args

    # ==========================================================================
    # PRIORITY 13: Keyword-based fallback
    # ==========================================================================
    scores = {}
    for tool, keywords in TOOL_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[tool] = score

    if scores:
        best_tool = max(scores, key=scores.get)

        # Extract relevant arguments based on tool
        district = extract_district(query)
        unit = extract_unit(query)

        if district:
            args["district_name"] = district
        if unit and best_tool not in ["search_personnel"]:
            args["unit_name"] = unit

        # Extract days for transfer queries
        if best_tool in ["query_recent_transfers", "get_unit_command_history"]:
            days_match = re.search(r"(\d+)\s*days?", query_lower)
            if days_match:
                args["days"] = int(days_match.group(1))

        return best_tool, args

    # ==========================================================================
    # DEFAULT: Personnel distribution (general overview)
    # ==========================================================================
    district = extract_district(query)
    if district:
        args["district_name"] = district
    args["group_by"] = "rank"
    return "get_personnel_distribution", args


@app.post("/api/v1/query", tags=["Query"])
async def natural_language_query(
    request: QueryRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
):
    """
    Execute a natural language query.

    The system routes the query to the appropriate tool based on keywords.
    Returns both a natural language response and raw data.
    """
    handler = get_tool_handler()
    request_id = getattr(http_request.state, "request_id", None)
    chat_owner = _chat_owner_key(http_request, context)
    chat_thread = _get_chat_thread(chat_owner, request.chat_id) if request.chat_id else None
    resolved_session_id = request.session_id or (chat_thread.get("sessionId") if isinstance(chat_thread, dict) else None)
    session_key = resolved_session_id or "default"
    started_at = time.time()

    await log_session_event(
        event="query_received",
        endpoint="/api/v1/query",
        request_id=request_id,
        session_id=session_key,
        user_id=context.user_id,
        query=request.query,
        metadata={
            "tool_hint": request.tool_hint,
            "output_format": request.output_format,
            "allow_download": request.allow_download,
        },
    )

    try:
        from mcp.router import repair_route, needs_clarification

        # Use last assistant response as hint for ordinal follow-ups.
        history_hint = _query_history_get(session_key)
        route_hint = request.tool_hint or history_hint

        if needs_clarification(request.query):
            nl_response = (
                "Could you clarify what you want to know? "
                "Please include at least one target like a person name/user ID, district, unit, or rank. "
                "Example: 'Circle Inspectors in Chittoor district'."
            )
            output_payload = build_output_payload(
                query=request.query,
                response_text=nl_response,
                routed_to=None,
                arguments={},
                result={},
                requested_format=request.output_format,
                allow_download=request.allow_download,
            )
            _query_history_set(session_key, nl_response)
            duration_ms = int((time.time() - started_at) * 1000)
            await log_session_event(
                event="query_clarification_returned",
                endpoint="/api/v1/query",
                request_id=request_id,
                session_id=session_key,
                user_id=context.user_id,
                query=request.query,
                success=True,
                duration_ms=duration_ms,
            )
            return {
                "success": True,
                "query": request.query,
                "response": nl_response,
                "routed_to": None,
                "extracted_arguments": {},
                "data": {},
                "chat_id": request.chat_id,
                "output": output_payload,
            }

        # Route query to tool
        tool_name, extracted_args = route_query_to_tool(request.query, route_hint)
        tool_name, extracted_args = repair_route(
            query=request.query,
            tool_name=tool_name,
            arguments=extracted_args,
            last_user_query=None,
            last_assistant_response=history_hint,
        )
        await log_session_event(
            event="query_routed",
            endpoint="/api/v1/query",
            request_id=request_id,
            session_id=session_key,
            user_id=context.user_id,
            query=request.query,
            tool_name=tool_name,
            arguments=extracted_args,
            metadata={"routing_mode": "rule_based"},
        )

        if tool_name == "__help__":
            nl_response = _build_capability_help_response_text()
            output_payload = build_output_payload(
                query=request.query,
                response_text=nl_response,
                routed_to=None,
                arguments={},
                result={},
                requested_format=request.output_format,
                allow_download=request.allow_download,
            )
            _query_history_set(session_key, nl_response)
            duration_ms = int((time.time() - started_at) * 1000)
            await log_session_event(
                event="query_help_returned",
                endpoint="/api/v1/query",
                request_id=request_id,
                session_id=session_key,
                user_id=context.user_id,
                query=request.query,
                success=True,
                duration_ms=duration_ms,
            )
            return {
                "success": True,
                "query": request.query,
                "response": nl_response,
                "routed_to": None,
                "extracted_arguments": {},
                "data": {},
                "chat_id": request.chat_id,
                "output": output_payload,
            }

        # Execute tool
        result = await handler.execute(tool_name, extracted_args, context)

        # Generate natural language response
        nl_response = generate_natural_language_response(
            query=request.query,
            tool_name=tool_name,
            arguments=extracted_args,
            result=result,
        )
        output_payload = build_output_payload(
            query=request.query,
            response_text=nl_response,
            routed_to=tool_name,
            arguments=extracted_args,
            result=result,
            requested_format=request.output_format,
            allow_download=request.allow_download,
        )

        # Save assistant response for simple follow-up context.
        _query_history_set(session_key, nl_response)
        if request.chat_id:
            _upsert_chat_thread(chat_owner, chat_id=request.chat_id, session_id=resolved_session_id)
            _append_chat_message(chat_owner, request.chat_id, {"role": "user", "content": request.query})
            _append_chat_message(chat_owner, request.chat_id, {"role": "assistant", "content": nl_response})
            _set_chat_last_assistant_result(chat_owner, request.chat_id, {
                "success": bool(result.get("success")) if isinstance(result, dict) else True,
                "query": request.query,
                "response": nl_response,
                "routed_to": tool_name,
                "extracted_arguments": extracted_args,
                "data": result,
                "output": output_payload,
            })

        duration_ms = int((time.time() - started_at) * 1000)
        await log_session_event(
            event="query_completed",
            endpoint="/api/v1/query",
            request_id=request_id,
            session_id=session_key,
            user_id=context.user_id,
            query=request.query,
            tool_name=tool_name,
            arguments=extracted_args,
            success=bool(result.get("success")) if isinstance(result, dict) else None,
            duration_ms=duration_ms,
            metadata={"response_length": len(nl_response)},
        )

        return {
            "success": bool(result.get("success")) if isinstance(result, dict) else True,
            "query": request.query,
            "response": nl_response,  # Human-readable response
            "routed_to": tool_name,
            "extracted_arguments": extracted_args,
            "data": result,  # Raw data for programmatic use
            "chat_id": request.chat_id,
            "output": output_payload,
        }
    except HTTPException as e:
        duration_ms = int((time.time() - started_at) * 1000)
        await log_session_event(
            event="query_failed_http",
            endpoint="/api/v1/query",
            request_id=request_id,
            session_id=session_key,
            user_id=context.user_id,
            query=request.query,
            success=False,
            duration_ms=duration_ms,
            error=str(e.detail),
            metadata={"status_code": e.status_code},
        )
        raise
    except Exception as e:
        duration_ms = int((time.time() - started_at) * 1000)
        metrics["errors_total"] += 1
        await log_session_event(
            event="query_failed",
            endpoint="/api/v1/query",
            request_id=request_id,
            session_id=session_key,
            user_id=context.user_id,
            query=request.query,
            success=False,
            duration_ms=duration_ms,
            error=str(e),
            metadata={"error_type": type(e).__name__},
        )
        logger.exception(f"/api/v1/query failed: {e}")
        log_structured(
            logger,
            "error",
            "query_endpoint_failed",
            request_id=request_id,
            session_id=session_key,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=_build_error_detail(
                "Failed to process query",
                code="MCP-QUERY-001",
                request_id=request_id,
            ),
        )


# =============================================================================
# Intelligent LLM-Powered Query Endpoint
# =============================================================================

@app.post("/api/v1/ask", tags=["Query"])
async def intelligent_query(
    request: QueryRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
):
    """
    AI-Powered query endpoint that can understand ANY natural language question.

    Uses Claude/OpenAI to:
    1. Understand the intent of the question
    2. Route to the correct tool
    3. Extract parameters intelligently
    4. Format a natural language response

    Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable to enable.
    Falls back to keyword-based routing if no API key is set.
    """
    from mcp.llm_router import get_intelligent_handler

    request_id = getattr(http_request.state, "request_id", None)
    chat_owner = _chat_owner_key(http_request, context)
    chat_thread = _get_chat_thread(chat_owner, request.chat_id) if request.chat_id else None
    resolved_session_id = request.session_id or (chat_thread.get("sessionId") if isinstance(chat_thread, dict) else None)
    session_key = resolved_session_id or "default"
    started_at = time.time()

    await log_session_event(
        event="ask_received",
        endpoint="/api/v1/ask",
        request_id=request_id,
        session_id=session_key,
        user_id=context.user_id,
        query=request.query,
        metadata={
            "output_format": request.output_format,
            "allow_download": request.allow_download,
        },
    )

    try:
        handler = get_intelligent_handler()
        result = await handler.process_query(
            request.query,
            context,
            session_id=resolved_session_id,
            output_format=request.output_format,
            allow_download=request.allow_download,
        )

        if request.chat_id and isinstance(result, dict):
            _upsert_chat_thread(chat_owner, chat_id=request.chat_id, session_id=resolved_session_id)
            _append_chat_message(chat_owner, request.chat_id, {"role": "user", "content": request.query})
            assistant_text = str(result.get("response") or "")
            if assistant_text:
                _append_chat_message(chat_owner, request.chat_id, {"role": "assistant", "content": assistant_text})
            _set_chat_last_assistant_result(chat_owner, request.chat_id, result)
            result["chat_id"] = request.chat_id

        duration_ms = int((time.time() - started_at) * 1000)
        routed_to = result.get("routed_to") if isinstance(result, dict) else None
        await log_session_event(
            event="ask_completed",
            endpoint="/api/v1/ask",
            request_id=request_id,
            session_id=session_key,
            user_id=context.user_id,
            query=request.query,
            tool_name=routed_to,
            success=bool(result.get("success")) if isinstance(result, dict) else None,
            duration_ms=duration_ms,
            metadata={
                "confidence": result.get("confidence") if isinstance(result, dict) else None,
                "history_size": result.get("history_size") if isinstance(result, dict) else None,
                "llm_enabled": result.get("llm_enabled") if isinstance(result, dict) else None,
                "route_source": result.get("route_source") if isinstance(result, dict) else None,
            },
        )

        return result
    except HTTPException as e:
        duration_ms = int((time.time() - started_at) * 1000)
        await log_session_event(
            event="ask_failed_http",
            endpoint="/api/v1/ask",
            request_id=request_id,
            session_id=session_key,
            user_id=context.user_id,
            query=request.query,
            success=False,
            duration_ms=duration_ms,
            error=str(e.detail),
            metadata={"status_code": e.status_code},
        )
        raise
    except Exception as e:
        duration_ms = int((time.time() - started_at) * 1000)
        metrics["errors_total"] += 1
        await log_session_event(
            event="ask_failed",
            endpoint="/api/v1/ask",
            request_id=request_id,
            session_id=session_key,
            user_id=context.user_id,
            query=request.query,
            success=False,
            duration_ms=duration_ms,
            error=str(e),
            metadata={"error_type": type(e).__name__},
        )
        logger.exception(f"/api/v1/ask failed: {e}")
        log_structured(
            logger,
            "error",
            "ask_endpoint_failed",
            request_id=request_id,
            session_id=session_key,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=_build_error_detail(
                "Failed to process request",
                code="MCP-ASK-001",
                request_id=request_id,
            ),
        )


# =============================================================================
# SSE Endpoint for MCP Protocol
# =============================================================================

@app.get("/api/v1/mcp/sse", tags=["MCP Protocol"])
async def mcp_sse_endpoint(
    request: Request,
    context: UserContext = Depends(get_current_user_context),
):
    """
    Server-Sent Events endpoint for MCP protocol.
    Enables real-time communication with MCP clients.
    """
    async def event_generator():
        handler = get_tool_handler()

        # Send initial tools list
        tools_event = {
            "type": "tools_list",
            "tools": handler.get_tools(),
        }
        yield f"data: {json.dumps(tools_event)}\n\n"

        # Keep connection alive
        while True:
            if await request.is_disconnected():
                break

            # Send heartbeat every 30 seconds
            yield f": heartbeat\n\n"
            await asyncio.sleep(30)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the HTTP server"""
    import uvicorn

    host = getattr(settings, 'MCP_HOST', '0.0.0.0')
    port = int(getattr(settings, 'MCP_PORT', 8090))

    uvicorn.run(
        "mcp.server_http:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()


