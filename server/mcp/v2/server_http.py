"""MCP V2 HTTP/SSE server based on the TFS layered architecture."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    import redis.asyncio as redis
except Exception:  # pragma: no cover - optional dependency
    redis = None

from mcp.config import mcp_settings
from mcp.core.config import settings
from mcp.core.database import close_mongodb_connection, connect_to_mongodb, get_database
from mcp.core.error_catalog import build_error_payload, normalize_error_code
from mcp.core.logging_config import configure_logging, log_structured
from mcp.core.security import decode_access_token
from mcp.router.llm_client import has_llm_api_key
from mcp.schemas.context_schema import UserContext
from mcp.utils.output_layer import build_output_payload
from mcp.v2.handlers import get_tool_handler
from mcp.v2.orchestration import route_query_to_tool_enriched
from mcp.v2.utils import generate_natural_language_response

configure_logging()
logger = logging.getLogger(__name__)

# Optional Redis client for rate limiting.
redis_client: Optional["redis.Redis"] = None

metrics: Dict[str, Any] = {
    "requests_total": 0,
    "requests_by_tool": {},
    "errors_total": 0,
}


class ToolExecuteRequest(BaseModel):
    arguments: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    tool_hint: Optional[str] = None
    output_format: Optional[str] = "auto"
    allow_download: Optional[bool] = None
    session_id: Optional[str] = None
    chat_id: Optional[str] = None


class ChatMessageRequest(BaseModel):
    role: str
    content: str
    rawHtml: bool = False
    tone: str = "normal"
    timestamp: Optional[str] = None


class ChatCreateRequest(BaseModel):
    chat_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatThreadRequest(BaseModel):
    active_chat_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    mongodb: str
    redis: str
    tools_loaded: int


_MAX_QUERY_HISTORY_SESSIONS = 500
_query_history: "OrderedDict[str, str]" = OrderedDict()

_MAX_CHAT_OWNERS = 500
_MAX_CHATS_PER_OWNER = 100
_MAX_MESSAGES_PER_CHAT = 500
_chat_threads_by_owner: "OrderedDict[str, OrderedDict[str, Dict[str, Any]]]" = OrderedDict()


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


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
        "id": chat_id or f"chat-{uuid4().hex}",
        "sessionId": session_id or f"session-{uuid4().hex}",
        "createdAt": now,
        "updatedAt": now,
        "messages": [],
        "lastAssistantResult": None,
    }


def _list_chat_threads(owner_key: str) -> List[Dict[str, Any]]:
    bucket = _ensure_owner_chat_bucket(owner_key)
    rows = list(bucket.values())
    rows.sort(key=lambda t: t.get("updatedAt") or t.get("createdAt") or "", reverse=True)
    return [copy.deepcopy(row) for row in rows]


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
    if session_id:
        thread["sessionId"] = session_id
    thread["updatedAt"] = _now_iso()
    bucket.move_to_end(thread["id"])
    while len(bucket) > _MAX_CHATS_PER_OWNER:
        bucket.popitem(last=False)
    return thread


def _get_chat_thread(owner_key: str, chat_id: str) -> Optional[Dict[str, Any]]:
    bucket = _ensure_owner_chat_bucket(owner_key)
    thread = bucket.get(chat_id)
    if thread is not None:
        bucket.move_to_end(chat_id)
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


def _delete_chat_thread(owner_key: str, chat_id: str) -> bool:
    bucket = _ensure_owner_chat_bucket(owner_key)
    if chat_id in bucket:
        del bucket[chat_id]
        return True
    return False


def _clear_chat_threads(owner_key: str) -> None:
    bucket = _ensure_owner_chat_bucket(owner_key)
    bucket.clear()


def _build_error_detail(
    message: str,
    *,
    code: str,
    request_id: Optional[str],
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = normalize_error_code(code)
    return build_error_payload(
        normalized,
        message=message,
        details=details,
        request_id=request_id,
        legacy_code=code if normalized != code else None,
    )


def _determine_scope_level(payload: Dict[str, Any]) -> str:
    explicit = str(payload.get("scopeLevel") or payload.get("scope_level") or "").strip().lower()
    if explicit in {"unit", "district", "state"}:
        return explicit

    role_name = str(payload.get("roleName") or payload.get("role_name") or "").lower()
    if role_name in {"super admin", "state admin", "dgp"}:
        return "state"
    if role_name in {"sp", "district admin", "range ig"}:
        return "district"
    return "unit"


async def get_current_user_context(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> UserContext:
    if not authorization:
        anon_scope = str(getattr(settings, "MCP_ANON_SCOPE_LEVEL", "state") or "state").strip().lower()
        if anon_scope not in {"unit", "district", "state"}:
            anon_scope = "state"
        return UserContext(scope_level=anon_scope)

    try:
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

        return UserContext(
            user_id=payload.get("id") or payload.get("sub"),
            unit_id=payload.get("unitId"),
            unit_ids=[u.get("unitId") for u in payload.get("units", []) if isinstance(u, dict) and u.get("unitId")],
            district_id=payload.get("districtId"),
            district_ids=[payload.get("districtId")] if payload.get("districtId") else [],
            role_id=payload.get("roleId"),
            post_code=payload.get("postCode") or payload.get("post_code"),
            role_name=payload.get("roleName") or payload.get("role_name"),
            reports_to_post=payload.get("reportsToPost") or payload.get("reports_to_post"),
            scope_level=_determine_scope_level(payload),
        )
    except HTTPException:
        raise
    except Exception as error:
        log_structured(
            logger,
            "warning",
            "v2_auth_failed",
            error=str(error),
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


async def check_rate_limit(
    request: Request,
    context: UserContext = Depends(get_current_user_context),
) -> None:
    if not redis_client:
        return

    user_id = context.user_id or getattr(getattr(request, "client", None), "host", "unknown")
    key = f"mcp:v2:rate:{user_id}"
    limit = 60
    try:
        current = await redis_client.incr(key)
        if current == 1:
            await redis_client.expire(key, 60)
        if int(current) > limit:
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
    except Exception as error:
        log_structured(
            logger,
            "warning",
            "v2_rate_limit_check_failed",
            error=str(error),
            user_id=user_id,
            request_id=getattr(request.state, "request_id", None),
        )


def _build_capability_help_response_text() -> str:
    return (
        "I can help with V2 enriched lookups:\n\n"
        "- Personnel by person/rank-like designation/district (e.g., 'SP in Guntur')\n"
        "- Unit lookup (e.g., 'Where is Kuppam SDPO unit')\n"
        "- Responsible officer checks (e.g., 'Who is in-charge of Guntur PS')\n"
        "- Assignment lookup (e.g., 'Assignments for user id 14402876')"
    )


async def _startup() -> None:
    global redis_client
    await connect_to_mongodb()
    get_tool_handler().initialize()

    if redis is not None and settings.REDIS_URL:
        try:
            redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            await redis_client.ping()
            log_structured(logger, "info", "v2_redis_connected")
        except Exception as error:
            redis_client = None
            log_structured(logger, "warning", "v2_redis_unavailable", error=str(error))

    log_structured(logger, "info", "v2_server_startup", logic_version="v2")


async def _shutdown() -> None:
    global redis_client
    if redis_client:
        try:
            await redis_client.close()
        except Exception:
            pass
        redis_client = None
    await close_mongodb_connection()
    log_structured(logger, "info", "v2_server_shutdown")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await _startup()
    try:
        yield
    finally:
        await _shutdown()


app = FastAPI(
    title=f"{mcp_settings.MCP_SERVER_NAME}-v2",
    version=f"{mcp_settings.MCP_SERVER_VERSION}-v2",
    lifespan=lifespan,
)

allowed_origins = [o.strip() for o in str(settings.ALLOWED_ORIGINS or "*").split(",") if o.strip()]
if not allowed_origins:
    allowed_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    started = time.time()
    try:
        response = await call_next(request)
    except Exception:
        metrics["errors_total"] += 1
        duration_ms = int((time.time() - started) * 1000)
        log_structured(
            logger,
            "error",
            "v2_request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
        )
        raise

    response.headers["X-Request-ID"] = request_id
    duration_ms = int((time.time() - started) * 1000)
    log_structured(
        logger,
        "info",
        "v2_request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


def route_query_to_tool(query: str, hint: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    tool_aliases = {
        "count_vacancies_by_unit": "count_vacancies_by_unit_rank",
        "count_vacancies": "count_vacancies_by_unit_rank",
        "vacancies_by_unit": "count_vacancies_by_unit_rank",
    }

    def _normalize_with_repairs(tool_name: Optional[str], arguments: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        normalized_tool = str(tool_name or "__help__").strip() or "__help__"
        normalized_tool = tool_aliases.get(normalized_tool, normalized_tool)
        normalized_args = dict(arguments or {})
        if normalized_tool not in {"__help__", "search_assignment"}:
            try:
                from mcp.router import repair_route

                repaired_tool, repaired_args = repair_route(query, normalized_tool, normalized_args)
                if isinstance(repaired_tool, str) and repaired_tool.strip():
                    normalized_tool = repaired_tool.strip()
                if isinstance(repaired_args, dict):
                    normalized_args = repaired_args
            except Exception:
                pass
            normalized_tool = tool_aliases.get(normalized_tool, normalized_tool)
        return normalized_tool, normalized_args

    try:
        available_tools = set(get_tool_handler().get_tool_names())
    except Exception:
        available_tools = None

    # Keep V1 parity as default behavior for existing query styles.
    v1_tool: Optional[str] = None
    v1_arguments: Dict[str, Any] = {}
    try:
        from mcp.v1.server_http import route_query_to_tool as route_query_to_tool_v1

        v1_tool, v1_arguments = route_query_to_tool_v1(query, hint)
    except Exception:
        v1_tool, v1_arguments = None, {}

    tool_name, arguments = route_query_to_tool_enriched(
        query,
        hint=hint,
        available_tools=available_tools,
    )

    if v1_tool and (available_tools is None or v1_tool in available_tools):
        # Preserve V1 semantics for existing tools; only override with V2-only
        # routes when that adds new functionality.
        if tool_name == "search_assignment":
            return _normalize_with_repairs(tool_name, arguments)
        return _normalize_with_repairs(v1_tool, v1_arguments)

    if tool_name:
        return _normalize_with_repairs(tool_name, arguments)
    return "__help__", {"reason": "No route matched the query."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    handler = get_tool_handler()
    mongodb_status = "unhealthy"
    redis_status = "disabled"

    try:
        db = get_database()
        if db is not None:
            await db.command("ping")
            mongodb_status = "healthy"
    except Exception:
        mongodb_status = "unhealthy"

    if redis_client:
        try:
            await redis_client.ping()
            redis_status = "healthy"
        except Exception:
            redis_status = "unhealthy"

    return HealthResponse(
        status="healthy" if mongodb_status == "healthy" else "degraded",
        version=f"{mcp_settings.MCP_SERVER_VERSION}-v2",
        mongodb=mongodb_status,
        redis=redis_status,
        tools_loaded=len(handler.get_tool_names()),
    )


@app.get("/ready", tags=["Health"])
async def readiness_check() -> Dict[str, str]:
    db = get_database()
    if db is None:
        raise HTTPException(
            status_code=503,
            detail=_build_error_detail("Not ready", code="MCP-READY-001", request_id=None),
        )
    try:
        await db.command("ping")
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=_build_error_detail("Not ready", code="MCP-READY-001", request_id=None),
        )
    return {"status": "ready"}


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics() -> str:
    handler = get_tool_handler()
    lines = [
        "# HELP mcp_requests_total Total MCP requests",
        "# TYPE mcp_requests_total counter",
        f"mcp_requests_total {metrics['requests_total']}",
        "# HELP mcp_tools_loaded Number of tools loaded",
        "# TYPE mcp_tools_loaded gauge",
        f"mcp_tools_loaded {len(handler.get_tool_names())}",
        "# HELP mcp_errors_total Total server errors",
        "# TYPE mcp_errors_total counter",
        f"mcp_errors_total {metrics['errors_total']}",
    ]
    return "\n".join(lines)


@app.get("/api/v1/mcp/tools", tags=["MCP Tools"])
async def list_tools(_: None = Depends(check_rate_limit)) -> Dict[str, Any]:
    handler = get_tool_handler()
    return {
        "success": True,
        "tools": handler.get_tools(),
        "count": len(handler.get_tool_names()),
        "logic_version": "v2",
    }


@app.get("/api/v1/mcp/tools/{tool_name}", tags=["MCP Tools"])
async def get_tool_schema(
    tool_name: str,
    http_request: Request,
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    handler = get_tool_handler()
    tools = {tool["name"]: tool for tool in handler.get_tools()}
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
    return {"success": True, "tool": tools[tool_name], "logic_version": "v2"}


@app.post("/api/v1/mcp/tools/{tool_name}/execute", tags=["MCP Tools"])
async def execute_tool(
    tool_name: str,
    request: ToolExecuteRequest,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    handler = get_tool_handler()
    metrics["requests_total"] += 1
    metrics["requests_by_tool"][tool_name] = metrics["requests_by_tool"].get(tool_name, 0) + 1
    return await handler.execute(tool_name, request.arguments, context)


@app.get("/api/v1/chats", tags=["Chat"])
async def list_chats(
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    owner_key = _chat_owner_key(http_request, context)
    chats = _list_chat_threads(owner_key)
    if not chats:
        thread = _upsert_chat_thread(owner_key)
        chats = [copy.deepcopy(thread)]
    active_chat_id = chats[0]["id"]
    return {"success": True, "chats": chats, "active_chat_id": active_chat_id}


@app.post("/api/v1/chats", tags=["Chat"])
async def create_chat(
    request: ChatCreateRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    owner_key = _chat_owner_key(http_request, context)
    thread = _upsert_chat_thread(
        owner_key,
        chat_id=request.chat_id,
        session_id=request.session_id,
    )
    return {"success": True, "chat": copy.deepcopy(thread), "active_chat_id": thread["id"]}


@app.get("/api/v1/chats/{chat_id}", tags=["Chat"])
async def get_chat(
    chat_id: str,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    owner_key = _chat_owner_key(http_request, context)
    thread = _get_chat_thread(owner_key, chat_id)
    if not thread:
        raise HTTPException(
            status_code=404,
            detail=_build_error_detail(
                f"Chat not found: {chat_id}",
                code="MCP-DATA-001",
                request_id=getattr(http_request.state, "request_id", None),
            ),
        )
    return {"success": True, "chat": copy.deepcopy(thread)}


@app.delete("/api/v1/chats/{chat_id}", tags=["Chat"])
async def delete_chat(
    chat_id: str,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    owner_key = _chat_owner_key(http_request, context)
    _delete_chat_thread(owner_key, chat_id)
    chats = _list_chat_threads(owner_key)
    if not chats:
        thread = _upsert_chat_thread(owner_key)
        chats = [copy.deepcopy(thread)]
    return {"success": True, "chats": chats, "active_chat_id": chats[0]["id"]}


@app.delete("/api/v1/chats", tags=["Chat"])
async def clear_chats(
    request: ChatThreadRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    owner_key = _chat_owner_key(http_request, context)
    _clear_chat_threads(owner_key)
    thread = _upsert_chat_thread(owner_key, chat_id=request.active_chat_id)
    chats = [copy.deepcopy(thread)]
    return {"success": True, "chats": chats, "active_chat_id": thread["id"]}


@app.post("/api/v1/chats/{chat_id}/messages", tags=["Chat"])
async def append_chat_message(
    chat_id: str,
    request: ChatMessageRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    owner_key = _chat_owner_key(http_request, context)
    thread = _append_chat_message(owner_key, chat_id, request.model_dump())
    return {"success": True, "chat": copy.deepcopy(thread)}


@app.post("/api/v1/query", tags=["Query"])
async def natural_language_query(
    request: QueryRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    handler = get_tool_handler()
    metrics["requests_total"] += 1

    chat_owner = _chat_owner_key(http_request, context)
    chat_thread = _get_chat_thread(chat_owner, request.chat_id) if request.chat_id else None
    resolved_session_id = request.session_id or (chat_thread.get("sessionId") if isinstance(chat_thread, dict) else None)
    session_key = resolved_session_id or "default"
    last_response_hint = _query_history_get(session_key)

    route_mode = "rule_based"
    route_source: Optional[str] = None
    route_confidence: Optional[float] = None
    understood_query: Optional[str] = None

    if request.tool_hint and request.tool_hint in handler.get_tool_names():
        tool_name, extracted_args = request.tool_hint, {}
        route_mode = "explicit_hint"
    else:
        if has_llm_api_key():
            from mcp.v2.llm_router import llm_route_query

            llm_context = [{"role": "assistant", "content": last_response_hint}] if last_response_hint else None
            tool_name, extracted_args, understood_query, route_confidence, route_source = await llm_route_query(
                request.query,
                llm_context,
                available_tools=handler.get_tool_names(),
            )
            route_mode = route_source or "llm"
        else:
            tool_name, extracted_args = route_query_to_tool(request.query, request.tool_hint)

    if tool_name == "__help__":
        reason = str(extracted_args.get("reason") or "").strip()
        response_text = f"{reason}\n\n{_build_capability_help_response_text()}".strip() if reason else _build_capability_help_response_text()
        output_payload = build_output_payload(
            query=request.query,
            response_text=response_text,
            routed_to=None,
            arguments={},
            result={},
            requested_format=request.output_format,
            allow_download=request.allow_download,
        )
        _query_history_set(session_key, response_text)
        return {
            "success": True,
            "query": request.query,
            "response": response_text,
            "routed_to": None,
            "extracted_arguments": {},
            "data": {},
            "chat_id": request.chat_id,
            "route_mode": route_mode,
            "route_source": route_source,
            "route_confidence": route_confidence,
            "understood_query": understood_query,
            "output": output_payload,
        }

    result = await handler.execute(tool_name, extracted_args, context)
    response_text = generate_natural_language_response(
        request.query,
        tool_name,
        extracted_args,
        result,
    )
    output_payload = build_output_payload(
        query=request.query,
        response_text=response_text,
        routed_to=tool_name,
        arguments=extracted_args,
        result=result,
        requested_format=request.output_format,
        allow_download=request.allow_download,
    )
    _query_history_set(session_key, response_text)

    if request.chat_id:
        _upsert_chat_thread(chat_owner, chat_id=request.chat_id, session_id=resolved_session_id)
        _append_chat_message(chat_owner, request.chat_id, {"role": "user", "content": request.query})
        _append_chat_message(chat_owner, request.chat_id, {"role": "assistant", "content": response_text})
        _set_chat_last_assistant_result(
            chat_owner,
            request.chat_id,
            {
                "success": bool(result.get("success")) if isinstance(result, dict) else True,
                "query": request.query,
                "response": response_text,
                "routed_to": tool_name,
                "extracted_arguments": extracted_args,
                "data": result,
                "output": output_payload,
            },
        )

    return {
        "success": bool(result.get("success")) if isinstance(result, dict) else True,
        "query": request.query,
        "response": response_text,
        "routed_to": tool_name,
        "extracted_arguments": extracted_args,
        "data": result,
        "chat_id": request.chat_id,
        "route_mode": route_mode,
        "route_source": route_source,
        "route_confidence": route_confidence,
        "understood_query": understood_query,
        "output": output_payload,
    }


@app.post("/api/v1/ask", tags=["Query"])
async def intelligent_query(
    request: QueryRequest,
    http_request: Request,
    context: UserContext = Depends(get_current_user_context),
    _: None = Depends(check_rate_limit),
) -> Dict[str, Any]:
    from mcp.v2.llm_router import get_intelligent_handler

    handler = get_intelligent_handler()
    chat_owner = _chat_owner_key(http_request, context)
    chat_thread = _get_chat_thread(chat_owner, request.chat_id) if request.chat_id else None
    resolved_session_id = request.session_id or (chat_thread.get("sessionId") if isinstance(chat_thread, dict) else None)

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

    return result


@app.get("/api/v1/mcp/sse", tags=["MCP Protocol"])
async def mcp_sse(
    request: Request,
    _: None = Depends(check_rate_limit),
):
    handler = get_tool_handler()

    async def event_generator():
        hello_event = {
            "event": "connected",
            "server": mcp_settings.MCP_SERVER_NAME,
            "version": f"{mcp_settings.MCP_SERVER_VERSION}-v2",
            "logic_version": "v2",
        }
        yield f"data: {json.dumps(hello_event, default=str)}\n\n"

        tools_event = {
            "event": "tools",
            "tools": handler.get_tools(),
            "count": len(handler.get_tool_names()),
        }
        yield f"data: {json.dumps(tools_event, default=str)}\n\n"

        while True:
            if await request.is_disconnected():
                break
            yield ": heartbeat\n\n"
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


def main() -> None:
    import uvicorn

    host = getattr(settings, "MCP_HOST", "0.0.0.0")
    port = int(getattr(settings, "MCP_PORT", 8090))
    uvicorn.run(
        "mcp.server_http:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
