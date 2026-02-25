"""Shared user-readable error codes and payload helpers."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ErrorDefinition:
    """Metadata for a user-facing error code."""

    code: str
    default_message: str
    user_action: str


ERROR_CATALOG: Dict[str, ErrorDefinition] = {
    "MCP-AUTH-001": ErrorDefinition(
        code="MCP-AUTH-001",
        default_message="Authentication header format is invalid.",
        user_action="Use the header format: Authorization: Bearer <token>.",
    ),
    "MCP-AUTH-002": ErrorDefinition(
        code="MCP-AUTH-002",
        default_message="Authentication token is invalid or expired.",
        user_action="Sign in again and retry with a fresh token.",
    ),
    "MCP-AUTH-003": ErrorDefinition(
        code="MCP-AUTH-003",
        default_message="Authentication failed.",
        user_action="Verify your credentials and retry.",
    ),
    "MCP-RATE-001": ErrorDefinition(
        code="MCP-RATE-001",
        default_message="Too many requests. Rate limit exceeded.",
        user_action="Wait a minute and try again.",
    ),
    "MCP-TOOL-001": ErrorDefinition(
        code="MCP-TOOL-001",
        default_message="Requested tool was not found.",
        user_action="Check the tool name from /api/v1/mcp/tools and retry.",
    ),
    "MCP-TOOL-002": ErrorDefinition(
        code="MCP-TOOL-002",
        default_message="Tool execution failed.",
        user_action="Retry the request. If it keeps failing, share the request ID with support.",
    ),
    "MCP-QUERY-001": ErrorDefinition(
        code="MCP-QUERY-001",
        default_message="Failed to process query.",
        user_action="Try rephrasing the query or reducing filters.",
    ),
    "MCP-ASK-001": ErrorDefinition(
        code="MCP-ASK-001",
        default_message="Failed to process request.",
        user_action="Retry once. If it fails again, report the error code and request ID.",
    ),
    "MCP-VALIDATION-001": ErrorDefinition(
        code="MCP-VALIDATION-001",
        default_message="Input validation failed.",
        user_action="Check required fields and data types, then retry.",
    ),
    "MCP-INPUT-001": ErrorDefinition(
        code="MCP-INPUT-001",
        default_message="Required input parameter is missing.",
        user_action="Provide all required fields and retry.",
    ),
    "MCP-INPUT-002": ErrorDefinition(
        code="MCP-INPUT-002",
        default_message="Input parameter value is invalid.",
        user_action="Correct the invalid field value and retry.",
    ),
    "MCP-DATA-001": ErrorDefinition(
        code="MCP-DATA-001",
        default_message="Requested record was not found.",
        user_action="Verify names/IDs and retry the request.",
    ),
    "MCP-SCOPE-001": ErrorDefinition(
        code="MCP-SCOPE-001",
        default_message="You do not have permission for this data scope.",
        user_action="Use a permitted scope or request higher access.",
    ),
    "MCP-READY-001": ErrorDefinition(
        code="MCP-READY-001",
        default_message="Service is not ready.",
        user_action="Try again in a few seconds.",
    ),
    "MCP-INTERNAL-001": ErrorDefinition(
        code="MCP-INTERNAL-001",
        default_message="Unexpected internal server error.",
        user_action="Retry the request. If the issue persists, report the request ID.",
    ),
}

DEFAULT_ERROR = ErrorDefinition(
    code="MCP-INTERNAL-001",
    default_message="Unexpected internal server error.",
    user_action="Retry the request. If the issue persists, report the request ID.",
)


LEGACY_CODE_MAP: Dict[str, str] = {
    "VALIDATION_ERROR": "MCP-VALIDATION-001",
    "QUERY_ERROR": "MCP-TOOL-002",
    "SCOPE_ACCESS_ERROR": "MCP-SCOPE-001",
    "TOOL_NOT_FOUND": "MCP-TOOL-001",
    "INTERNAL_ERROR": "MCP-INTERNAL-001",
    "TOOL_EXECUTION_ERROR": "MCP-TOOL-002",
    "QUERY_PROCESSING_ERROR": "MCP-QUERY-001",
    "ASK_PROCESSING_ERROR": "MCP-ASK-001",
    "MISSING_PARAMETER": "MCP-INPUT-001",
    "INVALID_PARAMETER": "MCP-INPUT-002",
    "NOT_FOUND": "MCP-DATA-001",
}


def normalize_error_code(code: str) -> str:
    """Normalize legacy/internal codes to user-facing MCP codes."""
    if code in ERROR_CATALOG:
        return code
    if code in LEGACY_CODE_MAP:
        return LEGACY_CODE_MAP[code]
    return DEFAULT_ERROR.code


def build_error_payload(
    code: str,
    *,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    legacy_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a consistent user-readable error payload."""
    normalized_code = normalize_error_code(code)
    definition = ERROR_CATALOG.get(normalized_code, DEFAULT_ERROR)

    payload_details: Dict[str, Any] = dict(details or {})
    if request_id:
        payload_details["request_id"] = request_id
    if legacy_code and legacy_code != normalized_code:
        payload_details["legacy_code"] = legacy_code

    return {
        "code": normalized_code,
        "message": message or definition.default_message,
        "user_action": definition.user_action,
        "details": payload_details,
    }
