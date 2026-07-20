from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import math
import os
import secrets
import time
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Path
from pydantic import BaseModel, Field

from zhivex_ai import (
    Agent,
    ApprovalDecision,
    ConfigurationError,
    GatewayConfig,
    GatewayMessage,
    GatewayModelTarget,
    PendingApproval,
    ProviderHTTPError,
    UnsupportedFeatureError,
    ValidationError,
    create_gateway,
    create_openai,
    create_otel_agent_observer,
    create_postgres_agent_run_store,
    get_pending_agent_approvals,
    resume_agent_run,
    run_agent,
    tool,
)


MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024
DEFAULT_RATE_LIMIT_PER_MINUTE = 60
_RUN_IDENTIFIER_PATTERN = r"^[A-Za-z0-9_.:-]+$"
_RATE_LIMIT_LOCK = asyncio.Lock()
_RATE_LIMIT_WINDOWS: dict[str, deque[float]] = {}
logger = logging.getLogger(__name__)


class _RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Reject declared and chunked oversized bodies before FastAPI parses them."""

    def __init__(self, app: Any, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                await self._send_error(send, 400, "Invalid Content-Length header.")
                return
            if content_length < 0:
                await self._send_error(send, 400, "Invalid Content-Length header.")
                return
            if content_length > self.max_body_bytes:
                await self._send_error(send, 413, "Request body too large.")
                return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await self._send_error(send, 413, "Request body too large.")

    @staticmethod
    async def _send_error(send: Any, status: int, detail: str) -> None:
        body = (f'{{"detail":"{detail}"}}').encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


app = FastAPI(title="Zhivex AI SDK Production Agent API")
app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=MAX_REQUEST_BODY_BYTES)


@dataclass(frozen=True, slots=True)
class APIIdentity:
    tenant_id: str


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_768)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)


class ApprovalResponse(BaseModel):
    approval_id: str
    tool_name: str
    reason: str | None = None
    permissions: list[str] = Field(default_factory=list)
    created_at_ms: int | None = None


class AgentResponse(BaseModel):
    text: str
    run_id: str
    session_id: str
    request_id: str
    status: str | None = None
    pending_approvals: list[ApprovalResponse] = Field(default_factory=list)


class ApprovalDecisionRequest(BaseModel):
    approved: bool = True
    reason: str | None = Field(default=None, max_length=2_048)


def _configured_model() -> str:
    model = (os.getenv("ZHIVEX_AGENT_MODEL") or "").strip()
    if not model or len(model) > 256 or any(character.isspace() for character in model):
        raise HTTPException(status_code=503, detail="Server model configuration is invalid.")
    return model


def _configured_rate_limit() -> int:
    raw_value = os.getenv("ZHIVEX_RATE_LIMIT_PER_MINUTE", str(DEFAULT_RATE_LIMIT_PER_MINUTE))
    try:
        value = int(raw_value)
    except ValueError:
        raise HTTPException(status_code=503, detail="Server rate-limit configuration is invalid.") from None
    if not 1 <= value <= 10_000:
        raise HTTPException(status_code=503, detail="Server rate-limit configuration is invalid.")
    return value


async def _enforce_rate_limit(identity: APIIdentity) -> None:
    now = time.monotonic()
    cutoff = now - 60.0
    limit = _configured_rate_limit()
    async with _RATE_LIMIT_LOCK:
        window = _RATE_LIMIT_WINDOWS.setdefault(identity.tenant_id, deque())
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded.", headers={"Retry-After": "60"})
        window.append(now)


async def require_api_identity(
    authorization: str | None = Header(default=None, alias="Authorization", max_length=4_096),
    requested_tenant: str | None = Header(default=None, alias="X-Tenant-ID", min_length=1, max_length=128),
) -> APIIdentity:
    configured_token = os.getenv("ZHIVEX_AGENT_API_TOKEN")
    configured_tenant = os.getenv("ZHIVEX_TENANT_ID")
    if not configured_token or not configured_tenant:
        raise HTTPException(status_code=503, detail="API authentication is not configured.")

    scheme, _, supplied_token = (authorization or "").partition(" ")
    token_matches = secrets.compare_digest(supplied_token.encode("utf-8"), configured_token.encode("utf-8"))
    tenant_matches = requested_tenant is not None and secrets.compare_digest(
        requested_tenant.encode("utf-8"),
        configured_tenant.encode("utf-8"),
    )
    if scheme.lower() != "bearer" or not token_matches or not tenant_matches:
        raise HTTPException(
            status_code=401,
            detail="Invalid API credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    identity = APIIdentity(tenant_id=configured_tenant)
    await _enforce_rate_limit(identity)
    return identity


def _http_exception_from_sdk_error(error: Exception) -> HTTPException:
    if isinstance(error, (ValidationError, UnsupportedFeatureError)):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, ConfigurationError):
        return HTTPException(status_code=500, detail="Server configuration error.")
    if isinstance(error, ProviderHTTPError):
        headers: dict[str, str] = {}
        if error.retry_after_ms is not None:
            headers["Retry-After"] = str(max(1, math.ceil(error.retry_after_ms / 1000)))
        return HTTPException(
            status_code=503 if error.retryable else 502,
            detail={
                "message": "Upstream provider request failed.",
                "provider_status": error.status,
                "retryable": error.retryable,
            },
            headers=headers,
        )
    return HTTPException(status_code=500, detail="Internal server error.")


def _optional_otel_observer():
    try:
        return create_otel_agent_observer()
    except RuntimeError:
        return None


def _approval_response(item: PendingApproval) -> ApprovalResponse:
    return ApprovalResponse(
        approval_id=item.id,
        tool_name=item.name,
        reason=item.reason,
        permissions=list(item.permissions),
        created_at_ms=item.created_at_ms,
    )


async def _approval_policy(request) -> ApprovalDecision | bool:
    if "finance:write" in request.tool_permissions:
        return ApprovalDecision.require_human("Finance write action requires manager approval.")
    return True


def _build_agent(*, model: str):
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise HTTPException(status_code=500, detail="DATABASE_URL is required for production run storage.")
    table_prefix = os.getenv("ZHIVEX_AGENT_TABLE_PREFIX", "zhivex_agent")
    run_store = create_postgres_agent_run_store(dsn, table_prefix=table_prefix)
    provider = create_openai()
    submit_invoice = tool(
        name="submit_invoice",
        description="Submit an invoice for payment after approval.",
        schema=dict[str, str],
        execute=lambda input: {"invoice_id": input["invoice_id"], "status": "submitted"},
        permissions=["finance:write"],
        requires_approval=True,
    )
    return Agent(
        name="production_assistant",
        model=provider(model),
        run_store=run_store,
        tools={"submit_invoice": submit_invoice},
        approval_policy=_approval_policy,
    )


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/agent", response_model=AgentResponse)
async def run_agent_endpoint(
    request: AgentRequest,
    request_id: str | None = Header(default=None, alias="X-Request-ID", min_length=1, max_length=200),
    identity: APIIdentity = Depends(require_api_identity),
) -> AgentResponse:
    resolved_request_id = request_id or f"req_{uuid4().hex}"
    agent = _build_agent(model=_configured_model())
    agent.metadata = {"request_id": resolved_request_id, "tenant_id": identity.tenant_id}

    try:
        result = await run_agent(
            agent=agent,
            prompt=request.prompt,
            idempotency_key=f"{identity.tenant_id}:{request.idempotency_key or resolved_request_id}",
            observer=_optional_otel_observer(),
        )
    except Exception as error:
        raise _http_exception_from_sdk_error(error) from error

    return AgentResponse(
        text=result.text,
        run_id=result.run_id,
        session_id=result.session.id,
        request_id=resolved_request_id,
        status=result.state.status if result.state is not None else None,
        pending_approvals=[_approval_response(item) for item in (result.state.pending_approvals if result.state else [])],
    )


@app.get("/v1/agent/runs/{run_id}/approvals", response_model=list[ApprovalResponse])
async def list_run_approvals(
    run_id: Annotated[str, Path(min_length=1, max_length=200, pattern=_RUN_IDENTIFIER_PATTERN)],
    _identity: APIIdentity = Depends(require_api_identity),
) -> list[ApprovalResponse]:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise HTTPException(status_code=500, detail="DATABASE_URL is required for production run storage.")
    table_prefix = os.getenv("ZHIVEX_AGENT_TABLE_PREFIX", "zhivex_agent")
    pending = await get_pending_agent_approvals(
        create_postgres_agent_run_store(dsn, table_prefix=table_prefix),
        run_id,
    )
    return [_approval_response(item) for item in pending]


@app.post("/v1/agent/runs/{run_id}/approvals/{approval_id}", response_model=AgentResponse)
async def resolve_run_approval(
    run_id: Annotated[str, Path(min_length=1, max_length=200, pattern=_RUN_IDENTIFIER_PATTERN)],
    approval_id: Annotated[str, Path(min_length=1, max_length=200, pattern=_RUN_IDENTIFIER_PATTERN)],
    request: ApprovalDecisionRequest,
    request_id: str | None = Header(default=None, alias="X-Request-ID", min_length=1, max_length=200),
    identity: APIIdentity = Depends(require_api_identity),
) -> AgentResponse:
    resolved_request_id = request_id or f"req_{uuid4().hex}"
    agent = _build_agent(model=_configured_model())
    agent.metadata = {"request_id": resolved_request_id, "tenant_id": identity.tenant_id}
    try:
        result = await resume_agent_run(
            agent=agent,
            run_id=run_id,
            approval_id=approval_id,
            approved=request.approved,
            reason=request.reason,
            observer=_optional_otel_observer(),
        )
    except Exception as error:
        raise _http_exception_from_sdk_error(error) from error
    return AgentResponse(
        text=result.text,
        run_id=result.run_id,
        session_id=result.session.id,
        request_id=resolved_request_id,
        status=result.state.status if result.state is not None else None,
        pending_approvals=[_approval_response(item) for item in (result.state.pending_approvals if result.state else [])],
    )


def _log_gateway_attempt(attempt: dict[str, Any]) -> None:
    logger.info(
        "gateway_attempt provider=%s model=%s ok=%s retryable=%s retry=%s",
        attempt.get("provider"),
        attempt.get("model_id"),
        attempt.get("ok"),
        attempt.get("retryable"),
        attempt.get("retry"),
    )


@app.post("/v1/gateway")
async def gateway_endpoint(
    request: AgentRequest,
    _identity: APIIdentity = Depends(require_api_identity),
) -> dict[str, object]:
    gateway = create_gateway(
        GatewayConfig(
            adapters={"openai": create_openai()},
            fail_on_missing_adapter=True,
            on_attempt=_log_gateway_attempt,
        )
    )
    try:
        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content=request.prompt)],
            primary=GatewayModelTarget(provider="openai", model_id=_configured_model()),
        )
    except Exception as error:
        raise _http_exception_from_sdk_error(error) from error
    return {
        "text": result.text,
        "provider": result.provider_used,
        "model": result.model_used,
        "attempts": [
            {
                "provider": attempt.provider,
                "model_id": attempt.model_id,
                "ok": attempt.ok,
                "latency_ms": attempt.latency_ms,
                "retryable": attempt.retryable,
            }
            for attempt in result.attempts
        ],
    }
