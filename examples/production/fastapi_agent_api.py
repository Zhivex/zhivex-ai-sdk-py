from __future__ import annotations

import math
import os
from dataclasses import asdict
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
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


app = FastAPI(title="Zhivex AI SDK Production Agent API")


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = "gpt-5.4-mini"
    idempotency_key: str | None = None


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
    reason: str | None = None
    model: str = "gpt-5.4-mini"


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
    run_store = create_postgres_agent_run_store(dsn)
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
    request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> AgentResponse:
    resolved_request_id = request_id or f"req_{uuid4().hex}"
    agent = _build_agent(model=request.model)
    agent.metadata = {"request_id": resolved_request_id}

    try:
        result = await run_agent(
            agent=agent,
            prompt=request.prompt,
            idempotency_key=request.idempotency_key or resolved_request_id,
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
async def list_run_approvals(run_id: str) -> list[ApprovalResponse]:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise HTTPException(status_code=500, detail="DATABASE_URL is required for production run storage.")
    pending = await get_pending_agent_approvals(create_postgres_agent_run_store(dsn), run_id)
    return [_approval_response(item) for item in pending]


@app.post("/v1/agent/runs/{run_id}/approvals/{approval_id}", response_model=AgentResponse)
async def resolve_run_approval(
    run_id: str,
    approval_id: str,
    request: ApprovalDecisionRequest,
    request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> AgentResponse:
    resolved_request_id = request_id or f"req_{uuid4().hex}"
    agent = _build_agent(model=request.model)
    agent.metadata = {"request_id": resolved_request_id}
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


@app.post("/v1/gateway")
async def gateway_endpoint(request: AgentRequest) -> dict[str, object]:
    gateway = create_gateway(
        GatewayConfig(
            adapters={"openai": create_openai()},
            fail_on_missing_adapter=True,
            on_attempt=lambda attempt: print({"gateway_attempt": attempt}),
        )
    )
    try:
        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content=request.prompt)],
            primary=GatewayModelTarget(provider="openai", model_id=request.model),
        )
    except Exception as error:
        raise _http_exception_from_sdk_error(error) from error
    return {
        "text": result.text,
        "provider": result.provider_used,
        "model": result.model_used,
        "attempts": [asdict(attempt) for attempt in result.attempts],
    }
