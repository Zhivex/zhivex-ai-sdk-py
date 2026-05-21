from __future__ import annotations

import math
import os
from dataclasses import asdict
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from zhivex_ai import (
    Agent,
    ConfigurationError,
    GatewayConfig,
    GatewayMessage,
    GatewayModelTarget,
    ProviderHTTPError,
    UnsupportedFeatureError,
    ValidationError,
    create_gateway,
    create_openai,
    create_otel_agent_observer,
    create_postgres_agent_run_store,
    run_agent,
)


app = FastAPI(title="Zhivex AI SDK Production Agent API")


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = "gpt-5.4-mini"
    idempotency_key: str | None = None


class AgentResponse(BaseModel):
    text: str
    run_id: str
    session_id: str
    request_id: str


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


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/agent", response_model=AgentResponse)
async def run_agent_endpoint(
    request: AgentRequest,
    request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> AgentResponse:
    resolved_request_id = request_id or f"req_{uuid4().hex}"
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise HTTPException(status_code=500, detail="DATABASE_URL is required for production run storage.")
    run_store = create_postgres_agent_run_store(dsn)
    provider = create_openai()
    agent = Agent(
        name="production_assistant",
        model=provider(request.model),
        run_store=run_store,
        metadata={"request_id": resolved_request_id},
    )

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
