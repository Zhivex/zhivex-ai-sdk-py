from __future__ import annotations

import math

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from zhivex_ai import (
    ConfigurationError,
    GatewayConfig,
    GatewayMessage,
    GatewayModelTarget,
    ParseError,
    ProviderHTTPError,
    UnsupportedFeatureError,
    ValidationError,
    create_anthropic,
    create_gateway,
    create_openai,
)

app = FastAPI(title="Zhivex AI SDK FastAPI Gateway Example")


class GatewayRequest(BaseModel):
    prompt: str = Field(min_length=1)
    primary_model: str = "gpt-5.6-terra"
    fallback_model: str = "claude-sonnet-5"
    timeout_ms: int = Field(default=30_000, gt=0)
    routing_mode: str = "balanced"
    task_intent: str = "chat"


class GatewayResponseBody(BaseModel):
    text: str
    provider_used: str
    model_used: str
    latency_ms: int
    finish_attempts: int


def _http_exception_from_sdk_error(error: Exception) -> HTTPException:
    if isinstance(error, (ValidationError, ParseError, UnsupportedFeatureError)):
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


@app.post("/v1/gateway/chat", response_model=GatewayResponseBody)
async def gateway_chat(request: GatewayRequest) -> GatewayResponseBody:
    gateway = create_gateway(
        GatewayConfig(
            adapters={
                "openai": create_openai(),
                "anthropic": create_anthropic(),
            },
            attempt_timeout_ms=request.timeout_ms,
        )
    )
    try:
        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content=request.prompt)],
            primary=GatewayModelTarget(provider="openai", model_id=request.primary_model),
            fallbacks=[
                GatewayModelTarget(
                    provider="anthropic",
                    model_id=request.fallback_model,
                )
            ],
            routing_mode=request.routing_mode,
            task_intent=request.task_intent,
        )
    except Exception as error:
        raise _http_exception_from_sdk_error(error) from error

    return GatewayResponseBody(
        text=result.text,
        provider_used=result.provider_used,
        model_used=result.model_used,
        latency_ms=result.latency_ms,
        finish_attempts=len(result.attempts),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "examples.integrations.fastapi_gateway_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
