from __future__ import annotations

import math

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from zhivex_ai import (
    ConfigurationError,
    ParseError,
    ProviderHTTPError,
    UnsupportedFeatureError,
    ValidationError,
    create_openai,
    generate_text,
)

app = FastAPI(title="Zhivex AI SDK FastAPI Chat Example")


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system: str | None = None
    model: str = "gpt-5.4-mini"
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_ms: int = Field(default=30_000, gt=0)


class ChatResponse(BaseModel):
    text: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, int | None] | None = None


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
                "message": str(error),
                "provider_status": error.status,
                "retryable": error.retryable,
            },
            headers=headers,
        )
    return HTTPException(status_code=500, detail="Internal server error.")


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    provider = create_openai()
    try:
        result = await generate_text(
            model=provider(request.model),
            prompt=request.prompt,
            system=request.system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            timeout_ms=request.timeout_ms,
        )
    except Exception as error:
        raise _http_exception_from_sdk_error(error) from error

    usage: dict[str, int | None] | None = None
    if result.usage is not None:
        usage = {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        }

    return ChatResponse(
        text=result.text,
        model=request.model,
        finish_reason=result.finish_reason,
        usage=usage,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("examples.integrations.fastapi_chat_api:app", host="127.0.0.1", port=8000, reload=True)
