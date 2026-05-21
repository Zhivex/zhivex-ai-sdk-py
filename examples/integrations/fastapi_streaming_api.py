from __future__ import annotations

import math
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from zhivex_ai import (
    ConfigurationError,
    ParseError,
    ProviderHTTPError,
    UnsupportedFeatureError,
    ValidationError,
    create_openai,
    stream_text,
    to_text_stream_response,
    to_ui_message_stream_response,
)

app = FastAPI(title="Zhivex AI SDK FastAPI Streaming Example")


class StreamRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system: str | None = None
    model: str = "gpt-5.4-mini"
    timeout_ms: int = Field(default=30_000, gt=0)


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


def _to_fastapi_stream(response: Any) -> StreamingResponse:
    return StreamingResponse(
        response.body,
        status_code=response.status_code,
        headers=response.headers,
        media_type=response.headers.get("content-type"),
    )


@app.post("/v1/chat/stream")
async def stream_chat(request: StreamRequest) -> StreamingResponse:
    provider = create_openai()
    try:
        result = stream_text(
            model=provider(request.model),
            prompt=request.prompt,
            system=request.system,
            timeout_ms=request.timeout_ms,
        )
        return _to_fastapi_stream(to_text_stream_response(result))
    except Exception as error:
        raise _http_exception_from_sdk_error(error) from error


@app.post("/v1/chat/ui-stream")
async def stream_chat_ui(request: StreamRequest) -> StreamingResponse:
    provider = create_openai()
    try:
        result = stream_text(
            model=provider(request.model),
            prompt=request.prompt,
            system=request.system,
            timeout_ms=request.timeout_ms,
        )
        return _to_fastapi_stream(to_ui_message_stream_response(result))
    except Exception as error:
        raise _http_exception_from_sdk_error(error) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "examples.integrations.fastapi_streaming_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
