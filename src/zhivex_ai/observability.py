from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _SpanHandle:
    manager: Any
    span: Any

    def end(self, *, attributes: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        if self.span is not None and attributes:
            for key, value in attributes.items():
                self.span.set_attribute(key, value)
        if self.span is not None and error is not None:
            self.span.record_exception(error)
            try:
                from opentelemetry.trace import Status, StatusCode

                self.span.set_status(Status(StatusCode.ERROR, str(error)))
            except Exception:
                pass
        self.manager.__exit__(type(error) if error is not None else None, error, getattr(error, "__traceback__", None))


class OTelAgentObserver:
    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> _SpanHandle:
        manager = self._tracer.start_as_current_span(name)
        span = manager.__enter__()
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        return _SpanHandle(manager=manager, span=span)


def create_otel_agent_observer(*, tracer_name: str = "zhivex_ai.agent", version: str | None = None) -> OTelAgentObserver:
    try:
        from opentelemetry import trace
    except Exception as error:
        raise RuntimeError("OpenTelemetry is not installed. Install opentelemetry-api/sdk to use OTEL observability.") from error
    tracer = trace.get_tracer(tracer_name, version)
    return OTelAgentObserver(tracer)
