from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from zhivex_ai import OTelAgentObserver


def test_observer_does_not_export_exception_payload_or_arbitrary_metadata():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observer = OTelAgentObserver(provider.get_tracer("test"))
    secret = "synthetic-private-payload"
    span = observer.start_span(
        "test", {"prompt": secret, "run.idempotency_key": secret, "run.id": "fixture"}
    )
    span.end(attributes={"response": secret}, error=ValueError(secret))
    exported = exporter.get_finished_spans()[0]
    assert secret not in exported.to_json()
    assert not exported.events
    assert exported.attributes["error.type"] == "ValueError"
    assert exported.status.status_code.name == "ERROR"
    provider.shutdown()


@pytest.mark.asyncio
async def test_recipe_correlates_operations_and_bounds_metric_labels():
    spec = importlib.util.spec_from_file_location(
        "recipe", Path(__file__).parents[1] / "examples/observability/otlp_recipe.py"
    )
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    metrics = MeterProvider(metric_readers=[reader])
    await recipe.exercise(provider.get_tracer("test"), metrics.get_meter("test"))
    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert {
        "application.request",
        "zhivex.generation",
        "zhivex.gateway.attempt",
        "zhivex.agent.run",
        "zhivex.agent.model",
    } <= names
    assert len({span.context.trace_id for span in spans}) == 1
    assert len([span for span in spans if span.name == "zhivex.agent.run"]) == 2
    for resource in reader.get_metrics_data().resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                for point in metric.data.data_points:
                    assert set(point.attributes) == {"operation", "outcome"}
                    assert point.attributes["operation"] in {
                        "generation",
                        "gateway",
                        "agent",
                    }
    provider.shutdown()
    metrics.shutdown()
