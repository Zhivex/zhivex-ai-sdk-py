import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).parents[1] / "benchmarks/run.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("latency", "cache_latency"),
        ("memory", "gateway_memory"),
        ("duplication", "cache_duplication"),
        ("missing", "postgres_missing"),
        ("fencing", "postgres_fencing"),
        ("nan", "cache_latency"),
    ],
)
def test_budget_rejects_regression(mutation, expected):
    report = {
        "cache": {"p95_ms": 1, "peak_bytes": 1, "max_downstream_calls": 1},
        "gateway": {"p95_ms": 1, "peak_bytes": 1},
        "postgres": {"status": "passed", "max_winners": 1, "recovery_consistent": True},
    }
    budget = {
        "max_p95_ms": 10,
        "max_peak_bytes": 10,
        "downstream_calls_per_round": 1,
        "lease_winners_per_round": 1,
    }
    assert benchmark.check_budget(report, budget, require_postgres=True) == []
    if mutation == "latency":
        report["cache"]["p95_ms"] = 11
    if mutation == "memory":
        report["gateway"]["peak_bytes"] = 11
    if mutation == "duplication":
        report["cache"]["max_downstream_calls"] = 2
    if mutation == "missing":
        report["postgres"]["status"] = "blocked"
    if mutation == "fencing":
        report["postgres"]["recovery_consistent"] = False
    if mutation == "nan":
        report["cache"]["p95_ms"] = float("nan")
    assert expected in benchmark.check_budget(report, budget, require_postgres=True)
