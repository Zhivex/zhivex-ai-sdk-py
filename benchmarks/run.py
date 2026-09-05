"""Seeded synthetic cache/gateway benchmark and bounded Postgres fencing soak."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time
import tracemalloc
import uuid
import zipfile

from zhivex_ai import (
    GatewayConfig,
    GatewayMessage,
    GatewayModelTarget,
    create_cached_generate_middleware,
    create_gateway,
    create_in_memory_generate_cache,
    generate_text,
    wrap_language_model,
)
from zhivex_ai.evals import GenerateResult, create_mock_language_model
from zhivex_ai.workflows import create_postgres_workflow_lease_manager


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * quantile) - 1)]


def check_budget(report: dict, budget: dict, *, require_postgres: bool) -> list[str]:
    failures = []
    for name in ("cache", "gateway"):
        item = report[name]
        if not math.isfinite(item["p95_ms"]) or item["p95_ms"] > budget["max_p95_ms"]:
            failures.append(name + "_latency")
        if item["peak_bytes"] > budget["max_peak_bytes"]:
            failures.append(name + "_memory")
    if report["cache"]["max_downstream_calls"] != budget["downstream_calls_per_round"]:
        failures.append("cache_duplication")
    soak = report["postgres"]
    if require_postgres and soak["status"] != "passed":
        failures.append("postgres_missing")
    if soak["status"] == "passed" and (
        soak["max_winners"] != budget["lease_winners_per_round"]
        or not soak["recovery_consistent"]
    ):
        failures.append("postgres_fencing")
    return failures


async def microbench(rounds: int, concurrency: int, seed: int) -> dict:
    rng = random.Random(seed)
    report = {}
    for area in ("cache", "gateway"):
        durations = []
        calls_per_round = []
        tracemalloc.start()
        start = time.perf_counter()
        for index in range(rounds + 3):
            calls = 0
            base = create_mock_language_model()

            class Model:
                provider = "openai"
                model_id = "synthetic"
                capabilities = base.capabilities

                async def generate(self, _input):
                    nonlocal calls
                    calls += 1
                    await asyncio.sleep(
                        0
                    )  # overlap identical requests, no provider latency
                    return GenerateResult(text="ok", finish_reason="stop")

            model = Model()
            prompt = f"synthetic-{rng.randrange(1_000_000)}"
            before = time.perf_counter()
            if area == "cache":
                wrapped = wrap_language_model(
                    model,
                    [
                        create_cached_generate_middleware(
                            cache=create_in_memory_generate_cache()
                        )
                    ],
                )
                results = await asyncio.gather(
                    *(
                        generate_text(model=wrapped, prompt=prompt)
                        for _ in range(concurrency)
                    )
                )
                assert all(result.text == "ok" for result in results)
            else:

                class Adapter:
                    def language_model(self, _name):
                        return model

                gateway = create_gateway(
                    GatewayConfig(adapters={"openai": Adapter()}, max_retries=0)
                )
                await gateway.generate(
                    messages=[GatewayMessage(role="user", content=prompt)],
                    primary=GatewayModelTarget(provider="openai", model_id="synthetic"),
                )
            if index >= 3:
                durations.append((time.perf_counter() - before) * 1000)
                calls_per_round.append(calls)
        elapsed = sum(durations) / 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report[area] = {
            "p50_ms": statistics.median(durations),
            "p95_ms": percentile(durations, 0.95),
            "throughput_requests_per_second": rounds
            * (concurrency if area == "cache" else 1)
            / elapsed,
            "peak_bytes": peak,
            "max_downstream_calls": max(calls_per_round),
            "wall_seconds_with_warmup": time.perf_counter() - start,
        }
    return report


async def soak(dsn: str | None, rounds: int, concurrency: int) -> dict:
    if not dsn:
        return {"status": "blocked", "reason": "postgres_not_configured"}
    prefix = "zhivex_bench_" + uuid.uuid4().hex[:12]
    managers = [
        create_postgres_workflow_lease_manager(dsn, table_prefix=prefix)
        for _ in range(concurrency)
    ]
    max_winners = 0
    latencies = []
    try:
        for index in range(rounds):
            run_id = f"synthetic-{index}"
            before = time.perf_counter()
            leases = await asyncio.gather(
                *(
                    manager.acquire(
                        run_id, owner_id=f"worker-{worker}", ttl_ms=100, now_ms=1000
                    )
                    for worker, manager in enumerate(managers)
                )
            )
            winners = [lease for lease in leases if lease is not None]
            max_winners = max(max_winners, len(winners))
            assert len(winners) == 1
            first = winners[0]
            # Inject owner disappearance: do not release; move logical lease time beyond TTL.
            replacement = await managers[0].acquire(
                run_id, owner_id="recovery", ttl_ms=100, now_ms=1101
            )
            assert (
                replacement is not None
                and replacement.fencing_token > first.fencing_token
            )
            assert not await managers[0].release(run_id, token=first.token)
            assert (
                await managers[0].renew(
                    run_id, token=first.token, ttl_ms=100, now_ms=1102
                )
                is None
            )
            assert await managers[0].release(run_id, token=replacement.token)
            latencies.append((time.perf_counter() - before) * 1000)
        return {
            "status": "passed",
            "rounds": rounds,
            "concurrency": concurrency,
            "max_winners": max_winners,
            "recovery_consistent": True,
            "fault": "owner_disappearance_logical_ttl",
            "p50_ms": statistics.median(latencies),
            "p95_ms": percentile(latencies, 0.95),
        }
    finally:
        await asyncio.gather(*(manager.close() for manager in managers))
        import asyncpg

        connection = await asyncpg.connect(dsn)
        try:
            # Prefix is generated here, never supplied by a caller.
            await connection.execute(f'DROP TABLE IF EXISTS "{prefix}_workflow_leases"')
        finally:
            await connection.close()


async def benchmark(args) -> dict:
    distribution = importlib.metadata.distribution("zhivex-ai-sdk")
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith("zhivex_ai/") and not name.endswith("/"):
                if distribution.locate_file(name).read_bytes() != archive.read(name):
                    raise ValueError("Installed package does not match the supplied wheel")
    async with asyncio.timeout(180):
        report = await microbench(args.rounds, args.concurrency, args.seed)
        report["postgres"] = await soak(
            os.getenv("ZHIVEX_TEST_POSTGRES_DSN"), args.rounds, args.concurrency
        )
    report.update(
        schema_version=1,
        seed=args.seed,
        rounds=args.rounds,
        concurrency=args.concurrency,
        python=platform.python_version(),
        platform=platform.system(),
        sdk_version=importlib.metadata.version("zhivex-ai-sdk"),
        commit=args.commit,
        wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
    )
    report["failures"] = check_budget(
        report,
        json.loads(args.budget.read_text()),
        require_postgres=args.require_postgres,
    )
    report["status"] = "failed" if report["failures"] else "passed"
    return report


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Benchmark verification requires Python assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--require-postgres", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 1000 or not 2 <= args.concurrency <= 64:
        parser.error("rounds must be 1..1000 and concurrency 2..64")
    try:
        result = asyncio.run(benchmark(args))
    except Exception as error:
        result = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "passed" else 1)
