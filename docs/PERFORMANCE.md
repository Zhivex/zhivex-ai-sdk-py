# Reproducible performance evidence

HU17 measures synthetic SDK overhead, not provider latency or model quality.
`benchmarks/run.py` records runtime, platform, wheel SHA256, source commit, seed,
p50/p95, throughput and tracemalloc peak bytes. Memory is Python allocation peak,
not RSS. Cache timing is one concurrent batch; gateway timing is one request.
Three warmup rounds are excluded from percentiles and throughput.

Install the exact candidate wheel with the `postgres` extra in a clean venv, then:

```bash
/path/to/consumer/bin/python -I benchmarks/run.py --rounds 20 --concurrency 16 --seed 14 --budget benchmarks/budgets.json --wheel /absolute/candidate.whl --commit SOURCE_COMMIT --require-postgres --output /tmp/performance.json
```

Set `ZHIVEX_TEST_POSTGRES_DSN` to a disposable Postgres database. Only generated
`zhivex_bench_*` lease tables are used and removed. The soak contests each lease,
requires one winner, simulates owner disappearance using a logical TTL, checks a
higher fencing token on replacement, rejects the stale owner's renewal/release,
and releases the current owner. This is a bounded lease/fencing scenario; it does
not claim real process-crash recovery of arbitrary external side effects. The
adoption smoke separately verifies approval/resume in new processes.

The versioned budget gates duplication, fencing, a conservative 1000 ms p95 ceiling
and 64 MiB allocation peak. It is a PR safety ceiling; compare trends on the same
dedicated host before enforcing tighter relative latency budgets. Shared-runner
measurements cannot establish universal capacity. Missing required Postgres fails;
optional missing Postgres remains explicitly blocked in JSON.

The Adoption evidence workflow runs the installed consumer and benchmark, retaining
reports even after failure. A manual run can use 200 soak rounds; all modes are
bounded to 180 seconds. No live provider credentials are used.
