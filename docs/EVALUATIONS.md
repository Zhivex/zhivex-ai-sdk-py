# Agent Evaluations

Zhivex AI SDK `0.16.0` adds beta experiment primitives on top of the existing deterministic agent evaluation helpers.

Use them to compare application-owned agent variants, calculate custom metrics, and fail CI when a candidate misses an absolute threshold or regresses against a baseline. They do not replace domain review, production monitoring, or live provider smoke tests.

Import evaluation contracts from the focused public namespace `zhivex_ai.evals`. Existing top-level imports remain available for compatibility, but new code should make the Beta dependency explicit.

## Single-variant evaluation

```python
from zhivex_ai.evals import (
    AgentEvaluationCase,
    AgentEvaluationExpectations,
    run_agent_evaluation,
)

dataset = [
    AgentEvaluationCase(
        name="refund-policy",
        prompt="Can this order be refunded?",
        expectations=AgentEvaluationExpectations(output_contains="review"),
    )
]

result = await run_agent_evaluation(
    agent=agent_factory,
    dataset=dataset,
    repetitions=5,
    max_concurrency=4,
    cost_estimator=estimate_run_cost,
)
```

Each case produces ordered `AgentEvaluationTrialResult` records. Trials capture
monotonic latency, normalized token usage, optional application-calculated cost,
expectation failures, and a redacted trajectory containing orchestration and
event names only. `trial_pass_rate` includes a two-sided Wilson 95% confidence
interval (`trial_pass_rate_ci95_lower` and
`trial_pass_rate_ci95_upper`). Mean and p95 latency, sample latency standard
deviation (`latency_stddev_ms`), total/mean cost, and mean total tokens are
included when their inputs are available. Dataset names must be non-empty and
unique, and execution limits must be positive integers.

The Wilson interval remains finite for all-pass and all-fail samples and reports
`[0, 1]` when there are no trials. It quantifies sampling uncertainty in the
aggregate observed pass rate; it does not make trials independent or correct
for a dataset that mixes materially different tasks. Keep per-case failures and
sample size visible when interpreting it.

Concurrency is bounded by a real semaphore while output remains ordered by
dataset and repetition. Prefer an `AgentEvaluationAgentFactory` when
`max_concurrency > 1`; reusing one `Agent` instance is safe only when its model,
memory, tools, and application dependencies are reentrant.

## Baseline experiment and CI gates

```python
from zhivex_ai.evals import (
    AgentEvaluationGate,
    AgentEvaluationMetric,
    run_agent_evaluation_experiment,
)

experiment = await run_agent_evaluation_experiment(
    variants={"baseline": baseline_agent, "candidate": candidate_agent},
    baseline="baseline",
    dataset=dataset,
    metrics=[
        AgentEvaluationMetric(
            "answer_length",
            lambda case: len(case.output.text) if case.output else 0,
            higher_is_better=False,
        )
    ],
    gates=[
        AgentEvaluationGate("pass_rate", minimum=0.95, max_regression=0.01),
        AgentEvaluationGate("answer_length", maximum=1_000, max_regression=100),
    ],
    repetitions=5,
    max_concurrency=4,
)

print(experiment.to_json())
print(experiment.to_junit_xml())
raise SystemExit(0 if experiment.ok else 1)
```

Variants, cases, and metrics run in supplied order. A custom scorer may be synchronous or asynchronous; its case scores are averaged. `higher_is_better=False` reverses only regression direction. Explicit `minimum` and `maximum` gates retain their literal meanings.

`to_dict()` and `to_json()` emit `schema_version: 1`, validate the complete
artifact, and reject `NaN`, infinity, non-string mapping keys, and non-JSON
metadata. `to_junit_xml()` emits trial-level test cases suitable for common CI
test-report collectors. This prevents a CI job from publishing a partially
valid result.

## Deterministic scoring and custom judges

`judge_agent_evaluation(result)` is a compatibility name for the built-in
deterministic expectation scorer. It does not call an LLM: its score is the case
pass rate produced from `AgentEvaluationExpectations`, and its metadata reports
`judge_type: "deterministic_expectations"`.

Pass an application-owned synchronous or asynchronous `judge` callable when a
structured rubric or model judge is needed. The SDK stays provider-agnostic and
returns that callable's `AgentEvaluationJudgeResult` unchanged. Record rubric,
provider, model, and version information in its metadata, validate structured
model output in application code, and do not treat one model score as the only
approval for high-impact behavior.

## Curating datasets from traces

`create_agent_evaluation_dataset_from_traces(...)` converts persisted
`AgentRunState` records into cases only through application-owned prompt and
expectation extractors. It adds source run/provider/model provenance, but never
copies prompts or outputs implicitly. Use the optional name and metadata
extractors after applying consent, DLP, retention, and tenant-isolation policy.

## Dataset CLI

The general CLI accepts a JSON list or `{ "cases": [...] }`:

```json
{
  "cases": [
    {
      "name": "greeting",
      "prompt": "Say hello",
      "expectations": {"output_contains": "hello"}
    }
  ]
}
```

```bash
zhivex eval my_app.agents:support_agent \
  --dataset evals/support.json \
  --repetitions 5 \
  --max-concurrency 4 \
  --min-pass-rate 0.95 \
  --max-mean-latency-ms 3000 \
  --output-json artifacts/evaluation.json \
  --output-junit artifacts/evaluation.xml
```

Artifact writes are atomic. The command exits `0` when expectations and gates
pass, `1` when an expectation or gate fails, and `2` for invalid configuration,
dataset, imports, or I/O. Python module loading executes trusted application
code; do not load unreviewed agent modules.

## Operational boundary

- Keep datasets, expected behavior, and judge prompts under normal code-review controls.
- Redact production data before placing it in fixtures or result artifacts.
- Treat the historical `output_preview` field as content-bearing. The new
  trajectory is redacted by construction, but artifact access and retention
  still require an application policy.
- Do not use one model judge as the only approval for regulated, safety-sensitive, or financially material behavior.
- Pin provider/model configuration when comparing variants and record it in experiment metadata.
- Run provider-backed experiments in a credentialed integration environment; offline mock results prove contracts, not live quality.
