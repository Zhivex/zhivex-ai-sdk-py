# Agent Evaluations

Zhivex AI SDK `0.16.0` adds beta experiment primitives on top of the existing deterministic agent evaluation helpers.

Use them to compare application-owned agent variants, calculate custom metrics, and fail CI when a candidate misses an absolute threshold or regresses against a baseline. They do not replace domain review, production monitoring, or live provider smoke tests.

## Single-variant evaluation

```python
from zhivex_ai import (
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

result = await run_agent_evaluation(agent=agent, dataset=dataset)
```

## Baseline experiment and CI gates

```python
from zhivex_ai import (
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
)

print(experiment.to_json())
raise SystemExit(0 if experiment.ok else 1)
```

Variants, cases, and metrics run in supplied order. A custom scorer may be synchronous or asynchronous; its case scores are averaged. `higher_is_better=False` reverses only regression direction. Explicit `minimum` and `maximum` gates retain their literal meanings.

`to_dict()` and `to_json()` validate the complete artifact and reject `NaN`, infinity, non-string mapping keys, and non-JSON metadata. This prevents a CI job from publishing a partially valid result.

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
zhivex eval my_app.agents:support_agent --dataset evals/support.json
```

The command exits `0` when the report passes and `1` when an expectation fails. Python module loading executes trusted application code; do not load unreviewed agent modules.

## Operational boundary

- Keep datasets, expected behavior, and judge prompts under normal code-review controls.
- Redact production data before placing it in fixtures or result artifacts.
- Do not use one model judge as the only approval for regulated, safety-sensitive, or financially material behavior.
- Pin provider/model configuration when comparing variants and record it in experiment metadata.
- Run provider-backed experiments in a credentialed integration environment; offline mock results prove contracts, not live quality.
