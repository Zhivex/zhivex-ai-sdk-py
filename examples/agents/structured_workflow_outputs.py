from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import Agent, GenerateResult, SequentialAgent, WorkflowStep, create_mock_language_model


class LoanIntake(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str
    requested_amount: int
    monthly_revenue: int


class RiskReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: str
    max_offer: int


@dataclass(slots=True)
class StructuredWorkflowSummary:
    company: str
    rating: str
    max_offer: int
    state_keys: list[str]


def _json_agent(name: str, payload: dict[str, object]) -> Agent:
    return Agent(
        name=name,
        model=create_mock_language_model(
            responses=[
                GenerateResult(
                    text=json.dumps(payload, sort_keys=True),
                    finish_reason="stop",
                )
            ]
        ),
    )


async def run_structured_workflow_demo() -> StructuredWorkflowSummary:
    workflow = SequentialAgent(
        name="structured_loan_review",
        steps=[
            WorkflowStep(
                "intake",
                _json_agent(
                    "intake_agent",
                    {
                        "company": "Apollo Tools",
                        "requested_amount": 150_000,
                        "monthly_revenue": 95_000,
                    },
                ),
                prompt="Extract the loan request.",
                output_key="intake_json",
            ),
            WorkflowStep(
                "risk",
                _json_agent("risk_agent", {"rating": "low", "max_offer": 125_000}),
                input_template="Review {intake_json}.",
                output_key="risk_json",
            ),
        ],
    )

    result = await workflow.run()
    intake = LoanIntake.model_validate_json(str(result.state["intake_json"]))
    risk = RiskReview.model_validate_json(str(result.state["risk_json"]))

    return StructuredWorkflowSummary(
        company=intake.company,
        rating=risk.rating,
        max_offer=risk.max_offer,
        state_keys=sorted(result.state),
    )


async def main() -> None:
    summary = await run_structured_workflow_demo()
    print(
        {
            "company": summary.company,
            "rating": summary.rating,
            "max_offer": summary.max_offer,
            "state_keys": summary.state_keys,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
