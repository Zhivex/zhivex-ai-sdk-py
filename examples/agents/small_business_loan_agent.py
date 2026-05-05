from __future__ import annotations

import asyncio
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    Agent,
    AgentRunStore,
    SequentialAgent,
    WorkflowRunResult,
    WorkflowStep,
    create_agent_session,
    create_agent_trace_artifact,
    create_in_memory_agent_run_store,
    create_mock_language_model,
    replay_agent_run,
    run_workflow,
)

LoanProcessStatus = Literal["active", "needs_repair", "pending_approval", "completed", "failed"]


class LoanApplicationData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_name: str
    owner_name: str
    years_in_business: int
    annual_revenue: float
    requested_term_months: int
    credit_score: int
    existing_debt: float = 0
    loan_amount_requested: float | None = None


class UnderwritingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eligible: bool
    risk_tier: Literal["tier_1", "tier_2", "tier_3", "declined"]
    reasons: list[str]
    debt_to_revenue_ratio: float


class PricingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annual_rate: float
    monthly_payment: float
    total_interest: float


class LoanDecisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    decision_letter_id: str
    summary: str


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    findings: list[str] = []


@dataclass(slots=True)
class LoanProcessRecord:
    process_id: str
    status: LoanProcessStatus = "active"
    steps: dict[str, dict[str, object]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    workflow_run_ids: list[str] = field(default_factory=list)
    approval_requested: bool = False
    approved_by_user: bool | None = None


class LoanProcessStore(Protocol):
    async def load(self, process_id: str) -> LoanProcessRecord | None: ...

    async def save(self, record: LoanProcessRecord) -> None: ...

    async def apply_repair(self, process_id: str, updates: dict[str, object]) -> LoanProcessRecord: ...


class InMemoryLoanProcessStore:
    def __init__(self) -> None:
        self._records: dict[str, LoanProcessRecord] = {}

    async def load(self, process_id: str) -> LoanProcessRecord | None:
        record = self._records.get(process_id)
        return _clone_record(record) if record is not None else None

    async def save(self, record: LoanProcessRecord) -> None:
        self._records[record.process_id] = _clone_record(record)

    async def apply_repair(self, process_id: str, updates: dict[str, object]) -> LoanProcessRecord:
        record = await self.load(process_id)
        if record is None:
            raise KeyError(f'Loan process "{process_id}" does not exist.')
        extraction = record.steps.get("document_extraction", {})
        application = dict(extraction.get("data") or {})
        application.update(updates)
        record.steps["document_extraction"] = {
            "status": "completed",
            "data": application,
        }
        record.status = "active"
        record.issues = []
        await self.save(record)
        return record


@dataclass(slots=True)
class LoanScenarioResult:
    record: LoanProcessRecord
    workflow_results: list[WorkflowRunResult]


SAMPLE_COMPLETE_APPLICATION = {
    "business_name": "Cymbal Coffee Roasters LLC",
    "owner_name": "Jane Doe",
    "years_in_business": 6,
    "annual_revenue": 850_000,
    "requested_term_months": 60,
    "credit_score": 742,
    "existing_debt": 75_000,
    "loan_amount_requested": 150_000,
}

SAMPLE_INCOMPLETE_APPLICATION = {
    "business_name": "Cymbal Coffee Roasters LLC",
    "owner_name": "Jane Doe",
    "years_in_business": 6,
    "annual_revenue": 850_000,
    "requested_term_months": 60,
    "credit_score": 742,
    "existing_debt": 75_000,
}


def _clone_record(record: LoanProcessRecord) -> LoanProcessRecord:
    return LoanProcessRecord(
        process_id=record.process_id,
        status=record.status,
        steps=copy.deepcopy(record.steps),
        issues=list(record.issues),
        workflow_run_ids=list(record.workflow_run_ids),
        approval_requested=record.approval_requested,
        approved_by_user=record.approved_by_user,
    )


def _model_agent(name: str) -> Agent:
    return Agent(name=name, model=create_mock_language_model())


async def _run_sdk_workflow(
    *,
    name: str,
    steps: list[tuple[str, str]],
    run_store: AgentRunStore,
) -> WorkflowRunResult:
    workflow = SequentialAgent(
        name=name,
        run_store=run_store,
        steps=[
            WorkflowStep(step_name, _model_agent(agent_name), prompt=f"Run {step_name}", output_key=step_name)
            for step_name, agent_name in steps
        ],
    )
    return await run_workflow(workflow, session=create_agent_session())


def extract_application(payload: dict[str, object]) -> tuple[LoanApplicationData, list[str]]:
    application = LoanApplicationData.model_validate(payload)
    missing = ["loan_amount_requested"] if application.loan_amount_requested is None else []
    return application, missing


def underwrite(application: LoanApplicationData) -> UnderwritingReport:
    requested = application.loan_amount_requested or 0
    debt_ratio = round((application.existing_debt + requested) / application.annual_revenue, 4)
    reasons: list[str] = []
    if application.years_in_business < 2:
        reasons.append("Business must have at least 2 years of operating history.")
    if application.credit_score < 640:
        reasons.append("Owner credit score is below the approval threshold.")
    if debt_ratio > 0.45:
        reasons.append("Debt-to-revenue ratio exceeds policy threshold.")

    eligible = not reasons
    if not eligible:
        risk_tier: Literal["tier_1", "tier_2", "tier_3", "declined"] = "declined"
    elif application.credit_score >= 720 and debt_ratio <= 0.30:
        risk_tier = "tier_1"
    elif application.credit_score >= 680 and debt_ratio <= 0.38:
        risk_tier = "tier_2"
    else:
        risk_tier = "tier_3"

    return UnderwritingReport(
        eligible=eligible,
        risk_tier=risk_tier,
        reasons=reasons or ["Application meets baseline underwriting policy."],
        debt_to_revenue_ratio=debt_ratio,
    )


def price_loan(application: LoanApplicationData, underwriting: UnderwritingReport) -> PricingResult:
    if not underwriting.eligible or application.loan_amount_requested is None:
        raise ValueError("Pricing requires an eligible application with a requested amount.")

    rates = {"tier_1": 0.065, "tier_2": 0.0825, "tier_3": 0.109}
    annual_rate = rates[underwriting.risk_tier]
    monthly_rate = annual_rate / 12
    months = application.requested_term_months
    principal = application.loan_amount_requested
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    total_interest = monthly_payment * months - principal
    return PricingResult(
        annual_rate=annual_rate,
        monthly_payment=round(monthly_payment, 2),
        total_interest=round(total_interest, 2),
    )


def decide_loan(process_id: str, approved: bool) -> LoanDecisionResult:
    return LoanDecisionResult(
        approved=approved,
        decision_letter_id=f"DL-{process_id}-001",
        summary="Loan approved by reviewer." if approved else "Loan declined by reviewer.",
    )


def validate_decision(record: LoanProcessRecord) -> ValidationReport:
    required = ["document_extraction", "underwriting", "pricing", "loan_decision"]
    missing = [step for step in required if step not in record.steps]
    findings = [f"Missing required step: {step}" for step in missing]
    if record.steps.get("underwriting", {}).get("data", {}).get("eligible") is False and record.steps.get("loan_decision", {}).get("data", {}).get("approved") is True:
        findings.append("Decision approved a loan that underwriting marked ineligible.")
    return ValidationReport(valid=not findings, findings=findings)


async def start_loan_process(
    *,
    process_id: str,
    application_payload: dict[str, object],
    process_store: LoanProcessStore,
    run_store: AgentRunStore | None = None,
) -> LoanScenarioResult:
    sdk_run_store = run_store or create_in_memory_agent_run_store()
    record = await process_store.load(process_id) or LoanProcessRecord(process_id=process_id)
    workflow_results: list[WorkflowRunResult] = []

    extraction_result = await _run_sdk_workflow(
        name="loan_document_intake",
        steps=[("document_extraction", "document_extraction_agent")],
        run_store=sdk_run_store,
    )
    workflow_results.append(extraction_result)
    record.workflow_run_ids.append(extraction_result.run_id)

    application, missing = extract_application(application_payload)
    record.steps["document_extraction"] = {
        "status": "needs_repair" if missing else "completed",
        "data": application.model_dump(),
    }
    if missing:
        record.status = "needs_repair"
        record.issues = [f"Missing required field: {field}" for field in missing]
        await process_store.save(record)
        return LoanScenarioResult(record=record, workflow_results=workflow_results)

    resumed = await resume_loan_process(process_id=process_id, process_store=process_store, run_store=sdk_run_store, record=record)
    return LoanScenarioResult(record=resumed.record, workflow_results=workflow_results + resumed.workflow_results)


async def resume_loan_process(
    *,
    process_id: str,
    process_store: LoanProcessStore,
    run_store: AgentRunStore | None = None,
    record: LoanProcessRecord | None = None,
) -> LoanScenarioResult:
    sdk_run_store = run_store or create_in_memory_agent_run_store()
    resolved = record or await process_store.load(process_id)
    if resolved is None:
        raise KeyError(f'Loan process "{process_id}" does not exist.')
    workflow_results: list[WorkflowRunResult] = []

    if "document_extraction" not in resolved.steps or resolved.steps["document_extraction"].get("status") != "completed":
        resolved.status = "needs_repair"
        resolved.issues = resolved.issues or ["Document extraction must be completed before underwriting."]
        await process_store.save(resolved)
        return LoanScenarioResult(record=resolved, workflow_results=workflow_results)

    if "pricing" not in resolved.steps:
        underwriting_result = await _run_sdk_workflow(
            name="loan_underwriting_and_pricing",
            steps=[("underwriting", "underwriting_agent"), ("pricing", "pricing_agent")],
            run_store=sdk_run_store,
        )
        workflow_results.append(underwriting_result)
        resolved.workflow_run_ids.append(underwriting_result.run_id)
        application = LoanApplicationData.model_validate(resolved.steps["document_extraction"]["data"])
        underwriting = underwrite(application)
        resolved.steps["underwriting"] = {"status": "completed", "data": underwriting.model_dump()}
        if not underwriting.eligible:
            resolved.status = "failed"
            resolved.issues = underwriting.reasons
            await process_store.save(resolved)
            return LoanScenarioResult(record=resolved, workflow_results=workflow_results)
        pricing = price_loan(application, underwriting)
        resolved.steps["pricing"] = {"status": "completed", "data": pricing.model_dump()}

    resolved.status = "pending_approval"
    resolved.approval_requested = True
    await process_store.save(resolved)
    return LoanScenarioResult(record=resolved, workflow_results=workflow_results)


async def finalize_loan_process(
    *,
    process_id: str,
    approved: bool,
    process_store: LoanProcessStore,
    run_store: AgentRunStore | None = None,
) -> LoanScenarioResult:
    sdk_run_store = run_store or create_in_memory_agent_run_store()
    record = await process_store.load(process_id)
    if record is None:
        raise KeyError(f'Loan process "{process_id}" does not exist.')
    if record.status != "pending_approval":
        raise ValueError(f'Loan process "{process_id}" is not waiting for approval.')

    decision_result = await _run_sdk_workflow(
        name="loan_decision_and_validation",
        steps=[("loan_decision", "loan_decision_agent"), ("validation", "validation_judge_agent")],
        run_store=sdk_run_store,
    )
    record.workflow_run_ids.append(decision_result.run_id)
    record.approved_by_user = approved
    decision = decide_loan(process_id, approved)
    record.steps["loan_decision"] = {"status": "completed", "data": decision.model_dump()}
    validation = validate_decision(record)
    record.steps["validation"] = {"status": "completed", "data": validation.model_dump()}
    record.status = "completed" if validation.valid else "failed"
    record.issues = validation.findings
    await process_store.save(record)
    return LoanScenarioResult(record=record, workflow_results=[decision_result])


def summarize(record: LoanProcessRecord, workflow_results: list[WorkflowRunResult]) -> dict[str, object]:
    replay_events: list[str] = []
    trace_previews: list[str] = []
    for result in workflow_results:
        if result.state_snapshot is not None:
            trace_previews.append(create_agent_trace_artifact(result.state_snapshot).output_preview)
            replay_events.extend(event.type for event in replay_agent_run(result.state_snapshot).timeline)
    return {
        "process_id": record.process_id,
        "status": record.status,
        "steps": list(record.steps),
        "issues": record.issues,
        "approval_requested": record.approval_requested,
        "approved_by_user": record.approved_by_user,
        "workflow_runs": record.workflow_run_ids,
        "trace_previews": trace_previews,
        "replay_events": replay_events,
    }


async def demo_complete_application() -> dict[str, object]:
    process_store = InMemoryLoanProcessStore()
    run_store = create_in_memory_agent_run_store()
    started = await start_loan_process(
        process_id="SBL-2025-00142",
        application_payload=SAMPLE_COMPLETE_APPLICATION,
        process_store=process_store,
        run_store=run_store,
    )
    finalized = await finalize_loan_process(
        process_id="SBL-2025-00142",
        approved=True,
        process_store=process_store,
        run_store=run_store,
    )
    workflows = started.workflow_results + finalized.workflow_results
    return summarize(finalized.record, workflows)


async def demo_repair_and_resume() -> dict[str, object]:
    process_store = InMemoryLoanProcessStore()
    run_store = create_in_memory_agent_run_store()
    stopped = await start_loan_process(
        process_id="SBL-2025-00391",
        application_payload=SAMPLE_INCOMPLETE_APPLICATION,
        process_store=process_store,
        run_store=run_store,
    )
    await process_store.apply_repair("SBL-2025-00391", {"loan_amount_requested": 150_000})
    resumed = await resume_loan_process(
        process_id="SBL-2025-00391",
        process_store=process_store,
        run_store=run_store,
    )
    finalized = await finalize_loan_process(
        process_id="SBL-2025-00391",
        approved=True,
        process_store=process_store,
        run_store=run_store,
    )
    workflows = stopped.workflow_results + resumed.workflow_results + finalized.workflow_results
    return summarize(finalized.record, workflows)


async def main() -> None:
    complete = await demo_complete_application()
    repaired = await demo_repair_and_resume()

    print("Complete application")
    print(complete)
    print("Repair and resume")
    print(repaired)


if __name__ == "__main__":
    asyncio.run(main())
