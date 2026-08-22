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
    create_agent_session,
    create_agent_trace_artifact,
    create_in_memory_agent_run_store,
    replay_agent_run,
)
from zhivex_ai.evals import create_mock_language_model
from zhivex_ai.workflows import SequentialAgent, WorkflowRunResult, WorkflowStep, run_workflow

HiringProcessStatus = Literal[
    "active",
    "needs_repair",
    "pending_interview",
    "pending_recruiter_review",
    "completed",
    "rejected",
    "failed",
]


class JobDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    required_skills: list[str]
    preferred_skills: list[str] = []
    min_years_experience: int
    interview_focus: list[str] = []


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_name: str
    current_title: str
    skills: list[str]
    education: str | None = None
    years_experience: int | None = None
    work_samples: list[str] = []


class ScreeningReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match_score: float
    qualified: bool
    matched_skills: list[str]
    missing_skills: list[str]
    evidence: list[str]


class InterviewPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[str]
    focus_areas: list[str]


class InterviewEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technical_score: float
    communication_score: float
    evidence: list[str]
    concerns: list[str] = []


class HiringRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: Literal["advance", "hold", "reject"]
    confidence: float
    summary: str
    evidence: list[str]


class FairnessReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    findings: list[str] = []


@dataclass(slots=True)
class HiringProcessRecord:
    process_id: str
    status: HiringProcessStatus = "active"
    steps: dict[str, dict[str, object]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    workflow_run_ids: list[str] = field(default_factory=list)
    recruiter_review_requested: bool = False
    recruiter_approved: bool | None = None


class HiringProcessStore(Protocol):
    async def load(self, process_id: str) -> HiringProcessRecord | None: ...

    async def save(self, record: HiringProcessRecord) -> None: ...

    async def apply_repair(self, process_id: str, updates: dict[str, object]) -> HiringProcessRecord: ...


class InMemoryHiringProcessStore:
    def __init__(self) -> None:
        self._records: dict[str, HiringProcessRecord] = {}

    async def load(self, process_id: str) -> HiringProcessRecord | None:
        record = self._records.get(process_id)
        return _clone_record(record) if record is not None else None

    async def save(self, record: HiringProcessRecord) -> None:
        self._records[record.process_id] = _clone_record(record)

    async def apply_repair(self, process_id: str, updates: dict[str, object]) -> HiringProcessRecord:
        record = await self.load(process_id)
        if record is None:
            raise KeyError(f'Hiring process "{process_id}" does not exist.')
        intake = record.steps.get("resume_intake", {})
        profile = dict(intake.get("data") or {})
        profile.update(updates)
        record.steps["resume_intake"] = {
            "status": "completed",
            "data": profile,
        }
        record.status = "active"
        record.issues = []
        await self.save(record)
        return record


@dataclass(slots=True)
class HiringScenarioResult:
    record: HiringProcessRecord
    workflow_results: list[WorkflowRunResult]


SAMPLE_JOB = {
    "title": "Senior Backend Engineer",
    "required_skills": ["python", "apis", "postgres"],
    "preferred_skills": ["distributed systems", "observability"],
    "min_years_experience": 5,
    "interview_focus": ["system design", "incident response", "team communication"],
}

SAMPLE_STRONG_CANDIDATE = {
    "candidate_name": "Avery Rivera",
    "current_title": "Backend Engineer",
    "skills": ["python", "apis", "postgres", "observability", "distributed systems"],
    "education": "BS Computer Science",
    "years_experience": 7,
    "work_samples": ["Designed payment APIs", "Led incident reviews"],
}

SAMPLE_INCOMPLETE_CANDIDATE = {
    "candidate_name": "Sam Lee",
    "current_title": "Software Engineer",
    "skills": ["python", "apis", "postgres"],
    "education": "Bootcamp graduate",
}

SAMPLE_BIAS_RISK_CANDIDATE = {
    "candidate_name": "Bias Risk Candidate",
    "current_title": "Backend Engineer",
    "skills": ["python", "apis", "postgres", "observability"],
    "education": "MS Software Engineering",
    "years_experience": 8,
    "work_samples": ["Scaled hiring platform APIs"],
}

SAMPLE_STRONG_INTERVIEW_TRANSCRIPT = (
    "Candidate explains API design tradeoffs, Postgres indexing, incident follow-up, "
    "and gives specific examples of mentoring teammates."
)

SAMPLE_BIAS_RISK_INTERVIEW_TRANSCRIPT = (
    "Candidate answers the technical questions well and describes reliable API delivery."
)


def _clone_record(record: HiringProcessRecord) -> HiringProcessRecord:
    return HiringProcessRecord(
        process_id=record.process_id,
        status=record.status,
        steps=copy.deepcopy(record.steps),
        issues=list(record.issues),
        workflow_run_ids=list(record.workflow_run_ids),
        recruiter_review_requested=record.recruiter_review_requested,
        recruiter_approved=record.recruiter_approved,
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


def intake_candidate(payload: dict[str, object]) -> tuple[CandidateProfile, list[str]]:
    profile = CandidateProfile.model_validate(payload)
    missing: list[str] = []
    if profile.years_experience is None:
        missing.append("years_experience")
    if not profile.skills:
        missing.append("skills")
    return profile, missing


def screen_candidate(job: JobDescription, profile: CandidateProfile) -> ScreeningReport:
    required = {skill.lower() for skill in job.required_skills}
    candidate_skills = {skill.lower() for skill in profile.skills}
    matched = sorted(required & candidate_skills)
    missing = sorted(required - candidate_skills)
    skill_score = len(matched) / max(len(required), 1)
    experience_score = min((profile.years_experience or 0) / max(job.min_years_experience, 1), 1)
    match_score = round((skill_score * 0.7 + experience_score * 0.3) * 100, 2)
    qualified = match_score >= 75 and not missing and (profile.years_experience or 0) >= job.min_years_experience
    return ScreeningReport(
        match_score=match_score,
        qualified=qualified,
        matched_skills=matched,
        missing_skills=missing,
        evidence=[
            f"Matched required skills: {', '.join(matched) if matched else 'none'}.",
            f"Experience: {profile.years_experience or 0} years against {job.min_years_experience} required.",
        ],
    )


def create_interview_plan(job: JobDescription, screening: ScreeningReport) -> InterviewPlan:
    focus = job.interview_focus or ["role fit"]
    questions = [f"Describe a recent example involving {area}." for area in focus]
    for skill in screening.missing_skills:
        questions.append(f"How would you ramp up on {skill} for this role?")
    return InterviewPlan(questions=questions, focus_areas=focus + screening.missing_skills)


def evaluate_interview(transcript: str) -> InterviewEvaluation:
    lowered = transcript.lower()
    technical_hits = sum(1 for token in ("api", "postgres", "design", "incident", "scaled") if token in lowered)
    communication_hits = sum(1 for token in ("explains", "examples", "mentoring", "tradeoffs", "reliable") if token in lowered)
    technical_score = min(technical_hits / 5, 1)
    communication_score = min(communication_hits / 5, 1)
    concerns = [] if technical_score >= 0.5 else ["Transcript has limited role-specific technical evidence."]
    return InterviewEvaluation(
        technical_score=round(technical_score, 2),
        communication_score=round(communication_score, 2),
        evidence=[
            f"Technical signal score: {round(technical_score, 2)}.",
            f"Communication signal score: {round(communication_score, 2)}.",
        ],
        concerns=concerns,
    )


def recommend_candidate(profile: CandidateProfile, screening: ScreeningReport, evaluation: InterviewEvaluation) -> HiringRecommendation:
    combined = round((screening.match_score / 100 * 0.55 + evaluation.technical_score * 0.3 + evaluation.communication_score * 0.15), 2)
    if combined >= 0.75 and screening.qualified and not evaluation.concerns:
        recommendation: Literal["advance", "hold", "reject"] = "advance"
    elif combined >= 0.55:
        recommendation = "hold"
    else:
        recommendation = "reject"
    evidence = [*screening.evidence, *evaluation.evidence]
    summary = f"Recommend {recommendation} based on role evidence."
    if profile.candidate_name == "Bias Risk Candidate":
        summary = "Recommend advance because the candidate is young, energetic, and has strong role evidence."
        evidence.append("Unsafe note: young and energetic.")
    return HiringRecommendation(
        recommendation=recommendation,
        confidence=combined,
        summary=summary,
        evidence=evidence,
    )


def review_fairness(recommendation: HiringRecommendation) -> FairnessReview:
    prohibited_terms = ["age", "young", "old", "gender", "married", "nationality", "pregnant"]
    text = " ".join([recommendation.summary, *recommendation.evidence]).lower()
    findings = [f'Prohibited or non-job-related signal detected: "{term}".' for term in prohibited_terms if term in text]
    return FairnessReview(valid=not findings, findings=findings)


def recruiter_decision(process_id: str, approved: bool) -> dict[str, object]:
    return {
        "decision_id": f"HR-{process_id}-001",
        "approved": approved,
        "summary": "Recruiter approved advancing the candidate." if approved else "Recruiter rejected advancing the candidate.",
    }


async def start_hiring_process(
    *,
    process_id: str,
    job_payload: dict[str, object],
    candidate_payload: dict[str, object],
    process_store: HiringProcessStore,
    run_store: AgentRunStore | None = None,
) -> HiringScenarioResult:
    sdk_run_store = run_store or create_in_memory_agent_run_store()
    record = await process_store.load(process_id) or HiringProcessRecord(process_id=process_id)
    workflow_results: list[WorkflowRunResult] = []

    intake_result = await _run_sdk_workflow(
        name="hr_resume_intake",
        steps=[("resume_intake", "resume_extraction_agent")],
        run_store=sdk_run_store,
    )
    workflow_results.append(intake_result)
    record.workflow_run_ids.append(intake_result.run_id)

    profile, missing = intake_candidate(candidate_payload)
    record.steps["job_description"] = {"status": "completed", "data": JobDescription.model_validate(job_payload).model_dump()}
    record.steps["resume_intake"] = {
        "status": "needs_repair" if missing else "completed",
        "data": profile.model_dump(),
    }
    if missing:
        record.status = "needs_repair"
        record.issues = [f"Missing required field: {field}" for field in missing]
        await process_store.save(record)
        return HiringScenarioResult(record=record, workflow_results=workflow_results)

    resumed = await resume_hiring_process(process_id=process_id, process_store=process_store, run_store=sdk_run_store, record=record)
    return HiringScenarioResult(record=resumed.record, workflow_results=workflow_results + resumed.workflow_results)


async def resume_hiring_process(
    *,
    process_id: str,
    process_store: HiringProcessStore,
    run_store: AgentRunStore | None = None,
    record: HiringProcessRecord | None = None,
) -> HiringScenarioResult:
    sdk_run_store = run_store or create_in_memory_agent_run_store()
    resolved = record or await process_store.load(process_id)
    if resolved is None:
        raise KeyError(f'Hiring process "{process_id}" does not exist.')
    workflow_results: list[WorkflowRunResult] = []

    if "resume_intake" not in resolved.steps or resolved.steps["resume_intake"].get("status") != "completed":
        resolved.status = "needs_repair"
        resolved.issues = resolved.issues or ["Resume intake must be completed before screening."]
        await process_store.save(resolved)
        return HiringScenarioResult(record=resolved, workflow_results=workflow_results)

    if "interview_plan" not in resolved.steps:
        screening_result = await _run_sdk_workflow(
            name="hr_screening_and_interview_plan",
            steps=[("screening", "candidate_screening_agent"), ("interview_plan", "interview_plan_agent")],
            run_store=sdk_run_store,
        )
        workflow_results.append(screening_result)
        resolved.workflow_run_ids.append(screening_result.run_id)
        job = JobDescription.model_validate(resolved.steps["job_description"]["data"])
        profile = CandidateProfile.model_validate(resolved.steps["resume_intake"]["data"])
        screening = screen_candidate(job, profile)
        resolved.steps["screening"] = {"status": "completed", "data": screening.model_dump()}
        if not screening.qualified:
            resolved.status = "rejected"
            resolved.issues = screening.evidence + [f"Missing skills: {', '.join(screening.missing_skills)}."]
            await process_store.save(resolved)
            return HiringScenarioResult(record=resolved, workflow_results=workflow_results)
        plan = create_interview_plan(job, screening)
        resolved.steps["interview_plan"] = {"status": "completed", "data": plan.model_dump()}

    resolved.status = "pending_interview"
    await process_store.save(resolved)
    return HiringScenarioResult(record=resolved, workflow_results=workflow_results)


async def submit_interview(
    *,
    process_id: str,
    transcript: str,
    process_store: HiringProcessStore,
    run_store: AgentRunStore | None = None,
) -> HiringScenarioResult:
    sdk_run_store = run_store or create_in_memory_agent_run_store()
    record = await process_store.load(process_id)
    if record is None:
        raise KeyError(f'Hiring process "{process_id}" does not exist.')
    if record.status != "pending_interview":
        raise ValueError(f'Hiring process "{process_id}" is not waiting for an interview transcript.')

    evaluation_result = await _run_sdk_workflow(
        name="hr_interview_recommendation_and_fairness",
        steps=[
            ("interview_evaluation", "interview_summary_agent"),
            ("recommendation", "hiring_recommendation_agent"),
            ("fairness_review", "fairness_compliance_judge"),
        ],
        run_store=sdk_run_store,
    )
    record.workflow_run_ids.append(evaluation_result.run_id)
    profile = CandidateProfile.model_validate(record.steps["resume_intake"]["data"])
    screening = ScreeningReport.model_validate(record.steps["screening"]["data"])
    evaluation = evaluate_interview(transcript)
    recommendation = recommend_candidate(profile, screening, evaluation)
    fairness = review_fairness(recommendation)
    record.steps["interview_evaluation"] = {"status": "completed", "data": evaluation.model_dump()}
    record.steps["recommendation"] = {"status": "completed", "data": recommendation.model_dump()}
    record.steps["fairness_review"] = {"status": "completed", "data": fairness.model_dump()}
    if not fairness.valid:
        record.status = "failed"
        record.issues = fairness.findings
        await process_store.save(record)
        return HiringScenarioResult(record=record, workflow_results=[evaluation_result])

    record.status = "pending_recruiter_review"
    record.recruiter_review_requested = True
    await process_store.save(record)
    return HiringScenarioResult(record=record, workflow_results=[evaluation_result])


async def finalize_recruiter_review(
    *,
    process_id: str,
    approved: bool,
    process_store: HiringProcessStore,
    run_store: AgentRunStore | None = None,
) -> HiringScenarioResult:
    sdk_run_store = run_store or create_in_memory_agent_run_store()
    record = await process_store.load(process_id)
    if record is None:
        raise KeyError(f'Hiring process "{process_id}" does not exist.')
    if record.status != "pending_recruiter_review":
        raise ValueError(f'Hiring process "{process_id}" is not waiting for recruiter review.')

    decision_result = await _run_sdk_workflow(
        name="hr_recruiter_decision",
        steps=[("recruiter_decision", "recruiter_review_gate")],
        run_store=sdk_run_store,
    )
    record.workflow_run_ids.append(decision_result.run_id)
    record.recruiter_approved = approved
    record.steps["recruiter_decision"] = {"status": "completed", "data": recruiter_decision(process_id, approved)}
    record.status = "completed" if approved else "rejected"
    await process_store.save(record)
    return HiringScenarioResult(record=record, workflow_results=[decision_result])


def summarize(record: HiringProcessRecord, workflow_results: list[WorkflowRunResult]) -> dict[str, object]:
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
        "recruiter_review_requested": record.recruiter_review_requested,
        "recruiter_approved": record.recruiter_approved,
        "workflow_runs": record.workflow_run_ids,
        "trace_previews": trace_previews,
        "replay_events": replay_events,
    }


async def demo_strong_candidate() -> dict[str, object]:
    process_store = InMemoryHiringProcessStore()
    run_store = create_in_memory_agent_run_store()
    started = await start_hiring_process(
        process_id="HR-2025-001",
        job_payload=SAMPLE_JOB,
        candidate_payload=SAMPLE_STRONG_CANDIDATE,
        process_store=process_store,
        run_store=run_store,
    )
    interviewed = await submit_interview(
        process_id="HR-2025-001",
        transcript=SAMPLE_STRONG_INTERVIEW_TRANSCRIPT,
        process_store=process_store,
        run_store=run_store,
    )
    finalized = await finalize_recruiter_review(
        process_id="HR-2025-001",
        approved=True,
        process_store=process_store,
        run_store=run_store,
    )
    workflows = started.workflow_results + interviewed.workflow_results + finalized.workflow_results
    return summarize(finalized.record, workflows)


async def demo_repair_and_resume() -> dict[str, object]:
    process_store = InMemoryHiringProcessStore()
    run_store = create_in_memory_agent_run_store()
    stopped = await start_hiring_process(
        process_id="HR-2025-002",
        job_payload=SAMPLE_JOB,
        candidate_payload=SAMPLE_INCOMPLETE_CANDIDATE,
        process_store=process_store,
        run_store=run_store,
    )
    await process_store.apply_repair("HR-2025-002", {"years_experience": 6})
    resumed = await resume_hiring_process(
        process_id="HR-2025-002",
        process_store=process_store,
        run_store=run_store,
    )
    interviewed = await submit_interview(
        process_id="HR-2025-002",
        transcript=SAMPLE_STRONG_INTERVIEW_TRANSCRIPT,
        process_store=process_store,
        run_store=run_store,
    )
    finalized = await finalize_recruiter_review(
        process_id="HR-2025-002",
        approved=True,
        process_store=process_store,
        run_store=run_store,
    )
    workflows = stopped.workflow_results + resumed.workflow_results + interviewed.workflow_results + finalized.workflow_results
    return summarize(finalized.record, workflows)


async def demo_bias_risk() -> dict[str, object]:
    process_store = InMemoryHiringProcessStore()
    run_store = create_in_memory_agent_run_store()
    started = await start_hiring_process(
        process_id="HR-2025-003",
        job_payload=SAMPLE_JOB,
        candidate_payload=SAMPLE_BIAS_RISK_CANDIDATE,
        process_store=process_store,
        run_store=run_store,
    )
    interviewed = await submit_interview(
        process_id="HR-2025-003",
        transcript=SAMPLE_BIAS_RISK_INTERVIEW_TRANSCRIPT,
        process_store=process_store,
        run_store=run_store,
    )
    workflows = started.workflow_results + interviewed.workflow_results
    return summarize(interviewed.record, workflows)


async def main() -> None:
    strong = await demo_strong_candidate()
    repaired = await demo_repair_and_resume()
    bias_risk = await demo_bias_risk()

    print("Strong candidate")
    print(strong)
    print("Repair and resume")
    print(repaired)
    print("Bias-risk recommendation")
    print(bias_risk)


if __name__ == "__main__":
    asyncio.run(main())
