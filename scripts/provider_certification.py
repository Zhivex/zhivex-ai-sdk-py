from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path, PurePosixPath
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "docs/provider-certification-policy.json"
DEFAULT_SCHEMA_PATH = ROOT / "docs/schemas/provider-certification-evidence.schema.json"
SUPPORT_CERTIFICATION_BEGIN = "<!-- BEGIN GENERATED PROVIDER CERTIFICATION -->"
SUPPORT_CERTIFICATION_END = "<!-- END GENERATED PROVIDER CERTIFICATION -->"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+:-]{0,127}$")
PACKAGE_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?$")
WHEEL_FILENAME_RE = re.compile(r"^zhivex_ai_sdk-[A-Za-z0-9._+-]+-py3-none-any\.whl$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
WORKFLOW_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._/-]{0,127}$")
DIAGNOSTIC_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
SENSITIVE_VALUE_RE = re.compile(
    r"^(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,}|github_pat_|AKIA[A-Z0-9]{8,}|Bearer\s)",
    re.IGNORECASE,
)

CertificationSurface = Literal["standard", "contributor", "deployment"]
StabilityLevel = Literal["stable", "beta", "experimental"]
OperationName = Literal[
    "generation",
    "streaming",
    "structured-output",
    "agent-tool",
    "portable-retrieval",
    "embeddings",
    "grounding",
    "transcription",
    "speech",
]
OperationStatus = Literal["passed", "failed", "blocked", "unsupported"]
TargetResult = Literal["passed", "failed", "blocked", "unsupported"]
CertificationStatus = Literal[
    "certified",
    "stale",
    "failed",
    "blocked",
    "unsupported",
    "integration-only",
    "missing",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_identifier(value: str, *, label: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase kebab-case identifier")
    return value


def _validate_model_id(value: str) -> str:
    if (
        not MODEL_RE.fullmatch(value)
        or "://" in value
        or "@" in value
        or SENSITIVE_VALUE_RE.match(value)
    ):
        raise ValueError("model must be a bounded provider model ID, not a URL or credential")
    return value


class CertificationTargetSpec(_StrictModel):
    id: str = Field(pattern=IDENTIFIER_RE.pattern)
    provider: str = Field(pattern=IDENTIFIER_RE.pattern)
    surface: CertificationSurface
    stability: StabilityLevel
    tier1: bool
    expected_model: str | None = Field(default=None, pattern=MODEL_RE.pattern)
    required_operations: tuple[OperationName, ...] = ()
    unsupported_operations: tuple[OperationName, ...] = ()

    @field_validator("id", "provider")
    @classmethod
    def validate_identifiers(cls, value: str, info: ValidationInfo) -> str:
        return _validate_identifier(value, label=info.field_name)

    @field_validator("expected_model")
    @classmethod
    def validate_expected_model(cls, value: str | None) -> str | None:
        return _validate_model_id(value) if value is not None else None

    @model_validator(mode="after")
    def validate_operation_policy(self) -> CertificationTargetSpec:
        required = set(self.required_operations)
        unsupported = set(self.unsupported_operations)
        if len(required) != len(self.required_operations):
            raise ValueError("required_operations must be unique")
        if len(unsupported) != len(self.unsupported_operations):
            raise ValueError("unsupported_operations must be unique")
        if required & unsupported:
            raise ValueError("required and unsupported operations must be disjoint")
        return self


class ExpectedArtifact(_StrictModel):
    package_version: str = Field(pattern=PACKAGE_VERSION_RE.pattern)
    source_revision: str = Field(pattern=COMMIT_RE.pattern)
    sha256: str = Field(pattern=SHA256_RE.pattern)


class CertificationPolicy(_StrictModel):
    expected_artifact: ExpectedArtifact | None = None
    schema_version: Literal[1]
    max_age_days: int = Field(gt=0, le=365)
    future_tolerance_minutes: int = Field(ge=0, le=60)
    evidence_files: list[str]
    targets: list[CertificationTargetSpec]

    @field_validator("evidence_files")
    @classmethod
    def validate_evidence_paths(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("evidence_files must be unique")
        for value in values:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
                raise ValueError("evidence_files must be safe repository-relative JSON paths")
        return values

    @model_validator(mode="after")
    def validate_targets(self) -> CertificationPolicy:
        ids = [target.id for target in self.targets]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("policy target IDs must be non-empty and unique")
        tier1_providers = [target.provider for target in self.targets if target.tier1]
        if len(tier1_providers) != len(set(tier1_providers)):
            raise ValueError("each Tier-1 provider must have exactly one canonical target")
        return self


class ArtifactEvidence(_StrictModel):
    kind: Literal["wheel", "source"]
    package_name: Literal["zhivex-ai-sdk"] = "zhivex-ai-sdk"
    package_version: str = Field(pattern=PACKAGE_VERSION_RE.pattern)
    source_revision: str = Field(pattern=COMMIT_RE.pattern)
    installation_status: Literal["passed", "failed", "not-applicable"]
    filename: str | None = Field(default=None, pattern=WHEEL_FILENAME_RE.pattern)
    sha256: str | None = Field(default=None, pattern=SHA256_RE.pattern)

    @field_validator("package_version")
    @classmethod
    def validate_package_version(cls, value: str) -> str:
        if not PACKAGE_VERSION_RE.fullmatch(value):
            raise ValueError("package_version must be a bounded semantic version")
        return value

    @field_validator("source_revision")
    @classmethod
    def validate_source_revision(cls, value: str) -> str:
        if not COMMIT_RE.fullmatch(value):
            raise ValueError("source_revision must be a lowercase 40-character commit SHA")
        return value

    @model_validator(mode="after")
    def validate_artifact_shape(self) -> ArtifactEvidence:
        if self.kind == "wheel":
            if self.installation_status == "not-applicable":
                raise ValueError("wheel evidence must record installation_status")
            if self.filename is None or not WHEEL_FILENAME_RE.fullmatch(self.filename):
                raise ValueError("wheel evidence requires a sanitized zhivex wheel filename")
            if self.sha256 is None or not SHA256_RE.fullmatch(self.sha256):
                raise ValueError("wheel evidence requires a lowercase SHA256")
        elif self.filename is not None or self.sha256 is not None:
            raise ValueError("source evidence cannot include wheel filename or SHA256")
        elif self.installation_status != "not-applicable":
            raise ValueError("source evidence installation_status must be not-applicable")
        return self


class WorkflowEvidence(_StrictModel):
    platform: Literal["github-actions", "local"]
    name: str = Field(pattern=WORKFLOW_NAME_RE.pattern)
    repository: str | None = Field(default=None, pattern=REPOSITORY_RE.pattern)
    run_id: int | None = Field(default=None, gt=0)
    run_attempt: int | None = Field(default=None, gt=0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not WORKFLOW_NAME_RE.fullmatch(value) or "://" in value or "@" in value:
            raise ValueError("workflow name must be a bounded non-sensitive label")
        return value

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str | None) -> str | None:
        if value is not None and not REPOSITORY_RE.fullmatch(value):
            raise ValueError("repository must use owner/name format")
        return value

    @model_validator(mode="after")
    def validate_platform_fields(self) -> WorkflowEvidence:
        if self.platform == "github-actions":
            if self.repository is None or self.run_id is None or self.run_attempt is None:
                raise ValueError("GitHub Actions evidence requires repository, run_id, and run_attempt")
        elif self.repository is not None or self.run_id is not None or self.run_attempt is not None:
            raise ValueError("local evidence cannot claim GitHub workflow identity")
        return self


class OperationEvidence(_StrictModel):
    name: OperationName
    status: OperationStatus


class TargetEvidence(_StrictModel):
    target_id: str = Field(pattern=IDENTIFIER_RE.pattern)
    provider: str = Field(pattern=IDENTIFIER_RE.pattern)
    surface: CertificationSurface
    model: str = Field(pattern=MODEL_RE.pattern)
    result: TargetResult
    operations: list[OperationEvidence] = Field(default_factory=list)
    diagnostic_code: str | None = Field(default=None, pattern=DIAGNOSTIC_CODE_RE.pattern)
    diagnostic_fingerprint: str | None = Field(default=None, pattern=SHA256_RE.pattern)

    @field_validator("target_id", "provider")
    @classmethod
    def validate_identifiers(cls, value: str, info: ValidationInfo) -> str:
        return _validate_identifier(value, label=info.field_name)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return _validate_model_id(value)

    @field_validator("diagnostic_code")
    @classmethod
    def validate_diagnostic_code(cls, value: str | None) -> str | None:
        if value is not None and not DIAGNOSTIC_CODE_RE.fullmatch(value):
            raise ValueError("diagnostic_code must be an allowlisted machine code")
        return value

    @field_validator("diagnostic_fingerprint")
    @classmethod
    def validate_diagnostic_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("diagnostic_fingerprint must be a lowercase SHA256")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> TargetEvidence:
        operation_names = [operation.name for operation in self.operations]
        if len(operation_names) != len(set(operation_names)):
            raise ValueError("operation names must be unique per target")
        if self.result == "passed":
            if not self.operations or not any(
                operation.status == "passed" for operation in self.operations
            ):
                raise ValueError("passed target evidence requires at least one passed operation")
            if any(operation.status not in {"passed", "unsupported"} for operation in self.operations):
                raise ValueError(
                    "passed target evidence may contain only passed or unsupported operations"
                )
            if self.diagnostic_code is not None or self.diagnostic_fingerprint is not None:
                raise ValueError("passed target evidence cannot include diagnostics")
        else:
            if self.diagnostic_code is None:
                raise ValueError("non-passed target evidence requires a diagnostic_code")
            allowed_status = self.result
            if any(
                operation.status not in {"passed", "unsupported", allowed_status}
                for operation in self.operations
            ):
                raise ValueError("operation status is inconsistent with the target result")
        return self


class CertificationEvidence(_StrictModel):
    schema_version: Literal[1]
    run_status: Literal["passed", "failed"]
    recorded_at: datetime
    artifact: ArtifactEvidence
    workflow: WorkflowEvidence
    targets: list[TargetEvidence]

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("recorded_at must include a UTC offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_evidence(self) -> CertificationEvidence:
        target_ids = [target.target_id for target in self.targets]
        if not target_ids or len(target_ids) != len(set(target_ids)):
            raise ValueError("evidence target IDs must be non-empty and unique")
        if any(target.result == "passed" for target in self.targets):
            if self.artifact.kind == "wheel" and self.artifact.installation_status != "passed":
                raise ValueError("passed wheel targets require a passed installation")
        return self


@dataclass(frozen=True, slots=True)
class TargetCertification:
    target_id: str
    provider: str
    surface: CertificationSurface
    stability: StabilityLevel
    tier1: bool
    model: str | None
    source_tests: str
    installed_wheel: str
    live_status: CertificationStatus
    recorded_at: str | None
    operations: tuple[str, ...]
    artifact_sha256: str | None
    source_revision: str | None
    workflow_run: str | None


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> CertificationPolicy:
    return CertificationPolicy.model_validate_json(path.read_text("utf-8"))


def validate_tier1_inventory(policy: CertificationPolicy, tier1_providers: tuple[str, ...]) -> None:
    observed = tuple(target.provider for target in policy.targets if target.tier1)
    if observed != tier1_providers:
        raise ValueError(
            "policy Tier-1 target order must match runtime metadata: "
            f"expected {', '.join(tier1_providers)}, observed {', '.join(observed)}"
        )


def load_evidence(path: Path) -> CertificationEvidence:
    return CertificationEvidence.model_validate_json(path.read_text("utf-8"))


def load_policy_evidence(
    policy: CertificationPolicy,
    *,
    root: Path = ROOT,
) -> list[CertificationEvidence]:
    return [load_evidence(root / relative_path) for relative_path in policy.evidence_files]


def evaluate_certifications(
    policy: CertificationPolicy,
    evidence_records: list[CertificationEvidence],
    *,
    now: datetime | None = None,
) -> list[TargetCertification]:
    evaluation_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    future_tolerance = timedelta(minutes=policy.future_tolerance_minutes)
    max_age = timedelta(days=policy.max_age_days)
    specs = {target.id: target for target in policy.targets}
    latest: dict[str, tuple[CertificationEvidence, TargetEvidence]] = {}

    for evidence in evidence_records:
        expected = policy.expected_artifact
        if expected is not None and (
            evidence.artifact.kind != "wheel"
            or evidence.artifact.package_version != expected.package_version
            or evidence.artifact.source_revision != expected.source_revision
            or evidence.artifact.sha256 != expected.sha256
        ):
            raise ValueError("evidence artifact identity does not match the reviewed policy")
        if evidence.recorded_at - evaluation_time > future_tolerance:
            raise ValueError("evidence recorded_at exceeds the future tolerance")
        for target in evidence.targets:
            spec = specs.get(target.target_id)
            if spec is None:
                raise ValueError(f'evidence references unknown target "{target.target_id}"')
            if target.provider != spec.provider or target.surface != spec.surface:
                raise ValueError(f'evidence target "{target.target_id}" does not match policy identity')
            if spec.expected_model is not None and target.model != spec.expected_model:
                raise ValueError(f'evidence target "{target.target_id}" model does not match policy')
            operation_statuses = {operation.name: operation.status for operation in target.operations}
            expected_operations = set(spec.required_operations) | set(
                spec.unsupported_operations
            )
            if expected_operations:
                if set(operation_statuses) != expected_operations:
                    raise ValueError(
                        f'evidence target "{target.target_id}" operations do not match policy'
                    )
                if any(
                    operation_statuses[operation] != "unsupported"
                    for operation in spec.unsupported_operations
                ):
                    raise ValueError(
                        f'evidence target "{target.target_id}" unsupported operations do not match policy'
                    )
                if target.result == "passed" and any(
                    operation_statuses[operation] != "passed"
                    for operation in spec.required_operations
                ):
                    raise ValueError(
                        f'evidence target "{target.target_id}" required operations did not pass'
                    )
            previous = latest.get(target.target_id)
            current_rank = 1 if evidence.workflow.platform == "github-actions" else 0
            previous_rank = (
                1 if previous is not None and previous[0].workflow.platform == "github-actions" else 0
            )
            if previous is None or (current_rank, evidence.recorded_at) > (
                previous_rank,
                previous[0].recorded_at,
            ):
                latest[target.target_id] = (evidence, target)

    reports: list[TargetCertification] = []
    for spec in policy.targets:
        matched = latest.get(spec.id)
        if matched is None:
            reports.append(
                TargetCertification(
                    target_id=spec.id,
                    provider=spec.provider,
                    surface=spec.surface,
                    stability=spec.stability,
                    tier1=spec.tier1,
                    model=spec.expected_model,
                    source_tests="contract-supported" if spec.tier1 else "beta-contract",
                    installed_wheel="missing",
                    live_status="missing",
                    recorded_at=None,
                    operations=(),
                    artifact_sha256=None,
                    source_revision=None,
                    workflow_run=None,
                )
            )
            continue

        evidence, target = matched
        if target.result == "passed" and evidence.run_status == "passed":
            if evaluation_time - evidence.recorded_at > max_age:
                live_status: CertificationStatus = "stale"
            elif evidence.artifact.kind != "wheel" or evidence.workflow.platform != "github-actions":
                live_status = "integration-only"
            else:
                live_status = "certified"
        elif target.result == "passed":
            live_status = "failed"
        else:
            live_status = target.result
        workflow_run = None
        if evidence.workflow.platform == "github-actions":
            workflow_run = f"{evidence.workflow.repository}#{evidence.workflow.run_id}"
        reports.append(
            TargetCertification(
                target_id=spec.id,
                provider=spec.provider,
                surface=spec.surface,
                stability=spec.stability,
                tier1=spec.tier1,
                model=target.model,
                source_tests="contract-supported" if spec.tier1 else "beta-contract",
                installed_wheel=(
                    evidence.artifact.installation_status
                    if evidence.artifact.kind == "wheel"
                    else "not-applicable"
                ),
                live_status=live_status,
                recorded_at=evidence.recorded_at.isoformat(),
                operations=tuple(
                    f"{operation.name}={operation.status}"
                    for operation in sorted(target.operations, key=lambda item: item.name)
                ),
                artifact_sha256=evidence.artifact.sha256,
                source_revision=evidence.artifact.source_revision,
                workflow_run=workflow_run,
            )
        )
    return reports


def certified_tier1_providers(reports: list[TargetCertification]) -> set[str]:
    return {
        report.provider
        for report in reports
        if report.tier1 and report.live_status == "certified"
    }


def render_certification_markdown(
    reports: list[TargetCertification],
    *,
    policy: CertificationPolicy,
) -> str:
    headers = [
        "Provider",
        "Target",
        "Surface",
        "Model",
        "Source tests",
        "Installed wheel",
        "Live",
        "Recorded at",
        "Operations",
    ]
    rows = [
        [
            report.provider,
            report.target_id,
            report.surface,
            report.model or "unassigned",
            report.source_tests,
            report.installed_wheel,
            report.live_status,
            report.recorded_at or "—",
            ", ".join(report.operations) or "—",
        ]
        for report in reports
    ]
    table = _render_table(headers, rows)
    return "\n".join(
        [
            "## Current provider certification",
            "",
            "This table is generated from the versioned certification policy and validated evidence records.",
            "Contract tests, installed-wheel execution, and live certification are separate evidence layers.",
            f"A passed exact-artifact live record remains current for {policy.max_age_days} days; older records are shown as `stale`.",
            "Missing, blocked, failed, unsupported, local-only, or malformed evidence never produces `release-certified` status.",
            "Meta Standard and Meta Contributor are independent targets; Contributor cannot certify the Stable Standard route.",
            "",
            table,
        ]
    )


def render_support_certification_block(
    reports: list[TargetCertification],
    *,
    policy: CertificationPolicy,
) -> str:
    content = render_certification_markdown(reports, policy=policy)
    return "\n".join([SUPPORT_CERTIFICATION_BEGIN, content, SUPPORT_CERTIFICATION_END])


def replace_support_certification(
    markdown: str,
    reports: list[TargetCertification],
    *,
    policy: CertificationPolicy,
) -> str:
    if SUPPORT_CERTIFICATION_BEGIN not in markdown or SUPPORT_CERTIFICATION_END not in markdown:
        raise ValueError("SUPPORT.md certification markers are missing")
    start = markdown.index(SUPPORT_CERTIFICATION_BEGIN)
    finish = markdown.index(SUPPORT_CERTIFICATION_END) + len(SUPPORT_CERTIFICATION_END)
    return markdown[:start] + render_support_certification_block(reports, policy=policy) + markdown[finish:]


def evidence_schema() -> dict[str, object]:
    schema = CertificationEvidence.model_json_schema()
    # Pydantic 2.8 emits both const and a redundant one-value enum for Literal.
    # Canonicalize this equivalent representation across supported versions.
    def canonicalize(value: object) -> None:
        if isinstance(value, dict):
            if "const" in value and value.get("enum") == [value["const"]]:
                value.pop("enum")
            for child in value.values():
                canonicalize(child)
        elif isinstance(value, list):
            for child in value:
                canonicalize(child)
    canonicalize(schema)
    schema["$id"] = "https://github.com/Zhivex/zhivex-ai-sdk-py/blob/main/docs/schemas/provider-certification-evidence.schema.json"
    schema["title"] = "Zhivex Tier-1 Provider Certification Evidence"
    return schema


def serialized_evidence_schema() -> str:
    return json.dumps(evidence_schema(), indent=2, sort_keys=True) + "\n"


def report_payload(
    reports: list[TargetCertification],
    *,
    policy: CertificationPolicy,
    evaluated_at: datetime,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluated_at": evaluated_at.astimezone(timezone.utc).isoformat(),
        "max_age_days": policy.max_age_days,
        "targets": [asdict(report) for report in reports],
    }


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
