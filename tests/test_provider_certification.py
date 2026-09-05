from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import TestCase

from pydantic import ValidationError

from scripts.provider_certification import (
    DEFAULT_POLICY_PATH,
    DEFAULT_SCHEMA_PATH,
    CertificationEvidence,
    CertificationPolicy,
    certified_tier1_providers,
    evaluate_certifications,
    load_evidence,
    load_policy,
    render_certification_markdown,
    serialized_evidence_schema,
    validate_tier1_inventory,
)
from zhivex_ai.provider_support import TIER_1_PROVIDERS


RECORDED_AT = datetime(2026, 8, 30, 3, 37, 39, tzinfo=timezone.utc)


def _evidence_payload(
    *,
    target_id: str = "openai-standard",
    provider: str = "openai",
    surface: str = "standard",
    model: str = "gpt-5.6-luna",
    recorded_at: datetime = RECORDED_AT,
    run_status: str = "passed",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_status": run_status,
        "recorded_at": recorded_at.isoformat(),
        "artifact": {
            "kind": "wheel",
            "package_name": "zhivex-ai-sdk",
            "package_version": "0.22.0",
            "source_revision": "a" * 40,
            "installation_status": "passed",
            "filename": "zhivex_ai_sdk-0.22.0-py3-none-any.whl",
            "sha256": "b" * 64,
        },
        "workflow": {
            "platform": "github-actions",
            "name": "Publish to PyPI",
            "repository": "Zhivex/zhivex-ai-sdk-py",
            "run_id": 123456,
            "run_attempt": 1,
        },
        "targets": [
            {
                "target_id": target_id,
                "provider": provider,
                "surface": surface,
                "model": model,
                "result": "passed",
                "operations": [
                    {"name": "generation", "status": "passed"},
                    {"name": "agent-tool", "status": "passed"},
                ],
            }
        ],
    }


def _policy() -> CertificationPolicy:
    return load_policy(DEFAULT_POLICY_PATH.parent / "releases/0.22.0-certification-policy.json")


class ProviderCertificationTests(TestCase):
    def test_reviewed_artifact_rejects_other_version_source_or_wheel(self) -> None:
        from scripts.provider_certification import ExpectedArtifact

        payload = _evidence_payload()
        artifact = payload["artifact"]
        policy = _policy().model_copy(update={"expected_artifact": ExpectedArtifact(
            package_version=artifact["package_version"], source_revision=artifact["source_revision"], sha256=artifact["sha256"],
        )})
        evaluate_certifications(policy, [CertificationEvidence.model_validate(payload)], now=RECORDED_AT)
        for field, value in [("package_version", "0.21.0"), ("source_revision", "c" * 40), ("sha256", "d" * 64)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(payload)
                changed["artifact"][field] = value
                with self.assertRaisesRegex(ValueError, "artifact identity"):
                    evaluate_certifications(policy, [CertificationEvidence.model_validate(changed)], now=RECORDED_AT)

    def test_old_local_integration_evidence_is_stale(self) -> None:
        payload = _evidence_payload()
        payload["workflow"] = {"platform": "local", "name": "local-smoke"}
        reports = evaluate_certifications(_policy(), [CertificationEvidence.model_validate(payload)], now=RECORDED_AT + timedelta(days=31))
        self.assertEqual(reports[0].live_status, "stale")

    def test_versioned_policy_matches_the_runtime_tier1_inventory(self) -> None:
        policy = _policy()

        validate_tier1_inventory(policy, TIER_1_PROVIDERS)

        self.assertEqual(
            tuple(target.provider for target in policy.targets if target.tier1),
            TIER_1_PROVIDERS,
        )
        meta_targets = {target.id: target for target in policy.targets if target.provider == "meta"}
        self.assertEqual(meta_targets["meta-standard"].surface, "standard")
        self.assertTrue(meta_targets["meta-standard"].tier1)
        self.assertEqual(meta_targets["meta-contributor"].surface, "contributor")
        self.assertFalse(meta_targets["meta-contributor"].tier1)

    def test_committed_schema_is_deterministic_and_current(self) -> None:
        first = serialized_evidence_schema()
        second = serialized_evidence_schema()

        self.assertEqual(first, second)
        self.assertEqual(DEFAULT_SCHEMA_PATH.read_text("utf-8"), first)

    def test_complete_exact_wheel_evidence_is_certified_until_the_30_day_boundary(self) -> None:
        evidence = CertificationEvidence.model_validate(_evidence_payload())
        policy = _policy()

        at_boundary = evaluate_certifications(
            policy,
            [evidence],
            now=RECORDED_AT + timedelta(days=30),
        )
        after_boundary = evaluate_certifications(
            policy,
            [evidence],
            now=RECORDED_AT + timedelta(days=30, microseconds=1),
        )

        self.assertEqual(at_boundary[0].live_status, "certified")
        self.assertEqual(after_boundary[0].live_status, "stale")

    def test_contributor_evidence_never_certifies_meta_standard(self) -> None:
        evidence = CertificationEvidence.model_validate(
            _evidence_payload(
                target_id="meta-contributor",
                provider="meta",
                surface="contributor",
                model="muse-spark-1.2-contributor",
            )
        )

        reports = evaluate_certifications(_policy(), [evidence], now=RECORDED_AT)
        by_target = {report.target_id: report for report in reports}

        self.assertEqual(by_target["meta-contributor"].live_status, "certified")
        self.assertEqual(by_target["meta-standard"].live_status, "missing")
        self.assertNotIn("meta", certified_tier1_providers(reports))

    def test_second_cohort_policy_requires_passed_and_explicitly_unsupported_operations(self) -> None:
        payload = _evidence_payload(
            target_id="qwen-standard",
            provider="qwen",
            model="qwen3.8-max",
        )
        target = payload["targets"][0]  # type: ignore[index]
        target["operations"] = [  # type: ignore[index]
            {"name": "generation", "status": "passed"},
            {"name": "streaming", "status": "passed"},
            {"name": "structured-output", "status": "passed"},
            {"name": "agent-tool", "status": "passed"},
            {"name": "portable-retrieval", "status": "unsupported"},
        ]
        evidence = CertificationEvidence.model_validate(payload)

        reports = evaluate_certifications(_policy(), [evidence], now=RECORDED_AT)
        qwen = next(report for report in reports if report.target_id == "qwen-standard")

        self.assertEqual(qwen.live_status, "certified")
        self.assertIn("portable-retrieval=unsupported", qwen.operations)

        missing = copy.deepcopy(payload)
        missing["targets"][0]["operations"] = missing["targets"][0]["operations"][:-1]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "operations do not match policy"):
            evaluate_certifications(
                _policy(),
                [CertificationEvidence.model_validate(missing)],
                now=RECORDED_AT,
            )

        false_success = copy.deepcopy(payload)
        false_success["targets"][0]["operations"][-1]["status"] = "passed"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "unsupported operations do not match policy"):
            evaluate_certifications(
                _policy(),
                [CertificationEvidence.model_validate(false_success)],
                now=RECORDED_AT,
            )

    def test_failed_workflow_cannot_certify_a_passed_target(self) -> None:
        evidence = CertificationEvidence.model_validate(_evidence_payload(run_status="failed"))

        report = evaluate_certifications(_policy(), [evidence], now=RECORDED_AT)[0]

        self.assertEqual(report.live_status, "failed")
        self.assertNotIn("openai", certified_tier1_providers([report]))

    def test_source_or_local_evidence_is_integration_only(self) -> None:
        payload = _evidence_payload()
        payload["artifact"] = {
            "kind": "source",
            "package_name": "zhivex-ai-sdk",
            "package_version": "0.22.0",
            "source_revision": "a" * 40,
            "installation_status": "not-applicable",
        }
        payload["workflow"] = {"platform": "local", "name": "Local smoke"}
        evidence = CertificationEvidence.model_validate(payload)

        report = evaluate_certifications(_policy(), [evidence], now=RECORDED_AT)[0]

        self.assertEqual(report.source_tests, "contract-supported")
        self.assertEqual(report.installed_wheel, "not-applicable")
        self.assertEqual(report.live_status, "integration-only")

    def test_newer_local_evidence_does_not_shadow_protected_certification(self) -> None:
        protected = CertificationEvidence.model_validate(_evidence_payload())
        local_payload = _evidence_payload(recorded_at=RECORDED_AT + timedelta(hours=1))
        local_payload["workflow"] = {"platform": "local", "name": "Local smoke"}
        local = CertificationEvidence.model_validate(local_payload)

        report = evaluate_certifications(
            _policy(),
            [protected, local],
            now=RECORDED_AT + timedelta(hours=1),
        )[0]

        self.assertEqual(report.live_status, "certified")
        self.assertEqual(report.recorded_at, RECORDED_AT.isoformat())

    def test_schema_rejects_sensitive_or_unbounded_fields(self) -> None:
        for path, field, value in (
            (("root",), "api_key", "secret"),
            (("target",), "prompt", "raw prompt"),
            (("workflow",), "headers", {"authorization": "secret"}),
            (("artifact",), "endpoint", "https://private.example"),
        ):
            with self.subTest(field=field):
                payload = copy.deepcopy(_evidence_payload())
                if path == ("root",):
                    payload[field] = value
                elif path == ("target",):
                    targets = payload["targets"]
                    assert isinstance(targets, list)
                    targets[0][field] = value
                else:
                    nested = payload[path[0]]
                    assert isinstance(nested, dict)
                    nested[field] = value
                with self.assertRaises(ValidationError):
                    CertificationEvidence.model_validate(payload)

        payload = _evidence_payload(model="https://private.example/model?token=secret")
        with self.assertRaises(ValidationError):
            CertificationEvidence.model_validate(payload)

        payload = _evidence_payload(model="Bearer synthetic-credential-canary")
        with self.assertRaises(ValidationError):
            CertificationEvidence.model_validate(payload)

    def test_identity_model_and_future_timestamp_mismatches_fail_closed(self) -> None:
        policy = _policy()
        wrong_identity = CertificationEvidence.model_validate(
            _evidence_payload(provider="anthropic")
        )
        with self.assertRaisesRegex(ValueError, "does not match policy identity"):
            evaluate_certifications(policy, [wrong_identity], now=RECORDED_AT)

        wrong_meta_model = CertificationEvidence.model_validate(
            _evidence_payload(
                target_id="meta-standard",
                provider="meta",
                model="muse-spark-1.2-contributor",
            )
        )
        with self.assertRaisesRegex(ValueError, "model does not match policy"):
            evaluate_certifications(policy, [wrong_meta_model], now=RECORDED_AT)

        future = CertificationEvidence.model_validate(
            _evidence_payload(recorded_at=RECORDED_AT + timedelta(minutes=6))
        )
        with self.assertRaisesRegex(ValueError, "future tolerance"):
            evaluate_certifications(policy, [future], now=RECORDED_AT)

    def test_versioned_evidence_generates_the_human_matrix_without_cross_certifying_meta(self) -> None:
        policy = _policy()
        evidence_path = Path("docs/releases/0.22.0-certification.json")
        evidence = load_evidence(evidence_path)

        reports = evaluate_certifications(policy, [evidence], now=RECORDED_AT)
        rendered = render_certification_markdown(reports, policy=policy)

        self.assertIn("| openai | openai-standard | standard |", rendered)
        self.assertIn("| meta | meta-standard | standard | muse-spark-1.2 |", rendered)
        self.assertIn("| meta | meta-contributor | contributor |", rendered)
        by_target = {report.target_id: report for report in reports}
        self.assertEqual(by_target["openai-standard"].live_status, "certified")
        self.assertEqual(by_target["meta-standard"].live_status, "missing")
        self.assertEqual(by_target["meta-contributor"].live_status, "certified")

    def test_schema_json_rejects_additional_properties_at_every_contract_boundary(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA_PATH.read_text("utf-8"))

        self.assertFalse(schema["additionalProperties"])
        definitions = schema["$defs"]
        for name in ("ArtifactEvidence", "WorkflowEvidence", "OperationEvidence", "TargetEvidence"):
            self.assertFalse(definitions[name]["additionalProperties"], name)
