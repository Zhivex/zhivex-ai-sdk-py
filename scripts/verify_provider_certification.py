from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.provider_certification import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    DEFAULT_SCHEMA_PATH,
    evaluate_certifications,
    load_evidence,
    load_policy,
    load_policy_evidence,
    report_payload,
    serialized_evidence_schema,
    validate_tier1_inventory,
)
from zhivex_ai.provider_support import TIER_1_PROVIDERS  # noqa: E402


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--as-of must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _safe_validation_summary(error: ValidationError) -> str:
    summaries: list[str] = []
    for item in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item.get("loc", ())) or "root"
        summaries.append(f"{location}: {item.get('msg', 'invalid value')}")
    return "; ".join(summaries[:8])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate versioned Tier-1 provider certification evidence without exposing sensitive payloads."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--evidence",
        action="append",
        type=Path,
        help="Override policy evidence files; may be repeated.",
    )
    parser.add_argument("--as-of", type=_parse_utc)
    parser.add_argument("--check-schema", action="store_true")
    parser.add_argument("--write-schema", action="store_true")
    parser.add_argument("--require-target", action="append", default=[])
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "dist/provider-certification-report.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluated_at = args.as_of or datetime.now(timezone.utc)
    try:
        policy = load_policy(args.policy)
        validate_tier1_inventory(policy, TIER_1_PROVIDERS)
        evidence_records = (
            [load_evidence(path) for path in args.evidence]
            if args.evidence
            else load_policy_evidence(policy, root=ROOT)
        )
        reports = evaluate_certifications(policy, evidence_records, now=evaluated_at)
    except ValidationError as error:
        print(f"Provider certification validation failed: {_safe_validation_summary(error)}")
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Provider certification validation failed: {error}")
        return 1

    schema_text = serialized_evidence_schema()
    if args.write_schema:
        DEFAULT_SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_SCHEMA_PATH.write_text(schema_text, "utf-8")
    if args.check_schema:
        if not DEFAULT_SCHEMA_PATH.is_file() or DEFAULT_SCHEMA_PATH.read_text("utf-8") != schema_text:
            print("Provider certification schema is missing or out of date.")
            return 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            report_payload(reports, policy=policy, evaluated_at=evaluated_at),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "utf-8",
    )

    report_by_target = {report.target_id: report for report in reports}
    unknown_required = sorted(set(args.require_target) - set(report_by_target))
    if unknown_required:
        print(f"Unknown required certification target(s): {', '.join(unknown_required)}")
        return 1
    failed_required = [
        target_id
        for target_id in args.require_target
        if report_by_target[target_id].live_status != "certified"
    ]
    if failed_required:
        details = ", ".join(
            f"{target_id}={report_by_target[target_id].live_status}"
            for target_id in failed_required
        )
        print(f"Required provider certification is not current: {details}")
        return 1

    counts: dict[str, int] = {}
    for report in reports:
        counts[report.live_status] = counts.get(report.live_status, 0) + 1
    rendered_counts = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"Provider certification evidence valid ({rendered_counts}); report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
