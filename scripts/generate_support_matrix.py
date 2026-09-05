from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.provider_certification import (  # noqa: E402
    certified_tier1_providers,
    evaluate_certifications,
    load_policy,
    load_policy_evidence,
    replace_support_certification,
    validate_tier1_inventory,
)

from zhivex_ai import (  # noqa: E402
    create_anthropic,
    create_azure_openai,
    create_bedrock,
    create_deepseek,
    create_gemini,
    create_kimi,
    create_meta,
    create_ollama,
    create_openai,
    create_openrouter,
    create_qwen,
    create_vllm,
    create_vertex,
)
from zhivex_ai.provider_support import (  # noqa: E402
    TIER_1_PROVIDERS,
    build_provider_support_rows,
    render_provider_support_markdown,
    replace_readme_support_matrix,
)


class _FakeBedrockClient:
    async def converse(self, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--as-of must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render or sync the fail-closed provider support matrix from runtime metadata. "
            "The offline generator reports contract support and never infers release certification from Tier-1 membership."
        )
    )
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Rewrite only the legacy generated support-matrix block in README.md.",
    )
    parser.add_argument(
        "--check-readme",
        action="store_true",
        help="Check only the legacy generated support-matrix block in README.md.",
    )
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="Rewrite README capability status and the SUPPORT certification matrix.",
    )
    parser.add_argument(
        "--check-docs",
        action="store_true",
        help="Exit non-zero when either generated documentation block is out of date.",
    )
    parser.add_argument("--as-of", type=_parse_utc)
    args = parser.parse_args()

    policy = load_policy()
    validate_tier1_inventory(policy, TIER_1_PROVIDERS)
    certification_reports = evaluate_certifications(
        policy,
        load_policy_evidence(policy),
        now=args.as_of or datetime.now(timezone.utc),
    )

    rows = build_provider_support_rows(
        [
            create_openai(api_key="test"),
            create_azure_openai(api_key="test", endpoint="https://example.openai.azure.com"),
            create_anthropic(api_key="test"),
            create_gemini(api_key="test"),
            create_vertex(access_token="test", project_id="project"),
            create_bedrock(client=_FakeBedrockClient()),
            create_deepseek(api_key="test"),
            create_openrouter(api_key="test"),
            create_qwen(api_key="test"),
            create_kimi(api_key="test"),
            create_meta(api_key="test"),
            create_ollama(),
            create_vllm(api_key="test"),
        ],
        validated_release_certifications=certified_tier1_providers(certification_reports),
    )
    rendered = render_provider_support_markdown(rows)

    if args.write_readme or args.check_readme or args.write_docs or args.check_docs:
        readme_path = ROOT / "README.md"
        current_readme = readme_path.read_text(encoding="utf-8")
        updated_readme = replace_readme_support_matrix(current_readme, rows)
        support_path = ROOT / "SUPPORT.md"
        current_support = support_path.read_text(encoding="utf-8")
        updated_support = replace_support_certification(
            current_support,
            certification_reports,
            policy=policy,
        )
        if args.check_readme or args.check_docs:
            stale_paths = []
            if current_readme != updated_readme:
                stale_paths.append("README.md")
            if args.check_docs and current_support != updated_support:
                stale_paths.append("SUPPORT.md")
            if stale_paths:
                print(f"Generated provider documentation is out of date: {', '.join(stale_paths)}")
                raise SystemExit(1)
            return
        if args.write_readme or args.write_docs:
            readme_path.write_text(updated_readme, encoding="utf-8")
            print(f"Updated {readme_path.relative_to(ROOT)}")
        if args.write_docs:
            support_path.write_text(updated_support, encoding="utf-8")
            print(f"Updated {support_path.relative_to(ROOT)}")
        return

    print(rendered)


if __name__ == "__main__":
    main()
