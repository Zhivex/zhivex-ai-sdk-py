from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
    build_provider_support_rows,
    render_provider_support_markdown,
    replace_readme_support_matrix,
)


class _FakeBedrockClient:
    async def converse(self, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Render or sync the provider support matrix from runtime metadata.")
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Rewrite the generated support-matrix block in README.md.",
    )
    parser.add_argument(
        "--check-readme",
        action="store_true",
        help="Exit non-zero when README.md does not match the generated support-matrix block.",
    )
    args = parser.parse_args()

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
        ]
    )
    rendered = render_provider_support_markdown(rows)

    if args.write_readme or args.check_readme:
        readme_path = ROOT / "README.md"
        current = readme_path.read_text(encoding="utf-8")
        updated = replace_readme_support_matrix(current, rows)
        if args.check_readme:
            raise SystemExit(0 if current == updated else 1)
        readme_path.write_text(updated, encoding="utf-8")
        print(f"Updated {readme_path.relative_to(ROOT)}")
        return

    print(rendered)


if __name__ == "__main__":
    main()
