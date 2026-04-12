from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    create_anthropic,
    create_azure_openai,
    create_bedrock,
    create_gemini,
    create_kimi,
    create_ollama,
    create_openai,
    create_openrouter,
    create_qwen,
    create_vertex,
)
from zhivex_ai.provider_support import build_provider_support_rows, render_provider_support_markdown


class _FakeBedrockClient:
    async def converse(self, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError


def main() -> None:
    rows = build_provider_support_rows(
        [
            create_openai(api_key="test"),
            create_azure_openai(api_key="test", endpoint="https://example.openai.azure.com"),
            create_anthropic(api_key="test"),
            create_gemini(api_key="test"),
            create_vertex(access_token="test", project_id="project"),
            create_bedrock(client=_FakeBedrockClient()),
            create_openrouter(api_key="test"),
            create_qwen(api_key="test"),
            create_kimi(api_key="test"),
            create_ollama(),
        ]
    )
    print(render_provider_support_markdown(rows))


if __name__ == "__main__":
    main()
