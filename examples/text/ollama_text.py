import asyncio
import os

from zhivex_ai import create_ollama, generate_text


async def main() -> None:
    provider = create_ollama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )
    result = await generate_text(
        model=provider.native.language_model(os.getenv("OLLAMA_MODEL", "llama3.2")),
        prompt="Explain Zhivex AI SDK in one sentence.",
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
