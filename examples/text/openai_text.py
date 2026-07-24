import asyncio

from zhivex_ai import create_openai, generate_text


async def main() -> None:
    provider = create_openai()
    result = await generate_text(
        model=provider("gpt-5.6-terra"),
        prompt="Explain Zhivex AI SDK in one sentence.",
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
