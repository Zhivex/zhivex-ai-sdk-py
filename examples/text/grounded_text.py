import asyncio

from zhivex_ai import create_openai, generate_grounded_text


async def main() -> None:
    provider = create_openai()
    result = await generate_grounded_text(
        model=provider.grounded_language_model("gpt-5.4-mini"),
        prompt="Find one recent fact about Buenos Aires tech news.",
    )

    print(result.text)
    for source in result.sources:
        print(source.title, source.url)


if __name__ == "__main__":
    asyncio.run(main())
