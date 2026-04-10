import asyncio

from zhivex_ai import create_gemini, generate_grounded_text


async def main() -> None:
    gemini = create_gemini()
    result = await generate_grounded_text(
        model=gemini.grounded_language_model("gemini-2.5-flash"),
        prompt="Find one recent fact about AI infrastructure and cite the source.",
    )

    print(result.text)
    print()
    print("Sources:")
    for source in result.sources:
        print(f"- {source.title}: {source.url}")


if __name__ == "__main__":
    asyncio.run(main())
