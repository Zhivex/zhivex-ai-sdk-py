import asyncio

from pydantic import BaseModel

from zhivex_ai import create_openai, generate_object


class Summary(BaseModel):
    title: str
    sentiment: str


async def main() -> None:
    provider = create_openai()
    result = await generate_object(
        model=provider("gpt-4o-mini"),
        prompt="Return a JSON summary for: Zhivex AI SDK helps unify multiple LLM providers.",
        schema=Summary,
    )
    print(result.object.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
