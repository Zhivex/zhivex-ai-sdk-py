import asyncio

from pydantic import BaseModel

from zhivex_ai import create_openai, stream_object


class Recipe(BaseModel):
    title: str
    servings: int


async def main() -> None:
    provider = create_openai()
    result = stream_object(
        model=provider("gpt-4o-mini"),
        prompt="Return a tiny JSON recipe with title and servings.",
        schema=Recipe,
    )

    async for partial in result.partial_object_stream():
        print("partial:", partial)

    final = await result.collect()
    print("final:", final.object.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
