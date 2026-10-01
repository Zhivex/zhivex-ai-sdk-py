"""Native Qwen text reranking. Export QWEN_API_KEY and DASHSCOPE_WORKSPACE_ID."""

import asyncio
import os

from zhivex_ai import create_qwen


async def main() -> None:
    provider = create_qwen(region="cn", workspace_id=os.environ["DASHSCOPE_WORKSPACE_ID"])
    documents = ["The sky is blue.", "Reset your password from the account settings page."]
    response = await provider.native.rerank().create(
        model="qwen3.7-text-rerank", query="How do I reset my password?",
        documents=documents, top_n=1,
    )
    for result in response["output"]["results"]:
        print(result["relevance_score"], documents[result["index"]])


if __name__ == "__main__":
    asyncio.run(main())
