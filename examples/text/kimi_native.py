from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import FilePart, ImagePart, ReasoningConfig, create_kimi, generate_text, user  # noqa: E402


TINY_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/axzD7kAAAAASUVORK5CYII="
)


async def main() -> None:
    kimi = create_kimi()
    model = os.getenv("KIMI_MODEL", "kimi-k3")

    chat = await generate_text(
        model=kimi.native.language_model(model),
        prompt="Explain Kimi support in Zhivex AI SDK in one sentence.",
        reasoning=ReasoningConfig(effort="high"),
    )
    print("chat:", chat.text)

    image = await generate_text(
        model=kimi.native.language_model(model),
        messages=[user([ImagePart(image=TINY_PNG)])],
        prompt="Describe the image in one short phrase.",
        reasoning=ReasoningConfig(effort="high"),
    )
    print("image:", image.text)

    file_path = os.getenv("KIMI_FILE_PATH")
    if file_path:
        source = Path(file_path)
        uploaded = await kimi.files().upload(data=source.read_bytes(), filename=source.name, purpose="file-extract")
        extracted = (await kimi.files().download(uploaded.id)).decode("utf-8", errors="replace")
        summary = await generate_text(
            model=kimi.native.language_model(model),
            system=extracted,
            prompt="Summarize the uploaded file in three bullets.",
            reasoning=ReasoningConfig(effort="high"),
        )
        print("file:", summary.text)

    if os.getenv("KIMI_RUN_BATCH") == "1":
        request = {
            "custom_id": "example-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "Say hello from a batch job."}]},
        }
        uploaded = await kimi.files().upload(
            data=(json.dumps(request) + "\n").encode("utf-8"),
            filename="kimi_batch.jsonl",
            media_type="application/jsonl",
            purpose="batch",
        )
        batch = await kimi.batches().create(
            {"input_file_id": uploaded.id, "endpoint": "/v1/chat/completions", "completion_window": "24h"}
        )
        print("batch:", batch.get("id"), batch.get("status"))

    tokens = await kimi.tokens().count(model_id=model, prompt="hello")
    print("tokens:", tokens.total_tokens)

    if os.getenv("KIMI_VIDEO_FILE_ID"):
        result = await generate_text(
            model=kimi.native.language_model(model),
            messages=[user([FilePart(file_id=os.environ["KIMI_VIDEO_FILE_ID"], media_type="video/mp4")])],
            prompt="Describe this video briefly.",
            reasoning=ReasoningConfig(effort="high"),
        )
        print("video:", result.text)


if __name__ == "__main__":
    asyncio.run(main())
