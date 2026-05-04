from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Literal, cast

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import (  # noqa: E402
    AudioInput,
    ReasoningConfig,
    create_qwen,
    embed,
    generate_speech,
    generate_text,
    qwen_web_search_tool,
    transcribe_audio,
)

QwenRegion = Literal["intl", "us", "cn"]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value and value.strip() else default


async def main() -> None:
    region = cast(QwenRegion, _env("QWEN_REGION", "intl"))
    qwen = create_qwen(
        region=region,  # intl, us, or cn
        base_url=os.getenv("QWEN_BASE_URL"),
        responses_base_url=os.getenv("QWEN_RESPONSES_BASE_URL"),
    )
    chat_model = _env("QWEN_MODEL", "qwen3.6-plus")

    chat = await generate_text(
        model=qwen.native.language_model(chat_model),
        prompt="Explain Qwen support in Zhivex AI SDK in one sentence.",
        reasoning=ReasoningConfig(effort="none"),
    )
    print("chat:", chat.text)

    search = await generate_text(
        model=qwen.native.language_model(chat_model),
        prompt="Summarize Alibaba Cloud Model Studio hosted tools in one sentence.",
        tools={"search": qwen_web_search_tool()},
        reasoning=ReasoningConfig(effort="none"),
    )
    print("hosted tool:", search.text)

    embedding_model = _env("QWEN_EMBEDDING_MODEL", "text-embedding-v4")
    embedding = await embed(
        model=qwen.native.embedding_model(embedding_model),
        value="Zhivex AI SDK can route Qwen text, tools, embeddings, ASR, and TTS through native adapters.",
    )
    print("embedding dimension:", len(embedding.embedding))

    audio_path = os.getenv("QWEN_ASR_AUDIO_PATH")
    if audio_path:
        source = Path(audio_path)
        transcription = await transcribe_audio(
            model=qwen.native.transcription_model(_env("QWEN_ASR_MODEL", "qwen3-asr-flash")),
            audio=AudioInput(
                data=source.read_bytes(),
                media_type=_env("QWEN_ASR_MEDIA_TYPE", "audio/wav"),
                filename=source.name,
            ),
            prompt=os.getenv("QWEN_ASR_PROMPT"),
            language=os.getenv("QWEN_ASR_LANGUAGE"),
        )
        print("asr:", transcription.text)

    if os.getenv("QWEN_GENERATE_TTS") == "1":
        speech = await generate_speech(
            model=qwen.native.speech_model(_env("QWEN_TTS_MODEL", "qwen3-tts-flash")),
            input="Zhivex AI SDK supports Qwen speech generation through the native adapter.",
            provider_options={"language_type": _env("QWEN_TTS_LANGUAGE_TYPE", "English")},
        )
        output_path = Path(os.getenv("QWEN_TTS_OUTPUT", "examples/audio/qwen_speech.wav"))
        output_path.write_bytes(speech.audio)
        print("tts:", output_path, speech.media_type)


if __name__ == "__main__":
    asyncio.run(main())
