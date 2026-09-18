"""Opt-in Qwen Omni integration smoke; prints only allowlisted operation outcomes.

Run from the checkout with .venv/bin/python scripts/smoke_qwen_omni.py.
Uses .env credentials, synthetic inline media, and optional QWEN_OMNI_SMOKE_VIDEO_PATH.
This is local integration evidence, not exact-wheel release certification.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
from email.parser import BytesParser
from importlib import metadata
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import struct
import sys
import wave
import zlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
USE_INSTALLED = os.getenv("ZHIVEX_SMOKE_USE_INSTALLED") == "1"
if not USE_INSTALLED:
    sys.path.insert(0, str(ROOT / "src"))

import zhivex_ai  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from zhivex_ai import (  # noqa: E402
    Agent, FilePart, ImagePart, ModelMessage, ReasoningConfig, TextPart,
    aclose_default_clients, create_qwen, generate_object, generate_text,
    qwen_web_search_tool, run_agent, stream_text, tool,
)


def verify_installed_wheel(wheel: Path) -> dict[str, str]:
    package = Path(zhivex_ai.__file__).resolve().parent
    if package.is_relative_to(ROOT / "src"):
        raise RuntimeError("Installed smoke must not import the checkout.")
    version = metadata.version("zhivex-ai-sdk")
    if wheel.name != f"zhivex_ai_sdk-{version}-py3-none-any.whl":
        raise RuntimeError("Wheel version differs from the installed package.")
    with zipfile.ZipFile(wheel) as archive:
        wheel_metadata = BytesParser().parsebytes(archive.read(f"zhivex_ai_sdk-{version}.dist-info/METADATA"))
        if wheel_metadata["Version"] != version or wheel_metadata["Name"] != "zhivex-ai-sdk":
            raise RuntimeError("Wheel metadata differs from the installed package.")
        members = [name for name in archive.namelist() if name.startswith("zhivex_ai/") and name.endswith(".py")]
        if not members:
            raise RuntimeError("Wheel has no SDK modules.")
        for name in members:
            installed = package / name.removeprefix("zhivex_ai/")
            if installed.read_bytes() != archive.read(name):
                raise RuntimeError("Installed SDK bytes differ from the supplied wheel.")
    return {"filename": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "package_version": version}


def synthetic_audio(channels: int = 1) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000))) * channels for i in range(16000)))
    return base64.b64encode(buffer.getvalue()).decode()


def synthetic_image() -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


class Answer(BaseModel):
    answer: str


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write a redacted local integration report.")
    parser.add_argument("--wheel", type=Path, help="Required exact wheel when ZHIVEX_SMOKE_USE_INSTALLED=1.")
    parser.add_argument("--video", type=Path, default=Path(os.getenv("QWEN_OMNI_SMOKE_VIDEO_PATH", str(ROOT / "tests/fixtures/qwen_omni_synthetic.mp4"))))
    args = parser.parse_args()
    artifact = None
    if USE_INSTALLED:
        if args.wheel is None:
            parser.error("--wheel is required for installed-package smoke")
        artifact = verify_installed_wheel(args.wheel)
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(ROOT / ".env")
    provider = create_qwen(base_url=os.getenv("QWEN_BASE_URL"), responses_base_url=os.getenv("QWEN_RESPONSES_BASE_URL"))
    model = provider.native.language_model("qwen3.8-omni-flash")
    common = dict(model=model, timeout_ms=45000, max_retries=0)
    outcomes: dict[str, str] = {}

    async def check(name, operation):
        try:
            await operation()
            outcomes[name] = "passed"
        except Exception as exc:
            outcomes[name] = "failed:" + type(exc).__name__
            status = getattr(exc, "status", None)
            if isinstance(status, int):
                outcomes[name] += ":" + str(status)
        print(json.dumps({"operation": name, "result": outcomes[name]}), flush=True)

    async def text():
        result = await generate_text(**common, prompt="Reply exactly OK.", reasoning=ReasoningConfig(effort="none"), max_tokens=64)
        assert "OK" in result.text.upper()

    async def streaming():
        result = await stream_text(**common, prompt="Reply exactly OK.", reasoning=ReasoningConfig(effort="low"), max_tokens=1024).collect()
        assert "OK" in result.text

    async def media(parts):
        result = await generate_text(**common, messages=[ModelMessage(role="user", parts=[TextPart(text="Briefly describe this synthetic input."), *parts])], reasoning=ReasoningConfig(effort="none"), max_tokens=256)
        assert result.text.strip()

    async def budget():
        result = await generate_text(**common, prompt="Reply OK.", reasoning=ReasoningConfig(budget_tokens=1024), max_tokens=2048)
        assert result.text.strip()

    async def budget_media():
        result = await generate_text(**common, messages=[ModelMessage(role="user", parts=[TextPart(text="Describe this synthetic tone briefly."), audio])], reasoning=ReasoningConfig(budget_tokens=1024), max_tokens=2048)
        assert result.text.strip()

    async def budget_search():
        result = await generate_text(**common, prompt="Search the web for Alibaba Cloud Model Studio documentation and summarize briefly.", tools={"search": qwen_web_search_tool()}, reasoning=ReasoningConfig(budget_tokens=1024), max_tokens=2048)
        assert result.text.strip()

    async def stream_media(parts):
        result = await stream_text(**common, messages=[ModelMessage(role="user", parts=[TextPart(text="Briefly describe this synthetic input."), *parts])], reasoning=ReasoningConfig(effort="none"), max_tokens=256).collect()
        assert result.text.strip()
        assert result.usage is not None

    async def structured():
        result = await generate_object(**common, prompt='Return answer equal to OK.', schema=Answer, mode="prompted", reasoning=ReasoningConfig(effort="none"), max_tokens=128)
        assert result.object.answer == "OK"

    async def agent():
        calls = []
        def verify(code: str) -> str:
            assert code == "omni-canary-731"
            calls.append(code)
            return "VERIFIED-731"
        runner = Agent(name="omni-smoke", model=model, tools={"verify": tool(verify)}, instructions="Call verify once with code omni-canary-731, then reply with its result. Do not call it again.")
        result = await run_agent(agent=runner, prompt="Verify now.", reasoning=ReasoningConfig(effort="none"), max_steps=3, max_tokens=256, timeout_ms=45000, max_retries=0)
        assert calls == ["omni-canary-731"]
        assert "VERIFIED-731" in result.text

    async def search():
        result = await generate_text(**common, prompt="Use web search to find Alibaba Cloud Model Studio documentation. Give one sentence.", tools={"search": qwen_web_search_tool()}, reasoning=ReasoningConfig(effort="none"), max_tokens=512)
        assert result.text.strip()
        assert any("web_search_call" in json.dumps(step.response.raw_response) for step in result.steps)

    audio = FilePart(data=synthetic_audio(), media_type="audio/wav")
    picture = ImagePart(image=synthetic_image())
    try:
        await check("text", text)
        await check("streaming-thinking", streaming)
        await check("image", lambda: media([picture]))
        await check("audio", lambda: media([audio]))
        await check("mixed-image-audio", lambda: media([picture, audio]))
        await check("multichannel-audio", lambda: media([FilePart(data=synthetic_audio(2), media_type="audio/wav", provider_metadata={"qwen": {"use_multichannel": True}})]))
        await check("budget-chat", budget)
        await check("budget-chat-audio", budget_media)
        await check("budget-chat-search-request", budget_search)
        await check("prompted-json", structured)
        await check("agent-tool-replay", agent)
        await check("web-search", search)
        video_path = args.video
        if video_path.is_file():
            video = FilePart(data=base64.b64encode(Path(video_path).read_bytes()).decode(), media_type="video/mp4")
            await check("video", lambda: media([video]))
            await check("mixed-image-audio-video", lambda: media([picture, audio, video]))
            await check("mixed-streaming", lambda: stream_media([picture, audio, video]))
        else:
            outcomes["video"] = "blocked:missing-synthetic-video"
    finally:
        await aclose_default_clients()
    root = Path(__file__).resolve().parents[1]
    source_files = ["src/zhivex_ai/providers/qwen.py", "src/zhivex_ai/providers/_qwen_omni.py", "src/zhivex_ai/providers/openai_compat.py", "src/zhivex_ai/catalog.py", "scripts/smoke_qwen_omni.py"]
    report = {
        "model": "qwen3.8-omni-flash", "provider": "qwen",
        "evidence": "installed-wheel-local-integration" if USE_INSTALLED else "local-integration",
        "artifact": artifact, "release_certified": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
        "operations": outcomes,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if any(result != "passed" for result in outcomes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
