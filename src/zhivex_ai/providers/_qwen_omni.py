"""Qwen3.8 Omni's media contract (distinct from generic Responses files)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..errors import UnsupportedFeatureError, ValidationError
from ..messages import validate_file_part
from ..types import FilePart, ModelGenerateInput, ReasoningConfig, ToolChoiceName

OMNI_MODEL = "qwen3.8-omni-flash"


def is_qwen_omni(model_id: str) -> bool:
    return model_id == OMNI_MODEL


def omni_file_content(part: FilePart, *, chat: bool = False) -> dict[str, Any]:
    validate_file_part(part)
    media_type = (part.media_type or "").lower()
    if not media_type.startswith(("image/", "audio/", "video/")):
        raise ValidationError("Qwen Omni requires an image, audio, or video media_type.")
    if part.text is not None or part.document_content is not None or part.file_id or part.file_uri:
        raise ValidationError("Qwen Omni media requires url or base64 data, not a file reference or document.")
    if (part.url is None) == (part.data is None):
        raise ValidationError("Qwen Omni media requires exactly one of url or base64 data.")
    source = part.url or f"data:{media_type};base64,{part.data}"
    if media_type.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": source}} if chat else {"type": "input_image", "image_url": source}
    if media_type.startswith("video/"):
        return {"type": "video_url", "video_url": {"url": source}} if chat else {"type": "input_video", "video_url": source}
    audio_format = {"mpeg": "mp3", "x-wav": "wav", "wave": "wav"}.get(media_type[6:], media_type[6:])
    payload: dict[str, Any] = {"data": source, "format": audio_format} if chat else {"type": "input_audio", "audio_url": source, "format": audio_format}
    options = part.provider_metadata.get("qwen", {})
    if not isinstance(options, dict) or set(options) - {"use_multichannel"}:
        raise ValidationError("Qwen Omni audio metadata supports only use_multichannel.")
    if "use_multichannel" in options:
        if not isinstance(options["use_multichannel"], bool):
            raise ValidationError("Qwen Omni use_multichannel must be a boolean.")
        payload["use_multichannel"] = options["use_multichannel"]
    return {"type": "input_audio", "input_audio": payload} if chat else payload


def validate_omni_input(input: ModelGenerateInput) -> None:
    for message in input.messages:
        if message.role != "user" and any(part.type in {"image", "file"} for part in message.parts):
            raise ValidationError("Qwen Omni media is allowed only in user messages.")
    options = input.provider_options or {}
    reserved = set(options) & {"model", "input", "messages", "instructions", "tool_choice", "text", "stream", "max_output_tokens", "temperature"}
    if reserved:
        raise ValidationError("Qwen Omni provider options cannot override SDK-owned fields: " + ", ".join(sorted(reserved)))
    if options.get("modalities", ["text"]) != ["text"] or "audio" in options:
        raise UnsupportedFeatureError("Qwen3.8-Omni-Flash generates text only.")
    if input.structured_output is not None and input.structured_output.mode == "native":
        raise UnsupportedFeatureError("Qwen Omni has no documented native JSON Schema contract; use prompted structured output.")
    if input.reasoning is not None and set(options) & {"reasoning", "reasoning_effort", "thinking_budget", "enable_thinking"}:
        raise ValidationError("Pass reasoning or Qwen reasoning provider options, not both.")


def prepare_omni_input(input: ModelGenerateInput) -> ModelGenerateInput:
    """Normalize native reasoning options before shared validation and route selection."""
    validate_omni_input(input)
    options = dict(input.provider_options or {})
    if "enable_thinking" in options:
        raise ValidationError("Qwen Omni uses reasoning effort none to disable thinking.")
    effort = options.pop("reasoning_effort", None)
    budget = options.pop("thinking_budget", None)
    raw = options.pop("reasoning", None)
    if raw is not None:
        if not isinstance(raw, dict) or set(raw) != {"effort"} or effort is not None or budget is not None:
            raise ValidationError("Qwen Omni reasoning options must specify exactly one effort or budget.")
        effort = raw["effort"]
    reasoning = input.reasoning
    if effort is not None or budget is not None:
        reasoning = ReasoningConfig(effort=effort, budget_tokens=budget)
    if reasoning is not None:
        if reasoning.effort is not None and reasoning.budget_tokens is not None:
            raise ValidationError("Qwen Omni does not accept effort and thinking_budget together.")
        if reasoning.effort is not None and reasoning.effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValidationError("Invalid Qwen Omni reasoning effort.")
        if reasoning.budget_tokens is not None and (isinstance(reasoning.budget_tokens, bool) or not isinstance(reasoning.budget_tokens, int) or reasoning.budget_tokens <= 0):
            raise ValidationError("Qwen Omni thinking_budget must be a positive integer.")
    if reasoning is not None and (reasoning.budget_tokens is not None or reasoning.effort not in {None, "none"}):
        if input.tool_choice == "required" or isinstance(input.tool_choice, ToolChoiceName):
            raise UnsupportedFeatureError("Qwen Omni cannot force a tool while thinking is enabled; use auto tool choice.")
    return replace(input, reasoning=reasoning, provider_options=options)
