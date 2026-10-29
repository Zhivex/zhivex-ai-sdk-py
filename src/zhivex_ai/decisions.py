"""Experimental decision models and gateway.decide.

This contract is independent of language-model chat routing. OpenAI Decisions
currently accepts ``gpt-6-luna`` only. See
https://developers.openai.com/api/docs/guides/decisions.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, NoReturn
from urllib.parse import urlparse

import httpx

from .errors import ConfigurationError, ProviderHTTPError, ValidationError

_MAX_BYTES = 1024 * 1024
_DATA_URL = re.compile(r"^data:image/(png|jpeg|webp|gif);base64,(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$")
_FALLBACK_STATUSES = {429, 500, 502, 503, 504}
DecisionQuestionType = Literal["predicate", "choice", "score"]


def _invalid() -> NoReturn:
    raise ValidationError("Invalid experimental decision request or response.")


def _record(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


def _text(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _probability(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0 <= float(value) <= 1


def _keys(value: Mapping[str, Any], allowed: set[str]) -> bool:
    return set(value).issubset(allowed)


def _token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid()
    return int(value)


@dataclass(frozen=True, slots=True)
class DecisionCapabilities:
    question_types: tuple[str, ...]
    input_types: tuple[str, ...]
    choice_value_types: tuple[str, ...]
    per_question_refusal: bool
    text_parts: bool


@dataclass(slots=True)
class DecisionInput:
    input: Any
    questions: dict[str, dict[str, Any]]
    timeout_ms: int | None = None
    cancel_event: asyncio.Event | None = None


@dataclass(slots=True)
class DecisionResult:
    answers: dict[str, dict[str, Any]]
    provenance: dict[str, str]
    usage: dict[str, int]
    request_id: str | None = None
    latency_ms: int | None = None


class DecisionModel:
    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        endpoint: str,
        capabilities: DecisionCapabilities,
        validate: Callable[[DecisionInput], None],
        decide: Callable[[DecisionInput], Awaitable[DecisionResult]],
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.endpoint = endpoint
        self.capabilities = capabilities
        self.validate = validate
        self.decide = decide


def snapshot_decision(decision_input: DecisionInput) -> DecisionInput:
    try:
        encoded = json.dumps(
            {"input": decision_input.input, "questions": decision_input.questions},
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        _invalid()
    if len(encoded.encode("utf-8")) > _MAX_BYTES:
        _invalid()
    payload = json.loads(encoded)
    return DecisionInput(
        input=payload["input"],
        questions=payload["questions"],
        timeout_ms=decision_input.timeout_ms,
        cancel_event=decision_input.cancel_event,
    )


def validate_decision(decision_input: DecisionInput, capabilities: DecisionCapabilities) -> None:
    if not isinstance(decision_input, DecisionInput):
        _invalid()
    if decision_input.timeout_ms is not None and (
        isinstance(decision_input.timeout_ms, bool)
        or not isinstance(decision_input.timeout_ms, int)
        or decision_input.timeout_ms <= 0
        or decision_input.timeout_ms > 86_400_000
    ):
        _invalid()
    evidence = decision_input.input
    if isinstance(evidence, str):
        if not _text(evidence) or "text" not in capabilities.input_types:
            _invalid()
    else:
        if not isinstance(evidence, list) or not evidence or not capabilities.text_parts:
            _invalid()
        for part in evidence:
            part_record = _record(part)
            if part_record is None or part_record.get("type") not in capabilities.input_types:
                _invalid()
            if part_record["type"] == "text":
                if not _keys(part_record, {"type", "text"}) or not _text(part_record.get("text")):
                    _invalid()
            elif part_record["type"] == "image":
                data_key = "data_url" if "data_url" in part_record else "dataURL" if "dataURL" in part_record else ""
                data_url = part_record.get(data_key) if data_key else None
                if not data_key or not _keys(part_record, {"type", data_key}):
                    _invalid()
                if not isinstance(data_url, str) or _DATA_URL.fullmatch(data_url) is None or data_url.endswith(","):
                    _invalid()
            else:
                _invalid()
    questions = decision_input.questions
    if not isinstance(questions, dict) or not questions:
        _invalid()
    for question_id, question in questions.items():
        question_record = _record(question)
        if not _text(question_id) or question_record is None or not _text(question_record.get("instructions")):
            _invalid()
        if question_record.get("type") not in capabilities.question_types:
            _invalid()
        if question_record["type"] == "predicate":
            if not _keys(question_record, {"type", "instructions"}):
                _invalid()
        elif question_record["type"] == "choice":
            choices = question_record.get("choices")
            if not _keys(question_record, {"type", "instructions", "choices"}) or not isinstance(choices, list) or len(choices) < 2:
                _invalid()
            seen: set[object] = set()
            for choice in choices:
                choice_record = _record(choice)
                if choice_record is None or not _keys(choice_record, {"value", "description"}):
                    _invalid()
                value = choice_record.get("value")
                value_type = "string" if isinstance(value, str) else "boolean" if isinstance(value, bool) else ""
                if value_type not in capabilities.choice_value_types or (isinstance(value, str) and not _text(value)) or not _text(choice_record.get("description")) or value in seen:
                    _invalid()
                seen.add(value)
        elif question_record["type"] == "score":
            levels = question_record.get("levels")
            if (
                not _keys(question_record, {"type", "instructions", "levels"})
                or not isinstance(levels, list)
                or len(levels) < 2
                or any(not _text(level) for level in levels)
                or len(set(levels)) != len(levels)
            ):
                _invalid()
        else:
            _invalid()
    snapshot_decision(decision_input)


def validate_decision_answers(answers: object, questions: Mapping[str, Mapping[str, Any]], refusal: bool) -> None:
    answer_record = _record(answers)
    if answer_record is None or len(answer_record) != len(questions):
        _invalid()
    for question_id, question in questions.items():
        if question_id not in answer_record:
            _invalid()
        answer = _record(answer_record[question_id])
        if answer is None:
            _invalid()
        if answer.get("type") == "refusal":
            if not refusal or not _keys(answer, {"type"}):
                _invalid()
            continue
        if answer.get("type") != question.get("type"):
            _invalid()
        if question["type"] == "predicate":
            if not _probability(answer.get("probability")):
                _invalid()
            continue
        if "provider_confidence" in answer and not _probability(answer.get("provider_confidence")):
            _invalid()
        if "providerConfidence" in answer and not _probability(answer.get("providerConfidence")):
            _invalid()
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, list):
            _invalid()
        expected: list[Any]
        if question["type"] == "choice":
            expected = [choice["value"] for choice in question["choices"]]
        else:
            expected = list(range(len(question["levels"])))
        if len(probabilities) != len(expected):
            _invalid()
        seen: set[object] = set()
        total = 0.0
        for item in probabilities:
            item_record = _record(item)
            if item_record is None or item_record.get("value") not in expected or item_record.get("value") in seen or not _probability(item_record.get("probability")):
                _invalid()
            if question["type"] == "score" and item_record.get("label") != question["levels"][item_record["value"]]:
                _invalid()
            seen.add(item_record["value"])
            total += float(item_record["probability"])
        if abs(total - 1) > 0.01:
            _invalid()
        if question["type"] == "choice":
            if answer.get("choice") not in expected:
                _invalid()
        else:
            score = answer.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)) or float(score) < 0 or float(score) > len(expected) - 1:
                _invalid()


async def _decision_operation(decision_input: DecisionInput, operation: Callable[[], Awaitable[DecisionResult]]) -> DecisionResult:
    timeout_s = (decision_input.timeout_ms if decision_input.timeout_ms is not None else 30_000) / 1000
    cancel_event = decision_input.cancel_event or asyncio.Event()

    async def runner() -> DecisionResult:
        return await operation()

    task = asyncio.create_task(runner())
    timer = asyncio.create_task(asyncio.sleep(timeout_s))
    cancelled = asyncio.create_task(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait({task, timer, cancelled}, return_when=asyncio.FIRST_COMPLETED)
        if task not in done:
            task.cancel()
            message = "Decision operation cancelled or timed out."
            raise TimeoutError(message)
        return task.result()
    except ProviderHTTPError as error:
        raise ProviderHTTPError(f"Decision request failed ({error.status}).", error.status) from None
    except ValidationError:
        raise ValidationError("Invalid experimental decision request or response.") from None
    except TimeoutError:
        raise
    except Exception:
        raise RuntimeError("Decision operation failed.") from None
    finally:
        for pending in (timer, cancelled):
            pending.cancel()
        if not task.done():
            task.cancel()


def _trusted_endpoint(url: str, *, allow_unsafe: bool) -> str:
    parsed = urlparse(url)
    if parsed.username or parsed.password:
        raise ConfigurationError("OpenAI decision endpoint must not include credentials.")
    if parsed.scheme != "https" and not allow_unsafe:
        raise ConfigurationError("OpenAI decision endpoint must use https.")
    if not parsed.netloc:
        raise ConfigurationError("OpenAI decision endpoint must include a host.")
    return url


def create_openai_decision_model(
    model_id: str,
    *,
    api_key: str,
    base_url: str,
    fetch: Callable[..., Awaitable[Any]] | None = None,
    allow_unsafe_endpoints: bool = False,
) -> DecisionModel:
    if model_id != "gpt-6-luna":
        raise ConfigurationError("OpenAI Decisions currently supports gpt-6-luna only.")
    trimmed = base_url.rstrip("/")
    endpoint = _trusted_endpoint(f"{trimmed}/decisions", allow_unsafe=allow_unsafe_endpoints)
    capabilities = DecisionCapabilities(
        question_types=("predicate", "choice", "score"),
        input_types=("text", "image"),
        choice_value_types=("string", "boolean"),
        per_question_refusal=True,
        text_parts=True,
    )

    def prepare(decision_input: DecisionInput) -> tuple[DecisionInput, str, list[dict[str, Any]]]:
        validate_decision(decision_input, capabilities)
        snapshot = snapshot_decision(decision_input)
        evidence = snapshot.input
        if isinstance(evidence, list) and sum(1 for part in evidence if part.get("type") == "image") > 128:
            _invalid()
        questions = []
        for name, question in snapshot.questions.items():
            if question["type"] == "score":
                questions.append({"name": name, "type": question["type"], "instructions": question["instructions"], "levels": [{"label": label} for label in question["levels"]]})
            else:
                questions.append({"name": name, **question})
        if isinstance(evidence, str):
            provider_input: Any = evidence
        else:
            content = []
            for part in evidence:
                if part["type"] == "text":
                    content.append({"type": "input_text", "text": part["text"]})
                else:
                    content.append({"type": "input_image", "image_url": part.get("data_url", part.get("dataURL"))})
            provider_input = [{"role": "user", "content": content}]
        body = json.dumps({"model": model_id, "input": provider_input, "questions": questions}, separators=(",", ":"))
        if len(body.encode("utf-8")) > _MAX_BYTES:
            _invalid()
        return snapshot, body, questions

    def validate(decision_input: DecisionInput) -> None:
        prepare(decision_input)

    async def decide(decision_input: DecisionInput) -> DecisionResult:
        snapshot, body, questions = prepare(decision_input)

        async def operation() -> DecisionResult:
            if fetch is None:
                async with httpx.AsyncClient(follow_redirects=False, timeout=httpx.Timeout((decision_input.timeout_ms or 30_000) / 1000)) as client:
                    response = await client.post(
                        endpoint,
                        headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"},
                        content=body,
                    )
                    status = response.status_code
                    payload = response.json() if status < 300 else None
                    text = response.text
            else:
                response = await fetch(endpoint, method="POST", headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"}, body=body)
                status = int(getattr(response, "status_code", getattr(response, "status", 0)))
                payload = response.json() if hasattr(response, "json") and status < 300 else None
                text = getattr(response, "text", "")
            if status >= 300:
                raise ProviderHTTPError("Decision request failed.", status, response_body=None if status < 400 else str(text)[:200])
            parsed = payload if isinstance(payload, dict) else None
            if parsed is None or parsed.get("model") != model_id or not isinstance(parsed.get("answers"), list) or len(parsed["answers"]) != len(questions) or not isinstance(parsed.get("usage"), dict):
                _invalid()
            answers: dict[str, dict[str, Any]] = {}
            for answer in parsed["answers"]:
                answer_record = _record(answer)
                if answer_record is None or not isinstance(answer_record.get("name"), str) or answer_record["name"] not in snapshot.questions or answer_record["name"] in answers:
                    _invalid()
                name = answer_record["name"]
                if answer_record.get("type") == "refusal":
                    if any(key not in {"type", "name"} for key in answer_record):
                        _invalid()
                    answers[name] = {"type": "refusal"}
                elif answer_record.get("type") == "predicate":
                    answers[name] = {"type": "predicate", "probability": answer_record.get("probability")}
                elif answer_record.get("type") in {"choice", "score"}:
                    if not _probability(answer_record.get("confidence")):
                        _invalid()
                    if answer_record["type"] == "choice":
                        answers[name] = {
                            "type": "choice",
                            "choice": answer_record.get("choice"),
                            "probabilities": answer_record.get("probabilities"),
                            "provider_confidence": answer_record.get("confidence"),
                        }
                    else:
                        answers[name] = {
                            "type": "score",
                            "score": answer_record.get("score"),
                            "probabilities": answer_record.get("probabilities"),
                            "provider_confidence": answer_record.get("confidence"),
                        }
                else:
                    _invalid()
            validate_decision_answers(answers, snapshot.questions, True)
            for answer in answers.values():
                if answer["type"] == "choice":
                    answer["probabilities"] = [{"value": item["value"], "probability": item["probability"]} for item in answer["probabilities"]]
                if answer["type"] == "score":
                    answer["probabilities"] = [{"value": item["value"], "label": item["label"], "probability": item["probability"]} for item in answer["probabilities"]]
                    weighted = sum(float(item["value"]) * float(item["probability"]) for item in answer["probabilities"])
                    if abs(float(answer["score"]) - weighted) > 0.01 * len(answer["probabilities"]):
                        _invalid()
            usage_payload = parsed["usage"]
            input_details = usage_payload.get("input_tokens_details")
            output_details = usage_payload.get("output_tokens_details")
            if not isinstance(input_details, dict) or not isinstance(output_details, dict):
                _invalid()
            usage = {
                "input_tokens": _token_count(usage_payload.get("input_tokens")),
                "output_tokens": _token_count(usage_payload.get("output_tokens")),
                "total_tokens": _token_count(usage_payload.get("total_tokens")),
                "cached_input_tokens": _token_count(input_details.get("cached_tokens")),
                "cache_write_tokens": _token_count(input_details.get("cache_write_tokens")),
                "reasoning_tokens": _token_count(output_details.get("reasoning_tokens")),
            }
            if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"] or usage["cached_input_tokens"] + usage["cache_write_tokens"] > usage["input_tokens"] or usage["reasoning_tokens"] > usage["output_tokens"]:
                _invalid()
            return DecisionResult(
                answers=answers,
                provenance={"provider": "openai", "model_id": model_id, "endpoint": endpoint, "protocol": "openai.decisions"},
                usage=usage,
            )

        return await _decision_operation(decision_input, operation)

    return DecisionModel(provider="openai", model_id=model_id, endpoint=endpoint, capabilities=capabilities, validate=validate, decide=decide)


def _decimal_usd(value: float) -> Decimal:
    return Decimal(json.dumps(value))


def _number_usd(amount: Decimal) -> float:
    value = float(amount)
    if not math.isfinite(value):
        raise ValidationError("Invalid decision reservation sum.")
    if amount > _decimal_usd(value):
        return math.nextafter(value, math.inf)
    return value


@dataclass(slots=True)
class GatewayDecisionTarget:
    model: DecisionModel
    reserve_usd: float | None = None


@dataclass(slots=True)
class GatewayDecisionResult(DecisionResult):
    routing: dict[str, Any] = field(default_factory=dict)


def create_gateway_decide(
    registry: Mapping[str, GatewayDecisionTarget] | None = None,
    *,
    default_timeout_ms: int | None = None,
) -> Callable[..., Awaitable[GatewayDecisionResult]]:
    registrations = {key: GatewayDecisionTarget(model=value.model, reserve_usd=value.reserve_usd) for key, value in (registry or {}).items()}

    async def decide(
        *,
        primary: str,
        input: Any,
        questions: dict[str, dict[str, Any]],
        alternatives: Sequence[str] | None = None,
        fallback_on: Sequence[int] | None = None,
        max_attempts: int | None = None,
        max_reserved_usd: float | None = None,
        allowed_providers: Sequence[str] | None = None,
        allowed_endpoints: Sequence[str] | None = None,
        require_per_question_refusal: bool | None = None,
        timeout_ms: int | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> GatewayDecisionResult:
        for values in (alternatives, allowed_providers, allowed_endpoints):
            if values is not None and (not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or any(not isinstance(value, str) or not value for value in values)):
                raise ValidationError("Invalid decision destination restrictions.")
        if require_per_question_refusal is not None and not isinstance(require_per_question_refusal, bool):
            raise ValidationError("Invalid decision routing controls.")
        if fallback_on is not None and not isinstance(fallback_on, Sequence):
            raise ValidationError("Invalid decision routing controls.")
        ids = [primary, *(alternatives or ())]
        attempts = 1 if max_attempts is None else max_attempts
        statuses = list(fallback_on or ())
        if (
            any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids)
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
            or attempts > len(ids)
            or any(status not in _FALLBACK_STATUSES for status in statuses)
            or (max_reserved_usd is not None and (isinstance(max_reserved_usd, bool) or not isinstance(max_reserved_usd, (int, float)) or not math.isfinite(max_reserved_usd) or max_reserved_usd < 0))
        ):
            raise ValidationError("Invalid decision routing controls.")
        decision_input = snapshot_decision(DecisionInput(input=input, questions=questions, timeout_ms=timeout_ms if timeout_ms is not None else default_timeout_ms, cancel_event=cancel_event))
        targets = []
        for destination_id in ids:
            target = registrations.get(destination_id)
            if target is None:
                raise ConfigurationError("Unknown decision destination.")
            model = target.model
            if allowed_providers is not None and model.provider not in allowed_providers:
                raise ConfigurationError("Decision destination violates explicit restrictions.")
            if allowed_endpoints is not None and model.endpoint not in allowed_endpoints:
                raise ConfigurationError("Decision destination violates explicit restrictions.")
            if require_per_question_refusal and not model.capabilities.per_question_refusal:
                raise ConfigurationError("Decision destination violates explicit restrictions.")
            if target.reserve_usd is not None and (isinstance(target.reserve_usd, bool) or not isinstance(target.reserve_usd, (int, float)) or not math.isfinite(float(target.reserve_usd)) or float(target.reserve_usd) < 0):
                raise ConfigurationError("A valid decision cost reservation is required for each selected destination.")
            if max_reserved_usd is not None and target.reserve_usd is None:
                raise ConfigurationError("A valid decision cost reservation is required for each selected destination.")
            validate_decision(decision_input, model.capabilities)
            model.validate(decision_input)
            targets.append(target)
        if max_reserved_usd is not None and float(targets[0].reserve_usd or 0) > max_reserved_usd:
            raise ValidationError("Decision reservation ceiling exceeded.")

        async def operation() -> DecisionResult:
            reserved = Decimal(0)
            known = True
            ceiling = None if max_reserved_usd is None else _decimal_usd(float(max_reserved_usd))
            for index in range(attempts):
                if decision_input.cancel_event is not None and decision_input.cancel_event.is_set():
                    raise RuntimeError("Decision cancelled.")
                target = targets[index]
                if target.reserve_usd is None:
                    known = False
                reserved += _decimal_usd(float(target.reserve_usd or 0))
                if ceiling is not None and reserved > ceiling:
                    raise ValidationError("Decision reservation ceiling exceeded.")
                reported = _number_usd(reserved)
                try:
                    result = await target.model.decide(decision_input)
                except ProviderHTTPError as error:
                    if index + 1 >= attempts or error.status not in statuses:
                        raise
                    continue
                if result.provenance.get("provider") != target.model.provider or result.provenance.get("model_id") != target.model.model_id or result.provenance.get("endpoint") != target.model.endpoint:
                    raise ValidationError("Invalid decision provenance.")
                validate_decision_answers(result.answers, decision_input.questions, target.model.capabilities.per_question_refusal)
                _token_count(result.usage.get("input_tokens"))
                routing = {"target": ids[index], "attempts": index + 1}
                if known:
                    routing["reserved_usd"] = reported
                return GatewayDecisionResult(answers=result.answers, provenance=result.provenance, usage=result.usage, request_id=result.request_id, latency_ms=result.latency_ms, routing=routing)
            raise RuntimeError("Decision attempts exhausted.")

        result = await _decision_operation(decision_input, operation)
        assert isinstance(result, GatewayDecisionResult)
        return result

    return decide
