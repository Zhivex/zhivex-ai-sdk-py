"""Agent skills helpers."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ._agent_context import _message_text
from ._agent_contracts import AgentEvent as AgentEvent
from ._agent_contracts import AgentSession as AgentSession
from ._agent_contracts import AgentSkillActivatedEvent as AgentSkillActivatedEvent
from ._agent_contracts import (
    AgentSkillDependencyCheckEvent as AgentSkillDependencyCheckEvent,
)
from ._agent_contracts import AgentSkillResolvedEvent as AgentSkillResolvedEvent
from ._agent_contracts import AgentSkillSkippedEvent as AgentSkillSkippedEvent
from ._agent_contracts import SkillActivationMode as SkillActivationMode
from ._agent_tools import discover_mcp_tools as discover_mcp_tools
from ._agent_tools import mcp_http_server as mcp_http_server
from ._agent_tools import mcp_stdio_server as mcp_stdio_server
from .errors import ValidationError
from .skills import SkillArtifact, SkillDefinition, SkillRegistry, SkillSet
from .types import ModelMessage, ToolExecutionResult, ToolSet

if TYPE_CHECKING:
    from .agent import Agent


def _skill_reference_name(skill: SkillDefinition) -> str:
    return skill.display_name or skill.name


@dataclass(slots=True)
class _SkillActivation:
    skill: SkillDefinition
    mode: SkillActivationMode


@dataclass(slots=True)
class _SkillSkip:
    skill_name: str
    reason: str
    mode: SkillActivationMode
    path: str | None = None


def _normalize_skill_names(values: Iterable[Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _tokenize_skill_text(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", value.lower())
        if len(token) >= 3
    }


def _matches_any_phrase(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    for pattern in patterns:
        candidate = pattern.strip().lower()
        if candidate and candidate in lowered:
            return True
    return False


def _explicit_skill_requested(skill: SkillDefinition, text: str) -> bool:
    pattern = re.compile(rf"(?<!\w)\${re.escape(skill.name)}(?![\w-])", re.IGNORECASE)
    return bool(pattern.search(text))


def _should_activate_skill_implicitly(skill: SkillDefinition, text: str) -> bool:
    if not skill.allow_implicit_invocation:
        return False
    trigger_matched = not skill.triggers or _matches_any_phrase(text, skill.triggers)
    if not trigger_matched:
        return False
    if skill.anti_triggers and _matches_any_phrase(text, skill.anti_triggers):
        return False
    if skill.triggers:
        return True
    lowered = text.lower()
    if skill.name.lower() in lowered:
        return True
    prompt_tokens = _tokenize_skill_text(text)
    if not prompt_tokens:
        return False
    name_tokens = _tokenize_skill_text(skill.name.replace("-", " "))
    description_tokens = _tokenize_skill_text(skill.description)
    overlap = len(prompt_tokens & (name_tokens | description_tokens))
    return overlap >= 2 or (
        overlap >= 1 and len(name_tokens) == 1 and bool(prompt_tokens & name_tokens)
    )


def _skill_allowed_for_agent(
    skill: SkillDefinition, agent: Agent
) -> tuple[bool, str | None]:
    provider = str(getattr(agent.model, "provider", "") or "")
    model_id = str(getattr(agent.model, "model_id", "") or "")
    if skill.allowed_providers and provider not in skill.allowed_providers:
        return False, f'provider "{provider}" is not allowed'
    if skill.allowed_models and not any(
        fnmatch.fnmatch(model_id, pattern) for pattern in skill.allowed_models
    ):
        return False, f'model "{model_id}" is not allowed'
    return True, None


def _skill_activation_sort_key(item: _SkillActivation) -> tuple[int, int, str]:
    mode_order = {"explicit": 0, "sticky": 1, "implicit": 2}
    return (-item.skill.priority, mode_order[item.mode], item.skill.name)


def _skill_system_message(skill: SkillDefinition) -> str:
    lines = [
        f"[Active skill: {_skill_reference_name(skill)}]",
        f"Description: {skill.description}",
    ]
    if skill.version:
        lines.append(f"Version: {skill.version}")
    if skill.path:
        lines.append(f"Skill file: {skill.path}")
        skill_dir = str(Path(skill.path).resolve().parent)
        lines.append(f"Skill directory: {skill_dir}")
        for label, directory in (
            ("scripts", Path(skill_dir) / "scripts"),
            ("references", Path(skill_dir) / "references"),
            ("assets", Path(skill_dir) / "assets"),
        ):
            if directory.exists():
                lines.append(f"Available {label}: {directory}")
    if skill.resources:
        lines.append(f"Available resources: {', '.join(skill.resources)}")
    if skill.entrypoints:
        lines.append(
            f"Available entrypoints: {', '.join(item.name for item in skill.entrypoints)}"
        )
    if skill.default_prompt:
        lines.append(f"Suggested surrounding prompt: {skill.default_prompt}")
    lines.append("Follow these skill instructions for the current task:")
    lines.append(skill.instructions.strip())
    return "\n".join(lines).strip()


def _resolve_skill_registry(
    agent: Agent, extra_skills: SkillSet | SkillRegistry | None
) -> SkillRegistry:
    base = (
        agent.skills
        if isinstance(agent.skills, SkillRegistry)
        else SkillRegistry(agent.skills)
    )
    return base.merge(extra_skills)


def _sticky_skill_names(session: AgentSession) -> list[str]:
    raw = session.metadata.get("sticky_skills")
    if not isinstance(raw, list):
        return []
    return _normalize_skill_names(raw)


async def _resolve_skill_tools(skill: SkillDefinition) -> ToolSet:
    resolved: ToolSet = dict(skill.tools)
    if skill.entrypoints:
        from .skillpacks import build_skill_entrypoint_tools

        resolved.update(build_skill_entrypoint_tools(skill))
    for dependency in skill.dependencies:
        if dependency.type != "mcp":
            continue
        if dependency.transport == "stdio":
            if not dependency.command:
                raise ValidationError(
                    f'Skill "{skill.name}" requires "command" for stdio MCP dependencies.'
                )
            server = mcp_stdio_server(
                name=dependency.value,
                command=dependency.command,
                args=dependency.args,
                env=dependency.env,
                timeout_ms=dependency.timeout_ms,
            )
        else:
            if not dependency.url:
                raise ValidationError(
                    f'Skill "{skill.name}" requires "url" for HTTP MCP dependencies.'
                )
            server = mcp_http_server(
                name=dependency.value,
                url=dependency.url,
                headers=dependency.headers,
                timeout_ms=dependency.timeout_ms,
            )
        prefix = dependency.prefix or f"{skill.name}_{dependency.value}"
        discovered = await discover_mcp_tools(
            server,
            prefix=prefix,
            include=dependency.include or None,
            exclude=dependency.exclude or None,
        )
        for name, definition in discovered.items():
            if name in resolved and resolved[name] != definition:
                raise ValidationError(
                    f'Tool name collision while activating skill "{skill.name}": "{name}".'
                )
            resolved[name] = definition
    return resolved


async def _select_active_skills(
    registry: SkillRegistry,
    *,
    agent: Agent,
    session: AgentSession,
    prompt: str | None,
    messages: list[ModelMessage] | None,
) -> tuple[list[_SkillActivation], list[_SkillSkip], ToolSet]:
    text = prompt or ""
    if messages is not None:
        message_text = _message_text(messages)
        text = f"{text}\n{message_text}".strip()
    selected: dict[str, _SkillActivation] = {}
    skipped: list[_SkillSkip] = []
    sticky_names = _sticky_skill_names(session)

    if text and registry.items():
        for _, definition in registry.items():
            if _explicit_skill_requested(definition, text):
                selected[definition.name] = _SkillActivation(
                    skill=definition, mode="explicit"
                )
            elif _should_activate_skill_implicitly(definition, text):
                selected.setdefault(
                    definition.name, _SkillActivation(skill=definition, mode="implicit")
                )

    for sticky_name in sticky_names:
        if sticky_name in selected:
            continue
        sticky_skill = registry.get(sticky_name)
        if sticky_skill is None:
            skipped.append(
                _SkillSkip(
                    skill_name=sticky_name,
                    reason="sticky skill is no longer registered",
                    mode="sticky",
                )
            )
            continue
        selected[sticky_name] = _SkillActivation(skill=sticky_skill, mode="sticky")

    if not selected:
        return [], skipped, {}
    ordered_candidates = sorted(selected.values(), key=_skill_activation_sort_key)
    activations: list[_SkillActivation] = []
    resolved_tools: ToolSet = {}
    for activation in ordered_candidates:
        allowed, reason = _skill_allowed_for_agent(activation.skill, agent)
        if not allowed:
            if activation.mode == "explicit":
                raise ValidationError(
                    f'Explicit skill "{activation.skill.name}" could not be activated because {reason}.'
                )
            skipped.append(
                _SkillSkip(
                    skill_name=activation.skill.name,
                    reason=reason or "skill is not allowed for this agent",
                    mode=activation.mode,
                    path=activation.skill.path,
                )
            )
            continue
        try:
            candidate_tools = await _resolve_skill_tools(activation.skill)
        except Exception as error:
            reason = f"dependency resolution failed: {error}"
            if (
                activation.mode == "explicit"
                or activation.skill.dependency_failure_mode == "fail"
            ):
                raise ValidationError(
                    f'Skill "{activation.skill.name}" could not be activated: {reason}'
                ) from error
            skipped.append(
                _SkillSkip(
                    skill_name=activation.skill.name,
                    reason=reason,
                    mode=activation.mode,
                    path=activation.skill.path,
                )
            )
            continue
        conflicts = [
            name
            for name, definition in candidate_tools.items()
            if name in resolved_tools and resolved_tools[name] != definition
        ]
        if conflicts:
            reason = f"conflicting tools: {', '.join(sorted(conflicts))}"
            if activation.mode == "explicit":
                raise ValidationError(
                    f'Explicit skill "{activation.skill.name}" could not be activated due to {reason}.'
                )
            skipped.append(
                _SkillSkip(
                    skill_name=activation.skill.name,
                    reason=reason,
                    mode=activation.mode,
                    path=activation.skill.path,
                )
            )
            continue
        resolved_tools.update(candidate_tools)
        activations.append(activation)
    return activations, skipped, resolved_tools


def _persist_active_skills(
    session: AgentSession, active_skills: list[_SkillActivation]
) -> None:
    if (
        not active_skills
        and "sticky_skills" not in session.metadata
        and "active_skills" not in session.metadata
    ):
        return
    sticky_skill_names = [
        item.skill.name for item in active_skills if item.skill.persist_to_session
    ]
    session.metadata = {
        **session.metadata,
        "active_skills": [
            {
                "name": item.skill.name,
                "path": item.skill.path,
                "metadata_path": item.skill.metadata_path,
                "version": item.skill.version,
                "entrypoints": [
                    entrypoint.name for entrypoint in item.skill.entrypoints
                ],
                "activation": item.mode,
                "priority": item.skill.priority,
            }
            for item in active_skills
        ],
        "sticky_skills": sticky_skill_names,
        "skill_versions": {
            item.skill.name: item.skill.version
            for item in active_skills
            if item.skill.version
        },
        "skill_entrypoints": {
            item.skill.name: [entrypoint.name for entrypoint in item.skill.entrypoints]
            for item in active_skills
            if item.skill.entrypoints
        },
    }


async def _emit_skill_events(
    *,
    active_skills: list[_SkillActivation],
    skipped_skills: list[_SkillSkip],
    emit: Callable[[AgentEvent], Awaitable[None]],
) -> None:
    for skipped in skipped_skills:
        await emit(
            AgentSkillSkippedEvent(
                skill_name=skipped.skill_name,
                activation=skipped.mode,
                reason=skipped.reason,
                path=skipped.path,
            )
        )
    for activation in active_skills:
        await emit(
            AgentSkillActivatedEvent(
                skill_name=activation.skill.name,
                activation=activation.mode,
                path=activation.skill.path,
                description=activation.skill.description,
            )
        )
        if activation.skill.version or activation.skill.entrypoints:
            await emit(
                AgentSkillResolvedEvent(
                    skill_name=activation.skill.name,
                    skill_version=activation.skill.version,
                    entrypoints=[item.name for item in activation.skill.entrypoints],
                )
            )
        for dependency in activation.skill.dependencies:
            await emit(
                AgentSkillDependencyCheckEvent(
                    skill_name=activation.skill.name,
                    dependency_type=dependency.type,
                    dependency_value=dependency.value,
                    available=True,
                )
            )


def _extract_skill_artifacts(
    tool_results: list[ToolExecutionResult],
) -> list[SkillArtifact]:
    artifacts: list[SkillArtifact] = []
    for result in tool_results:
        if not isinstance(result.output, dict):
            continue
        payload = result.output.get("artifacts")
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            artifact_path = str(item.get("path") or "").strip()
            if not artifact_path:
                continue
            artifact_metadata = item.get("metadata")
            artifacts.append(
                SkillArtifact(
                    name=str(item.get("name") or Path(artifact_path).name),
                    path=artifact_path,
                    media_type=str(item.get("media_type") or "") or None,
                    role=str(item.get("role") or "primary"),  # type: ignore[arg-type]
                    description=str(item.get("description") or "") or None,
                    metadata=cast(dict[str, Any], artifact_metadata)
                    if isinstance(artifact_metadata, dict)
                    else {},
                )
            )
    return artifacts
