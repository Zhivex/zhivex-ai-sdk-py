from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ValidationError
from .types import ToolDefinition

SkillDependencyType = Literal["mcp"]
SkillTransport = Literal["stdio", "streamable-http"]
SkillDependencyFailureMode = Literal["skip", "fail"]

_FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$", re.MULTILINE)


@dataclass(slots=True)
class SkillDependency:
    type: SkillDependencyType
    value: str
    description: str | None = None
    transport: SkillTransport = "streamable-http"
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_ms: int | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    prefix: str | None = None


@dataclass(slots=True)
class SkillDefinition:
    name: str
    description: str
    instructions: str
    path: str | None = None
    metadata_path: str | None = None
    display_name: str | None = None
    short_description: str | None = None
    default_prompt: str | None = None
    allow_implicit_invocation: bool = True
    priority: int = 0
    triggers: list[str] = field(default_factory=list)
    anti_triggers: list[str] = field(default_factory=list)
    allowed_providers: list[str] = field(default_factory=list)
    allowed_models: list[str] = field(default_factory=list)
    persist_to_session: bool = True
    dependency_failure_mode: SkillDependencyFailureMode = "skip"
    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    dependencies: list[SkillDependency] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


SkillSet = dict[str, SkillDefinition]


class SkillRegistry:
    def __init__(self, skills: SkillSet | None = None) -> None:
        self._skills: SkillSet = dict(skills or {})

    def register(self, definition: SkillDefinition) -> SkillDefinition:
        existing = self._skills.get(definition.name)
        if existing is not None and existing.path != definition.path:
            raise ValidationError(
                f'Skill name collision for "{definition.name}". '
                "Rename one of the skills or pass an explicit registry without duplicates."
            )
        self._skills[definition.name] = definition
        return definition

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def items(self) -> list[tuple[str, SkillDefinition]]:
        return list(self._skills.items())

    def merge(self, skills: SkillSet | "SkillRegistry" | None) -> "SkillRegistry":
        merged = SkillRegistry(self._skills)
        if isinstance(skills, SkillRegistry):
            for definition in skills._skills.values():
                merged.register(definition)
            return merged
        for definition in dict(skills or {}).values():
            merged.register(definition)
        return merged


def skill(
    definition: SkillDefinition | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    instructions: str | None = None,
    path: str | None = None,
    display_name: str | None = None,
    short_description: str | None = None,
    default_prompt: str | None = None,
    allow_implicit_invocation: bool | None = None,
    priority: int | None = None,
    triggers: list[str] | None = None,
    anti_triggers: list[str] | None = None,
    allowed_providers: list[str] | None = None,
    allowed_models: list[str] | None = None,
    persist_to_session: bool | None = None,
    dependency_failure_mode: SkillDependencyFailureMode | None = None,
    tools: dict[str, ToolDefinition] | None = None,
    dependencies: list[SkillDependency] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SkillDefinition:
    if definition is not None:
        return definition
    if path is not None:
        loaded = load_skill(path)
        if name is not None:
            loaded.name = name
        if description is not None:
            loaded.description = description
        if instructions is not None:
            loaded.instructions = instructions
        if display_name is not None:
            loaded.display_name = display_name
        if short_description is not None:
            loaded.short_description = short_description
        if default_prompt is not None:
            loaded.default_prompt = default_prompt
        if allow_implicit_invocation is not None:
            loaded.allow_implicit_invocation = allow_implicit_invocation
        if priority is not None:
            loaded.priority = int(priority)
        if triggers is not None:
            loaded.triggers = _normalize_text_list(triggers)
        if anti_triggers is not None:
            loaded.anti_triggers = _normalize_text_list(anti_triggers)
        if allowed_providers is not None:
            loaded.allowed_providers = _normalize_text_list(allowed_providers)
        if allowed_models is not None:
            loaded.allowed_models = _normalize_text_list(allowed_models)
        if persist_to_session is not None:
            loaded.persist_to_session = persist_to_session
        if dependency_failure_mode is not None:
            loaded.dependency_failure_mode = dependency_failure_mode
        if tools:
            loaded.tools.update(tools)
        if dependencies:
            loaded.dependencies.extend(dependencies)
        if metadata:
            loaded.metadata.update(metadata)
        return loaded
    if not name or not description or not instructions:
        raise ValueError('Pass either an existing SkillDefinition, a "path", or the trio "name", "description", and "instructions".')
    return SkillDefinition(
        name=name,
        description=description,
        instructions=instructions,
        display_name=display_name,
        short_description=short_description,
        default_prompt=default_prompt,
        allow_implicit_invocation=True if allow_implicit_invocation is None else allow_implicit_invocation,
        priority=int(priority or 0),
        triggers=_normalize_text_list(triggers or []),
        anti_triggers=_normalize_text_list(anti_triggers or []),
        allowed_providers=_normalize_text_list(allowed_providers or []),
        allowed_models=_normalize_text_list(allowed_models or []),
        persist_to_session=True if persist_to_session is None else persist_to_session,
        dependency_failure_mode=dependency_failure_mode or "skip",
        tools=dict(tools or {}),
        dependencies=list(dependencies or []),
        metadata=dict(metadata or {}),
    )


def load_skill(path: str | Path) -> SkillDefinition:
    skill_path = Path(path).expanduser().resolve()
    if skill_path.is_dir():
        skill_path = skill_path / "SKILL.md"
    if skill_path.name != "SKILL.md":
        raise ValidationError('Skill paths must point to a "SKILL.md" file or its parent directory.')
    if not skill_path.exists():
        raise ValidationError(f'Skill file "{skill_path}" does not exist.')

    body = skill_path.read_text("utf-8")
    frontmatter, instructions = _split_frontmatter(body, skill_path)
    payload = _parse_simple_yaml(frontmatter)
    name = str(payload.get("name") or "").strip()
    description = str(payload.get("description") or "").strip()
    if not name or not description:
        raise ValidationError(f'Skill "{skill_path}" must define both "name" and "description" in frontmatter.')

    metadata_path = skill_path.parent / "agents" / "openai.yaml"
    metadata = _load_optional_metadata(metadata_path)
    interface = dict(metadata.get("interface") or {})
    policy = dict(metadata.get("policy") or {})
    dependencies = _parse_skill_dependencies(metadata)

    return SkillDefinition(
        name=name,
        description=description,
        instructions=instructions.strip(),
        path=str(skill_path),
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
        display_name=_optional_text(interface.get("display_name")),
        short_description=_optional_text(interface.get("short_description")),
        default_prompt=_optional_text(interface.get("default_prompt")),
        allow_implicit_invocation=bool(policy.get("allow_implicit_invocation", True)),
        priority=_parse_skill_priority(policy.get("priority")),
        triggers=_normalize_text_list(policy.get("triggers") or []),
        anti_triggers=_normalize_text_list(policy.get("anti_triggers") or []),
        allowed_providers=_normalize_text_list(policy.get("allowed_providers") or policy.get("providers") or []),
        allowed_models=_normalize_text_list(policy.get("allowed_models") or policy.get("models") or []),
        persist_to_session=bool(policy.get("persist_to_session", True)),
        dependency_failure_mode=_parse_dependency_failure_mode(policy.get("dependency_failure_mode")),
        dependencies=dependencies,
        metadata=metadata,
    )


def discover_skills(
    *,
    cwd: str | Path | None = None,
    search_up: bool = True,
    extra_paths: list[str | Path] | None = None,
) -> SkillSet:
    start = Path(cwd or Path.cwd()).expanduser().resolve()
    search_roots = _skill_search_roots(start, search_up=search_up)
    for extra in extra_paths or []:
        search_roots.append(Path(extra).expanduser().resolve())

    discovered = SkillRegistry()
    seen_dirs: set[Path] = set()
    for root in search_roots:
        skill_root = root if root.name == "skills" else root / ".agents" / "skills"
        if skill_root in seen_dirs or not skill_root.exists():
            continue
        seen_dirs.add(skill_root)
        for child in sorted(skill_root.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.exists():
                continue
            discovered.register(load_skill(skill_file))
    return dict(discovered.items())


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_text_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        text = _optional_text(value)
        if text is None or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _parse_skill_priority(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValidationError('Skill metadata "policy.priority" must be an integer.')
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValidationError('Skill metadata "policy.priority" must be an integer.') from error


def _parse_dependency_failure_mode(value: Any) -> SkillDependencyFailureMode:
    if value is None:
        return "skip"
    normalized = str(value).strip().lower()
    if normalized not in {"skip", "fail"}:
        raise ValidationError('Skill metadata "policy.dependency_failure_mode" must be "skip" or "fail".')
    return normalized  # type: ignore[return-value]


def _split_frontmatter(body: str, path: Path) -> tuple[str, str]:
    if not body.startswith("---"):
        raise ValidationError(f'Skill "{path}" must begin with YAML frontmatter.')
    matches = list(_FRONTMATTER_BOUNDARY.finditer(body))
    if len(matches) < 2:
        raise ValidationError(f'Skill "{path}" has an unterminated YAML frontmatter block.')
    start, end = matches[0], matches[1]
    return body[start.end(): end.start()].strip(), body[end.end():].strip()


def _skill_search_roots(start: Path, *, search_up: bool) -> list[Path]:
    roots = [start]
    if not search_up:
        return roots
    repo_root = _find_repo_root(start)
    current = start.parent
    while repo_root is not None and current != current.parent:
        roots.append(current)
        if current == repo_root:
            break
        current = current.parent
    return roots


def _find_repo_root(start: Path) -> Path | None:
    current = start
    candidate: Path | None = None
    while True:
        if (current / ".git").exists():
            candidate = current
        if current == current.parent:
            return candidate
        current = current.parent


def _load_optional_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    parsed = _parse_simple_yaml(path.read_text("utf-8"))
    if not isinstance(parsed, dict):
        raise ValidationError(f'Skill metadata "{path}" must parse to a mapping.')
    return parsed


def _parse_skill_dependencies(metadata: dict[str, Any]) -> list[SkillDependency]:
    dependencies = metadata.get("dependencies")
    if not isinstance(dependencies, dict):
        return []
    tools = dependencies.get("tools")
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValidationError('Skill metadata "dependencies.tools" must be a list.')
    parsed: list[SkillDependency] = []
    for item in tools:
        if not isinstance(item, dict):
            raise ValidationError("Each skill dependency entry must be a mapping.")
        dep_type = str(item.get("type") or "").strip().lower()
        if dep_type != "mcp":
            raise ValidationError(f'Unsupported skill dependency type "{dep_type}". Only "mcp" is supported.')
        transport = str(item.get("transport") or "streamable_http").strip().lower().replace("_", "-")
        if transport not in {"stdio", "streamable-http"}:
            raise ValidationError(f'Unsupported MCP transport "{transport}" in skill metadata.')
        parsed.append(
            SkillDependency(
                type="mcp",
                value=str(item.get("value") or item.get("name") or "mcp"),
                description=_optional_text(item.get("description")),
                transport=transport,  # type: ignore[arg-type]
                url=_optional_text(item.get("url")),
                headers={str(key): str(value) for key, value in dict(item.get("headers") or {}).items()},
                timeout_ms=item.get("timeout_ms"),
                command=_optional_text(item.get("command")),
                args=[str(arg) for arg in list(item.get("args") or [])],
                env={str(key): str(value) for key, value in dict(item.get("env") or {}).items()},
                include=[str(name) for name in list(item.get("include") or [])],
                exclude=[str(name) for name in list(item.get("exclude") or [])],
                prefix=_optional_text(item.get("prefix")),
            )
        )
    return parsed


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending: tuple[int, dict[str, Any], str] | None = None
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        if pending is not None and indent > pending[0]:
            container = [] if stripped.startswith("- ") else {}
            pending[1][pending[2]] = container
            stack.append((pending[0], container))
            pending = None
        elif pending is not None:
            pending[1][pending[2]] = {}
            stack.append((pending[0], pending[1][pending[2]]))
            pending = None
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()

        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValidationError("Invalid YAML structure: list item found without a list parent.")
            item_value = stripped[2:].strip()
            if not item_value:
                child: Any = {}
                parent.append(child)
                stack.append((indent, child))
                continue
            if ":" in item_value:
                key, raw_value = item_value.split(":", 1)
                child = {key.strip(): _parse_scalar(raw_value.strip()) if raw_value.strip() else {}}
                parent.append(child)
                stack.append((indent, child))
                if not raw_value.strip():
                    pending = (indent, child, key.strip())
                continue
            parent.append(_parse_scalar(item_value))
            continue

        if not isinstance(parent, dict):
            raise ValidationError("Invalid YAML structure: mapping entry found without a mapping parent.")
        if ":" not in stripped:
            raise ValidationError(f'Unsupported YAML line "{stripped}".')
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            parent[key] = _parse_scalar(raw_value)
            continue
        pending = (indent, parent, key)

    if pending is not None:
        pending[1][pending[2]] = {}
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    return value
