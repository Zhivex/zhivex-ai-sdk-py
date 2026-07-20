from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ValidationError
from .types import ToolDefinition

SkillDependencyType = Literal["mcp", "python", "binary"]
SkillTransport = Literal["stdio", "streamable-http"]
SkillDependencyFailureMode = Literal["skip", "fail"]
SkillArtifactRole = Literal["primary", "preview", "intermediate", "report"]
SkillEntrypointRuntime = Literal["python"]

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
    version: str | None = None
    required: bool = True
    import_name: str | None = None


@dataclass(slots=True)
class SkillArtifact:
    name: str
    path: str
    media_type: str | None = None
    role: SkillArtifactRole = "primary"
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SkillPermissions:
    allow_network: bool = False
    read_paths: list[str] = field(default_factory=list)
    write_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SkillEntrypoint:
    name: str
    description: str | None = None
    runtime: SkillEntrypointRuntime = "python"
    script: str | None = None
    default: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)
    tool_name: str | None = None


@dataclass(slots=True)
class SkillPackageManifest:
    schema_version: int = 1
    name: str = ""
    version: str = ""
    description: str = ""
    entrypoints: list[SkillEntrypoint] = field(default_factory=list)
    dependencies: list[SkillDependency] = field(default_factory=list)
    artifacts: list[SkillArtifact] = field(default_factory=list)
    permissions: SkillPermissions = field(default_factory=SkillPermissions)
    resources: list[str] = field(default_factory=list)
    provider_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InstalledSkill:
    name: str
    version: str
    source: str
    checksum: str
    install_path: str
    content_checksum: str | None = None
    manifest_path: str | None = None
    locked_at: str | None = None


@dataclass(slots=True)
class SkillRegistryIndex:
    registry_url: str | None = None
    skills: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)


@dataclass(slots=True)
class SkillRunResult:
    skill_name: str
    skill_version: str | None = None
    entrypoint: str | None = None
    output: Any = None
    artifacts: list[SkillArtifact] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


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
    version: str | None = None
    entrypoints: list[SkillEntrypoint] = field(default_factory=list)
    artifacts: list[SkillArtifact] = field(default_factory=list)
    permissions: SkillPermissions = field(default_factory=SkillPermissions)
    resources: list[str] = field(default_factory=list)
    source: str | None = None
    checksum: str | None = None
    package_manifest: SkillPackageManifest | None = None
    package_manifest_path: str | None = None
    install_path: str | None = None


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
    version: str | None = None,
    entrypoints: list[SkillEntrypoint] | None = None,
    artifacts: list[SkillArtifact] | None = None,
    permissions: SkillPermissions | None = None,
    resources: list[str] | None = None,
    source: str | None = None,
    checksum: str | None = None,
    package_manifest: SkillPackageManifest | None = None,
    package_manifest_path: str | None = None,
    install_path: str | None = None,
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
        if version is not None:
            loaded.version = version
        if entrypoints is not None:
            loaded.entrypoints = list(entrypoints)
        if artifacts is not None:
            loaded.artifacts = list(artifacts)
        if permissions is not None:
            loaded.permissions = permissions
        if resources is not None:
            loaded.resources = list(resources)
        if source is not None:
            loaded.source = source
        if checksum is not None:
            loaded.checksum = checksum
        if package_manifest is not None:
            loaded.package_manifest = package_manifest
        if package_manifest_path is not None:
            loaded.package_manifest_path = package_manifest_path
        if install_path is not None:
            loaded.install_path = install_path
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
        version=version,
        entrypoints=list(entrypoints or []),
        artifacts=list(artifacts or []),
        permissions=permissions or SkillPermissions(),
        resources=list(resources or []),
        source=source,
        checksum=checksum,
        package_manifest=package_manifest,
        package_manifest_path=package_manifest_path,
        install_path=install_path,
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
    package_manifest_path = skill_path.parent / "skill.yaml"
    metadata = _load_optional_metadata(metadata_path)
    interface = dict(metadata.get("interface") or {})
    policy = dict(metadata.get("policy") or {})
    dependencies = _parse_mcp_skill_dependencies(metadata)
    package_manifest = _load_optional_skill_manifest(package_manifest_path)

    definition = SkillDefinition(
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
    if package_manifest is not None:
        if package_manifest.name and package_manifest.name != definition.name:
            raise ValidationError(
                f'Skill manifest "{package_manifest_path}" name "{package_manifest.name}" must match "{definition.name}".'
            )
        if package_manifest.description and package_manifest.description != definition.description:
            raise ValidationError(
                f'Skill manifest "{package_manifest_path}" description must match the SKILL.md frontmatter description.'
            )
        definition.version = package_manifest.version
        definition.entrypoints = list(package_manifest.entrypoints)
        definition.artifacts = list(package_manifest.artifacts)
        definition.permissions = package_manifest.permissions
        definition.resources = [str((skill_path.parent / resource).resolve()) for resource in package_manifest.resources]
        definition.dependencies.extend(package_manifest.dependencies)
        definition.source = str(skill_path.parent)
        definition.package_manifest = package_manifest
        definition.package_manifest_path = str(package_manifest_path)
    return definition


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
    local_skills: SkillSet = {}
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
            definition = load_skill(skill_file)
            if definition.name in local_skills and local_skills[definition.name].path != definition.path:
                raise ValidationError(
                    f'Skill name collision for "{definition.name}". Rename one of the skills or pass explicit extra_paths.'
                )
            local_skills[definition.name] = definition
    for definition in local_skills.values():
        discovered.register(definition)
    for definition in _discover_installed_skills(start).values():
        discovered._skills[definition.name] = definition
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


def _parse_mcp_skill_dependencies(metadata: dict[str, Any]) -> list[SkillDependency]:
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


def _load_optional_skill_manifest(path: Path) -> SkillPackageManifest | None:
    if not path.exists():
        return None
    parsed = _parse_simple_yaml(path.read_text("utf-8"))
    if not isinstance(parsed, dict):
        raise ValidationError(f'Skill manifest "{path}" must parse to a mapping.')
    return _parse_skill_manifest(parsed, path)


def _parse_skill_manifest(payload: dict[str, Any], path: Path) -> SkillPackageManifest:
    schema_version = payload.get("schema_version", 1)
    if isinstance(schema_version, bool):
        raise ValidationError(f'Skill manifest "{path}" field "schema_version" must be an integer.')
    try:
        normalized_schema_version = int(schema_version)
    except (TypeError, ValueError) as error:
        raise ValidationError(f'Skill manifest "{path}" field "schema_version" must be an integer.') from error
    if normalized_schema_version != 1:
        raise ValidationError(f'Skill manifest "{path}" schema_version must be 1.')
    name = str(payload.get("name") or "").strip()
    version = str(payload.get("version") or "").strip()
    description = str(payload.get("description") or "").strip()
    if not name or not version or not description:
        raise ValidationError(f'Skill manifest "{path}" must define "name", "version", and "description".')

    entrypoints_payload = payload.get("entrypoints") or []
    if not isinstance(entrypoints_payload, list) or not entrypoints_payload:
        raise ValidationError(f'Skill manifest "{path}" must define a non-empty "entrypoints" list.')
    entrypoints = [_parse_skill_entrypoint(item, path) for item in entrypoints_payload]
    if not any(item.default for item in entrypoints):
        entrypoints[0].default = True

    dependencies_payload = payload.get("dependencies") or []
    if not isinstance(dependencies_payload, list):
        raise ValidationError(f'Skill manifest "{path}" field "dependencies" must be a list.')
    dependencies = [_parse_skill_dependency_spec(item, path) for item in dependencies_payload]

    artifacts_payload = payload.get("artifacts") or []
    if not isinstance(artifacts_payload, list):
        raise ValidationError(f'Skill manifest "{path}" field "artifacts" must be a list.')
    artifacts = [_parse_skill_artifact(item, path) for item in artifacts_payload]

    permissions_payload = payload.get("permissions") or {}
    if not isinstance(permissions_payload, dict):
        raise ValidationError(f'Skill manifest "{path}" field "permissions" must be a mapping.')
    resources_payload = payload.get("resources") or []
    if isinstance(resources_payload, str):
        resources_payload = [resources_payload]
    if not isinstance(resources_payload, list):
        raise ValidationError(f'Skill manifest "{path}" field "resources" must be a list.')
    resources = [str(item) for item in resources_payload]
    provider_overrides = dict(payload.get("provider_overrides") or {})
    return SkillPackageManifest(
        schema_version=normalized_schema_version,
        name=name,
        version=version,
        description=description,
        entrypoints=entrypoints,
        dependencies=dependencies,
        artifacts=artifacts,
        permissions=_parse_skill_permissions(permissions_payload),
        resources=resources,
        provider_overrides=provider_overrides,
    )


def _parse_skill_entrypoint(payload: Any, path: Path) -> SkillEntrypoint:
    if not isinstance(payload, dict):
        raise ValidationError(f'Skill manifest "{path}" entrypoints must be mappings.')
    name = str(payload.get("name") or "").strip()
    script = _optional_text(payload.get("script"))
    if not name or not script:
        raise ValidationError(f'Skill manifest "{path}" entrypoints require "name" and "script".')
    runtime = str(payload.get("runtime") or "python").strip().lower()
    if runtime != "python":
        raise ValidationError(f'Skill manifest "{path}" entrypoint "{name}" runtime must be "python".')
    input_schema = dict(payload.get("input_schema") or {})
    if input_schema and not isinstance(input_schema, dict):
        raise ValidationError(f'Skill manifest "{path}" entrypoint "{name}" field "input_schema" must be a mapping.')
    return SkillEntrypoint(
        name=name,
        description=_optional_text(payload.get("description")),
        runtime="python",
        script=script,
        default=bool(payload.get("default", False)),
        input_schema=input_schema,
        tool_name=_optional_text(payload.get("tool_name")),
    )


def _parse_skill_dependency_spec(payload: Any, path: Path) -> SkillDependency:
    if not isinstance(payload, dict):
        raise ValidationError(f'Skill manifest "{path}" dependencies must be mappings.')
    dep_type = str(payload.get("type") or "").strip().lower()
    value = str(payload.get("value") or payload.get("name") or "").strip()
    if dep_type not in {"mcp", "python", "binary"}:
        raise ValidationError(f'Skill manifest "{path}" dependency type "{dep_type}" is not supported.')
    if not value:
        raise ValidationError(f'Skill manifest "{path}" dependencies require "value" or "name".')
    if dep_type == "mcp":
        transport = str(payload.get("transport") or "streamable_http").strip().lower().replace("_", "-")
        if transport not in {"stdio", "streamable-http"}:
            raise ValidationError(f'Skill manifest "{path}" MCP dependency "{value}" has an invalid transport.')
        return SkillDependency(
            type="mcp",
            value=value,
            description=_optional_text(payload.get("description")),
            transport=transport,  # type: ignore[arg-type]
            url=_optional_text(payload.get("url")),
            headers={str(key): str(item) for key, item in dict(payload.get("headers") or {}).items()},
            timeout_ms=payload.get("timeout_ms"),
            command=_optional_text(payload.get("command")),
            args=[str(item) for item in list(payload.get("args") or [])],
            env={str(key): str(item) for key, item in dict(payload.get("env") or {}).items()},
            include=[str(item) for item in list(payload.get("include") or [])],
            exclude=[str(item) for item in list(payload.get("exclude") or [])],
            prefix=_optional_text(payload.get("prefix")),
            version=_optional_text(payload.get("version")),
            required=bool(payload.get("required", True)),
            import_name=_optional_text(payload.get("import_name")),
        )
    return SkillDependency(
        type=dep_type,  # type: ignore[arg-type]
        value=value,
        description=_optional_text(payload.get("description")),
        version=_optional_text(payload.get("version")),
        required=bool(payload.get("required", True)),
        import_name=_optional_text(payload.get("import_name")),
    )


def _parse_skill_artifact(payload: Any, path: Path) -> SkillArtifact:
    if not isinstance(payload, dict):
        raise ValidationError(f'Skill manifest "{path}" artifacts must be mappings.')
    name = str(payload.get("name") or "").strip()
    artifact_path = str(payload.get("path") or "").strip()
    role = str(payload.get("role") or "primary").strip().lower()
    if not name or not artifact_path:
        raise ValidationError(f'Skill manifest "{path}" artifacts require "name" and "path".')
    if role not in {"primary", "preview", "intermediate", "report"}:
        raise ValidationError(f'Skill manifest "{path}" artifact "{name}" has an invalid role "{role}".')
    return SkillArtifact(
        name=name,
        path=artifact_path,
        media_type=_optional_text(payload.get("media_type")),
        role=role,  # type: ignore[arg-type]
        description=_optional_text(payload.get("description")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _parse_skill_permissions(payload: dict[str, Any]) -> SkillPermissions:
    return SkillPermissions(
        allow_network=bool(payload.get("allow_network", False)),
        read_paths=[str(item) for item in list(payload.get("read_paths") or [])],
        write_paths=[str(item) for item in list(payload.get("write_paths") or [])],
    )


def _discover_installed_skills(start: Path) -> SkillSet:
    manifests: list[Path] = []
    for root in _skill_search_roots(start, search_up=True):
        manifest = root / ".agents" / "skills.lock.toml"
        if manifest.exists():
            manifests.append(manifest)
            break
    discovered: SkillSet = {}
    for manifest in manifests:
        installed = _load_installed_skills_from_lockfile(manifest)
        for skill in installed:
            definition = load_skill(Path(skill.install_path))
            definition.source = skill.source
            definition.checksum = skill.checksum
            definition.install_path = skill.install_path
            discovered[definition.name] = definition
    return discovered


def _load_installed_skills_from_lockfile(path: Path) -> list[InstalledSkill]:
    if not path.exists():
        return []
    try:
        import tomllib

        payload = tomllib.loads(path.read_text("utf-8"))
    except Exception as error:
        raise ValidationError(f'Could not read skills lockfile "{path}": {error}') from error
    items = payload.get("skills") or []
    if not isinstance(items, list):
        raise ValidationError(f'Skills lockfile "{path}" field "skills" must be a list.')
    installed: list[InstalledSkill] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValidationError(f'Skills lockfile "{path}" entries must be mappings.')
        installed.append(
            InstalledSkill(
                name=str(item.get("name") or ""),
                version=str(item.get("version") or ""),
                source=str(item.get("source") or ""),
                checksum=str(item.get("checksum") or ""),
                content_checksum=_optional_text(item.get("content_checksum")),
                install_path=str(item.get("install_path") or ""),
                manifest_path=_optional_text(item.get("manifest_path")),
                locked_at=_optional_text(item.get("locked_at")),
            )
        )
    return installed


def load_skill_package(path: str | Path) -> SkillDefinition:
    definition = load_skill(path)
    if definition.package_manifest is None:
        raise ValidationError(f'Skill "{definition.name}" does not define a "skill.yaml" package manifest.')
    return definition


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
            container: list[Any] | dict[str, Any] = [] if stripped.startswith("- ") else {}
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
