from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import socket
import tarfile
from tempfile import TemporaryDirectory
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from .errors import ValidationError
from .skills import (
    InstalledSkill,
    SkillArtifact,
    SkillDefinition,
    SkillEntrypoint,
    SkillRegistryIndex,
    SkillRunResult,
    load_skill,
    load_skill_package,
)
from .types import ToolDefinition


_MAX_REGISTRY_INDEX_BYTES = 2 * 1024 * 1024
_MAX_SKILL_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_SKILL_ARCHIVE_MEMBERS = 1_024
_MAX_SKILL_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_SKILL_EXTRACTED_BYTES = 256 * 1024 * 1024
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PACKAGE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def validate_skill(path: str | Path) -> SkillDefinition:
    return load_skill(path)


def list_installed_skills(*, project_root: str | Path | None = None) -> list[InstalledSkill]:
    root = _project_root(project_root)
    return _read_lockfile(_skills_lock_path(root))


def install_skill(
    source: str | Path,
    *,
    project_root: str | Path | None = None,
    lock: bool = True,
    registry_url: str | None = None,
    trust_remote_code: bool = False,
) -> InstalledSkill:
    root = _project_root(project_root)
    source_path = Path(source).expanduser()
    if source_path.exists():
        definition = load_skill(source_path)
        _validate_skill_definition(definition)
        checksum = _hash_skill_directory(_skill_dir_from_input(source_path))
        installed = _materialize_installed_skill(
            definition,
            checksum=checksum,
            source=str(source_path.resolve()),
            project_root=root,
        )
        manifest_registry = registry_url
    else:
        name, version = _parse_registry_ref(str(source))
        resolved_registry_url = registry_url or _read_project_registry(root) or os.environ.get("ZHIVEX_SKILLS_REGISTRY")
        if not resolved_registry_url:
            raise ValidationError(
                'Registry installs require "registry_url", a project skills.toml registry, or ZHIVEX_SKILLS_REGISTRY.'
            )
        if not trust_remote_code:
            raise ValidationError(
                "Registry skill packages contain executable Python code. Review the package and checksum, then pass "
                "trust_remote_code=True only when you trust that registry and package."
            )
        installed = _install_from_registry(
            name=name,
            version=version,
            registry_url=resolved_registry_url,
            project_root=root,
        )
        manifest_registry = resolved_registry_url
    _upsert_project_manifest(root, installed=installed, registry_url=manifest_registry)
    if lock:
        _upsert_lockfile(root, installed)
    return installed


async def run_skill(
    name: str,
    *,
    entrypoint: str | None = None,
    input: Any = None,
    project_root: str | Path | None = None,
) -> SkillRunResult:
    root = _project_root(project_root)
    definition = _resolve_skill_for_run(name, root)
    chosen_entrypoint = _resolve_entrypoint(definition, entrypoint)
    payload = input if input is not None else {}
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValidationError('run_skill(...) currently requires "input" to be a mapping.')
    _validate_entrypoint_input(chosen_entrypoint, payload)
    _validate_skill_dependencies(definition)
    _validate_path_permissions(definition, payload, project_root=root)
    return await _execute_python_entrypoint(definition, chosen_entrypoint, payload, project_root=root)


def publish_skill(path: str | Path, *, registry_dir: str | Path) -> SkillRegistryIndex:
    definition = load_skill_package(path)
    _validate_skill_definition(definition)
    _validate_package_component(definition.name, field="name")
    _validate_package_component(str(definition.version or ""), field="version")
    registry_root = Path(registry_dir).expanduser().resolve()
    registry_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir = registry_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    skill_root = _skill_dir_from_input(Path(path))
    _validate_skill_tree(skill_root)
    tarball_name = f"{definition.name}-{definition.version}.tar.gz"
    tarball_path = artifacts_dir / tarball_name
    with tarfile.open(tarball_path, "w:gz") as archive:
        archive.add(skill_root, arcname=definition.name, filter=_skill_tar_filter)
    checksum = _sha256_file(tarball_path)
    index_path = registry_root / "index.json"
    payload: dict[str, Any]
    if index_path.exists():
        payload = json.loads(index_path.read_text("utf-8"))
    else:
        payload = {"skills": {}}
    skills = payload.setdefault("skills", {})
    versions = skills.setdefault(definition.name, {})
    versions[str(definition.version)] = {
        "artifact_url": f"artifacts/{tarball_name}",
        "checksum": checksum,
        "metadata_url": f"artifacts/{tarball_name}",
        "description": definition.description,
    }
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    return SkillRegistryIndex(registry_url=str(index_path), skills=skills)


def build_skill_entrypoint_tools(
    skill: SkillDefinition, *, project_root: str | Path | None = None
) -> dict[str, ToolDefinition]:
    tools: dict[str, ToolDefinition] = {}
    root = _project_root(project_root)
    permissions = _skill_tool_permissions(skill)
    for entrypoint in skill.entrypoints:
        tool_name = entrypoint.tool_name or f"{skill.name}_{entrypoint.name}"

        async def execute(
            input: Any,
            *,
            _skill_name: str = skill.name
            if skill.install_path
            else ((str(Path(skill.path).resolve().parent) if skill.path else None) or skill.source or skill.name),
            _entrypoint: str = entrypoint.name,
            _root: Path = root,
        ) -> dict[str, Any]:
            result = await run_skill(_skill_name, entrypoint=_entrypoint, input=input, project_root=_root)
            return {
                "skill_name": result.skill_name,
                "skill_version": result.skill_version,
                "entrypoint": result.entrypoint,
                "output": result.output,
                "artifacts": [_artifact_to_payload(item) for item in result.artifacts],
                "logs": list(result.logs),
            }

        tools[tool_name] = ToolDefinition(
            name=tool_name,
            description=entrypoint.description or f'Run skill "{skill.name}" entrypoint "{entrypoint.name}".',
            schema=entrypoint.input_schema or {"type": "object"},
            execute=execute,
            source="local",
            permissions=list(permissions),
            requires_approval=True,
            metadata={
                "skill_name": skill.name,
                "skill_version": skill.version,
                "skill_entrypoint": entrypoint.name,
                "skill_package": bool(skill.package_manifest is not None),
                "skill_permissions": {
                    "allow_network": skill.permissions.allow_network,
                    "read_paths": list(skill.permissions.read_paths),
                    "write_paths": list(skill.permissions.write_paths),
                },
            },
        )
    return tools


def _skill_tool_permissions(skill: SkillDefinition) -> list[str]:
    permissions: list[str] = ["code-execution"]
    if skill.permissions.read_paths or skill.permissions.write_paths:
        permissions.append("filesystem")
    if skill.permissions.write_paths:
        permissions.append("write")
    if skill.permissions.allow_network:
        permissions.append("network")
    return permissions


def _project_root(project_root: str | Path | None) -> Path:
    return Path(project_root or Path.cwd()).expanduser().resolve()


def _skills_dir(root: Path) -> Path:
    return root / ".agents"


def _skills_manifest_path(root: Path) -> Path:
    return _skills_dir(root) / "skills.toml"


def _skills_lock_path(root: Path) -> Path:
    return _skills_dir(root) / "skills.lock.toml"


def _installed_skills_cache_dir(root: Path) -> Path:
    return _skills_dir(root) / "installed"


def _read_project_registry(root: Path) -> str | None:
    manifest_path = _skills_manifest_path(root)
    if not manifest_path.exists():
        return None
    try:
        import tomllib

        payload = tomllib.loads(manifest_path.read_text("utf-8"))
    except Exception as error:
        raise ValidationError(f'Could not read skills manifest "{manifest_path}": {error}') from error
    registry_value = payload.get("registry")
    if registry_value is None:
        return None
    return str(registry_value)


def _upsert_project_manifest(root: Path, *, installed: InstalledSkill, registry_url: str | None) -> None:
    manifest_path = _skills_manifest_path(root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    skills = _read_manifest_skills(manifest_path)
    retained = [item for item in skills if item.get("name") != installed.name]
    retained.append(
        {
            "name": installed.name,
            "version": installed.version,
            "source": installed.source,
        }
    )
    lines = []
    if registry_url:
        lines.append(f"registry = {_toml_string(registry_url)}")
        lines.append("")
    for item in retained:
        lines.append("[[skills]]")
        lines.append(f"name = {_toml_string(item['name'])}")
        lines.append(f"version = {_toml_string(item['version'])}")
        lines.append(f"source = {_toml_string(item['source'])}")
        lines.append("")
    manifest_path.write_text("\n".join(lines).rstrip() + "\n", "utf-8")


def _read_manifest_skills(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        import tomllib

        payload = tomllib.loads(path.read_text("utf-8"))
    except Exception as error:
        raise ValidationError(f'Could not read skills manifest "{path}": {error}') from error
    skills = payload.get("skills") or []
    if not isinstance(skills, list):
        raise ValidationError(f'Skills manifest "{path}" field "skills" must be a list.')
    normalized: list[dict[str, str]] = []
    for item in skills:
        if not isinstance(item, dict):
            raise ValidationError(f'Skills manifest "{path}" entries must be mappings.')
        normalized.append(
            {
                "name": str(item.get("name") or ""),
                "version": str(item.get("version") or ""),
                "source": str(item.get("source") or ""),
            }
        )
    return normalized


def _read_lockfile(path: Path) -> list[InstalledSkill]:
    if not path.exists():
        return []
    try:
        import tomllib

        payload = tomllib.loads(path.read_text("utf-8"))
    except Exception as error:
        raise ValidationError(f'Could not read skills lockfile "{path}": {error}') from error
    skills = payload.get("skills") or []
    if not isinstance(skills, list):
        raise ValidationError(f'Skills lockfile "{path}" field "skills" must be a list.')
    installed: list[InstalledSkill] = []
    for item in skills:
        if not isinstance(item, dict):
            raise ValidationError(f'Skills lockfile "{path}" entries must be mappings.')
        installed.append(
            InstalledSkill(
                name=str(item.get("name") or ""),
                version=str(item.get("version") or ""),
                source=str(item.get("source") or ""),
                checksum=str(item.get("checksum") or ""),
                install_path=str(item.get("install_path") or ""),
                content_checksum=str(item.get("content_checksum") or "") or None,
                manifest_path=str(item.get("manifest_path") or "") or None,
                locked_at=str(item.get("locked_at") or "") or None,
            )
        )
    return installed


def _upsert_lockfile(root: Path, installed: InstalledSkill) -> None:
    lock_path = _skills_lock_path(root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    items = [item for item in _read_lockfile(lock_path) if item.name != installed.name]
    items.append(installed)
    lines: list[str] = []
    for item in items:
        lines.append("[[skills]]")
        lines.append(f"name = {_toml_string(item.name)}")
        lines.append(f"version = {_toml_string(item.version)}")
        lines.append(f"source = {_toml_string(item.source)}")
        lines.append(f"checksum = {_toml_string(item.checksum)}")
        if item.content_checksum:
            lines.append(f"content_checksum = {_toml_string(item.content_checksum)}")
        lines.append(f"install_path = {_toml_string(item.install_path)}")
        if item.manifest_path:
            lines.append(f"manifest_path = {_toml_string(item.manifest_path)}")
        if item.locked_at:
            lines.append(f"locked_at = {_toml_string(item.locked_at)}")
        lines.append("")
    lock_path.write_text("\n".join(lines).rstrip() + "\n", "utf-8")


def _materialize_installed_skill(
    definition: SkillDefinition,
    *,
    checksum: str,
    source: str,
    project_root: Path,
) -> InstalledSkill:
    skill_root = _skill_dir_from_input(Path(definition.source or definition.path or source))
    _validate_skill_definition(definition)
    _validate_skill_tree(skill_root)
    safe_name = _validate_package_component(definition.name, field="name")
    safe_version = _validate_package_component(str(definition.version or "0"), field="version")
    cache_root = _installed_skills_cache_dir(project_root).resolve()
    cache_dir = (cache_root / safe_name / safe_version).resolve()
    if not _is_relative_to(cache_dir, cache_root):
        raise ValidationError("Resolved skill install path escaped the project skill cache.")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    shutil.copytree(
        skill_root,
        cache_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    installed_definition = load_skill(cache_dir)
    content_checksum = _hash_skill_directory(cache_dir)
    return InstalledSkill(
        name=installed_definition.name,
        version=str(installed_definition.version or "0"),
        source=source,
        checksum=checksum,
        install_path=str(cache_dir),
        content_checksum=content_checksum,
        manifest_path=installed_definition.package_manifest_path,
        locked_at=str(int(time.time())),
    )


def _install_from_registry(*, name: str, version: str, registry_url: str, project_root: Path) -> InstalledSkill:
    registry_url = _validate_remote_url(registry_url, purpose="registry")
    try:
        payload = json.loads(_download_url(registry_url, timeout=30.0, max_bytes=_MAX_REGISTRY_INDEX_BYTES))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f'Registry "{registry_url}" did not return valid UTF-8 JSON.') from error
    if not isinstance(payload, dict):
        raise ValidationError(f'Registry "{registry_url}" index must be a JSON object.')
    skills = dict(payload.get("skills") or {})
    versions = dict(skills.get(name) or {})
    metadata = dict(versions.get(version) or {})
    if not metadata:
        raise ValidationError(f'Skill "{name}@{version}" was not found in registry "{registry_url}".')
    artifact_url = str(metadata.get("artifact_url") or "")
    checksum = str(metadata.get("checksum") or "")
    if not artifact_url or not checksum:
        raise ValidationError(f'Registry entry for "{name}@{version}" is missing "artifact_url" or "checksum".')
    if not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        raise ValidationError(f'Registry entry for "{name}@{version}" has an invalid SHA-256 checksum.')
    checksum = checksum.lower()
    resolved_artifact_url = _resolve_registry_artifact_url(registry_url, artifact_url)
    with TemporaryDirectory() as tmp:
        tarball_path = Path(tmp) / f"{name}-{version}.tar.gz"
        tarball_path.write_bytes(_download_url(resolved_artifact_url, timeout=60.0, max_bytes=_MAX_SKILL_ARCHIVE_BYTES))
        actual_checksum = _sha256_file(tarball_path)
        if actual_checksum != checksum:
            raise ValidationError(
                f'Registry skill "{name}@{version}" checksum mismatch: expected {checksum}, got {actual_checksum}.'
            )
        extract_dir = Path(tmp) / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball_path, "r:gz") as archive:
            _safe_extract_tar(archive, extract_dir)
        skill_dir = extract_dir / name
        if not skill_dir.exists():
            nested_dirs = [item for item in extract_dir.iterdir() if item.is_dir()]
            if len(nested_dirs) != 1:
                raise ValidationError(
                    f'Registry artifact for "{name}@{version}" did not unpack to a single skill directory.'
                )
            skill_dir = nested_dirs[0]
        definition = load_skill_package(skill_dir)
        if definition.name != name or str(definition.version) != version:
            raise ValidationError(
                f'Registry artifact identity mismatch: requested "{name}@{version}", '
                f'got "{definition.name}@{definition.version}".'
            )
        return _materialize_installed_skill(
            definition,
            checksum=checksum,
            source=f"{registry_url}#{name}@{version}",
            project_root=project_root,
        )


def _parse_registry_ref(source: str) -> tuple[str, str]:
    if "@" not in source:
        raise ValidationError('Registry installs require "name@version".')
    name, version = source.rsplit("@", 1)
    if not name or not version:
        raise ValidationError('Registry installs require "name@version".')
    return _validate_package_component(name, field="name"), _validate_package_component(version, field="version")


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    members: list[tarfile.TarInfo] = []
    total_size = 0
    for member_count, member in enumerate(archive, start=1):
        if member_count > _MAX_SKILL_ARCHIVE_MEMBERS:
            raise ValidationError(
                f"Registry artifact contains more than {_MAX_SKILL_ARCHIVE_MEMBERS} filesystem entries."
            )
        target = (destination / member.name).resolve()
        if not _is_relative_to(target, destination):
            raise ValidationError(f'Registry artifact contains unsafe path "{member.name}".')
        if member.islnk() or member.issym():
            raise ValidationError(f'Registry artifact contains unsupported link "{member.name}".')
        if member.isdev() or member.isfifo():
            raise ValidationError(f'Registry artifact contains unsupported special file "{member.name}".')
        if member.size > _MAX_SKILL_MEMBER_BYTES:
            raise ValidationError(f'Registry artifact member "{member.name}" exceeds the per-file size limit.')
        total_size += member.size
        if total_size > _MAX_SKILL_EXTRACTED_BYTES:
            raise ValidationError("Registry artifact exceeds the total extracted-size limit.")
        members.append(member)
    if "filter" in inspect.signature(archive.extractall).parameters:
        archive.extractall(destination, members=members, filter="data")
    else:
        archive.extractall(destination, members=members)


def _resolve_skill_for_run(name: str, root: Path) -> SkillDefinition:
    candidate_path = Path(name).expanduser()
    if candidate_path.exists():
        return load_skill(candidate_path)
    for installed in _read_lockfile(_skills_lock_path(root)):
        if installed.name == name:
            install_root = _installed_skills_cache_dir(root).resolve()
            install_path = Path(installed.install_path).expanduser().resolve()
            if not _is_relative_to(install_path, install_root):
                raise ValidationError(
                    f'Installed skill "{name}" path "{install_path}" is outside the project skill cache.'
                )
            _validate_skill_tree(install_path)
            actual_checksum = _hash_skill_directory(install_path)
            if installed.content_checksum:
                expected_checksum = installed.content_checksum
            elif _is_remote_skill_source(installed.source):
                raise ValidationError(
                    f'Installed registry skill "{name}" uses a legacy lock entry without a content checksum. '
                    "Review and reinstall it with explicit remote-code trust before running."
                )
            else:
                expected_checksum = installed.checksum
            if not expected_checksum or actual_checksum != expected_checksum:
                raise ValidationError(
                    f'Installed skill "{name}" failed its lockfile checksum verification. Reinstall it before running.'
                )
            definition = load_skill(install_path)
            definition.source = installed.source
            definition.checksum = installed.checksum
            definition.install_path = installed.install_path
            return definition
    builtin = _builtin_skill_path(name)
    if builtin is not None:
        return load_skill(builtin)
    raise ValidationError(f'Unknown skill "{name}". Install it first or pass a path to the skill directory.')


def _builtin_skill_path(name: str) -> Path | None:
    package_candidate = Path(__file__).resolve().parent / "official_skills" / name
    if package_candidate.exists():
        return package_candidate
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / ".agents" / "skills" / name
        if candidate.exists():
            return candidate
    return None


def _resolve_entrypoint(definition: SkillDefinition, entrypoint: str | None) -> SkillEntrypoint:
    if not definition.entrypoints:
        raise ValidationError(f'Skill "{definition.name}" does not define package entrypoints.')
    if entrypoint is None:
        for item in definition.entrypoints:
            if item.default:
                return item
        return definition.entrypoints[0]
    for item in definition.entrypoints:
        if item.name == entrypoint:
            return item
    raise ValidationError(f'Skill "{definition.name}" does not define entrypoint "{entrypoint}".')


def _validate_entrypoint_input(entrypoint: SkillEntrypoint, payload: dict[str, Any]) -> None:
    schema = dict(entrypoint.input_schema or {})
    if not schema:
        return
    if str(schema.get("type") or "object") != "object":
        raise ValidationError(f'Skill entrypoint "{entrypoint.name}" currently only supports object input schemas.')
    required = [str(item) for item in list(schema.get("required") or [])]
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValidationError(f'Skill entrypoint "{entrypoint.name}" is missing required fields: {", ".join(missing)}.')
    properties = dict(schema.get("properties") or {})
    for key, value in payload.items():
        property_schema = properties.get(key)
        if not isinstance(property_schema, dict):
            continue
        expected_type = str(property_schema.get("type") or "")
        if expected_type == "string" and not isinstance(value, str):
            raise ValidationError(f'Skill entrypoint "{entrypoint.name}" field "{key}" must be a string.')
        if expected_type == "array" and not isinstance(value, list):
            raise ValidationError(f'Skill entrypoint "{entrypoint.name}" field "{key}" must be an array.')
        if expected_type == "object" and not isinstance(value, dict):
            raise ValidationError(f'Skill entrypoint "{entrypoint.name}" field "{key}" must be an object.')
        if expected_type == "boolean" and not isinstance(value, bool):
            raise ValidationError(f'Skill entrypoint "{entrypoint.name}" field "{key}" must be a boolean.')


def _validate_skill_dependencies(definition: SkillDefinition) -> None:
    for dependency in definition.dependencies:
        if dependency.type == "mcp":
            continue
        if dependency.type == "python":
            import_name = dependency.import_name or dependency.value.replace("-", "_")
            if importlib.util.find_spec(import_name) is None:
                raise RuntimeError(
                    f'Skill "{definition.name}" requires Python package "{dependency.value}". '
                    f"Install it first, for example with `pip install {dependency.value}`."
                )
            continue
        if dependency.type == "binary":
            if shutil.which(dependency.value) is None:
                raise RuntimeError(f'Skill "{definition.name}" requires binary "{dependency.value}" to be installed.')


def _validate_path_permissions(definition: SkillDefinition, payload: dict[str, Any], *, project_root: Path) -> None:
    skill_root = _definition_skill_root(definition)
    allowed_read_roots = {skill_root}
    for item in definition.permissions.read_paths:
        allowed_read_roots.add(_resolve_permission_root(item, project_root=project_root, definition=definition))
    for item in definition.resources:
        resource = Path(item).expanduser().resolve()
        if not _is_relative_to(resource, skill_root):
            raise ValidationError(f'Skill "{definition.name}" resource "{resource}" is outside the skill root.')
        allowed_read_roots.add(resource)
    allowed_write_roots: set[Path] = set()
    for item in definition.permissions.write_paths:
        allowed_write_roots.add(_resolve_permission_root(item, project_root=project_root, definition=definition))
    for key, value in payload.items():
        if not isinstance(value, str) or not key.endswith("_path"):
            continue
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = (project_root / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if key.startswith("output") or key.endswith("output_path"):
            if not any(_is_relative_to(resolved, root) for root in allowed_write_roots):
                raise ValidationError(
                    f'Skill "{definition.name}" output path "{resolved}" is outside the allowed write roots.'
                )
        elif not any(_is_relative_to(resolved, root) for root in allowed_read_roots):
            raise ValidationError(
                f'Skill "{definition.name}" input path "{resolved}" is outside the allowed read roots.'
            )


async def _execute_python_entrypoint(
    definition: SkillDefinition,
    entrypoint: SkillEntrypoint,
    payload: dict[str, Any],
    *,
    project_root: Path,
) -> SkillRunResult:
    skill_root = _definition_skill_root(definition)
    script_value = Path(str(entrypoint.script or ""))
    if script_value.is_absolute():
        raise ValidationError(f'Skill "{definition.name}" entrypoint script must be relative to the skill root.')
    script_path = (skill_root / script_value).resolve()
    if not _is_relative_to(script_path, skill_root):
        raise ValidationError(f'Skill "{definition.name}" entrypoint script escapes the skill root.')
    if not script_path.is_file():
        raise ValidationError(
            f'Skill "{definition.name}" entrypoint "{entrypoint.name}" script "{script_path}" does not exist.'
        )
    context = {
        "skill_name": definition.name,
        "skill_version": definition.version,
        "entrypoint": entrypoint.name,
        "skill_root": str(skill_root),
        "project_root": str(project_root),
    }
    with _network_policy(definition.permissions.allow_network):
        runner = _load_python_entrypoint(script_path)
        result = runner(payload, context)
        if inspect.isawaitable(result):
            result = await result
    return _normalize_skill_run_result(definition, entrypoint, result, project_root=project_root)


def _load_python_entrypoint(path: Path):
    module_name = f"zhivex_skill_{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValidationError(f'Could not load skill entrypoint script "{path}".')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = getattr(module, "run", None) or getattr(module, "main", None)
    if runner is None or not callable(runner):
        raise ValidationError(f'Skill entrypoint script "{path}" must export a callable "run" or "main".')
    return runner


def _normalize_skill_run_result(
    definition: SkillDefinition,
    entrypoint: SkillEntrypoint,
    result: Any,
    *,
    project_root: Path,
) -> SkillRunResult:
    if isinstance(result, SkillRunResult):
        artifacts = [
            _artifact_from_payload(item, project_root=project_root, definition=definition) for item in result.artifacts
        ]
        result.artifacts = artifacts
        return result
    if isinstance(result, dict):
        artifacts = [
            _artifact_from_payload(item, project_root=project_root, definition=definition)
            for item in list(result.get("artifacts") or [])
        ]
        return SkillRunResult(
            skill_name=definition.name,
            skill_version=definition.version,
            entrypoint=entrypoint.name,
            output=result.get("output"),
            artifacts=artifacts,
            logs=[str(item) for item in list(result.get("logs") or [])],
        )
    return SkillRunResult(
        skill_name=definition.name,
        skill_version=definition.version,
        entrypoint=entrypoint.name,
        output=result,
    )


class _network_policy:
    def __init__(self, allow_network: bool) -> None:
        self.allow_network = allow_network
        self._originals: dict[str, Any] = {}

    def __enter__(self) -> None:
        if self.allow_network:
            return
        self._originals = {
            "socket": socket.socket,
            "create_connection": socket.create_connection,
            "getaddrinfo": socket.getaddrinfo,
        }

        def _deny(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Network access is disabled for this skill runtime.")

        socket.socket = _deny  # type: ignore[assignment,misc]
        socket.create_connection = _deny  # type: ignore[assignment]
        socket.getaddrinfo = _deny  # type: ignore[assignment]

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.allow_network:
            return
        for name, value in self._originals.items():
            setattr(socket, name, value)


def _artifact_to_payload(item: SkillArtifact) -> dict[str, Any]:
    return {
        "name": item.name,
        "path": item.path,
        "media_type": item.media_type,
        "role": item.role,
        "description": item.description,
        "metadata": dict(item.metadata),
    }


def _artifact_from_payload(payload: Any, *, project_root: Path, definition: SkillDefinition) -> SkillArtifact:
    if isinstance(payload, SkillArtifact):
        payload = _artifact_to_payload(payload)
    if not isinstance(payload, dict):
        raise ValidationError("Skill artifacts must be mappings.")
    raw_path = str(payload.get("path") or "")
    resolved_path = Path(raw_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = (project_root / resolved_path).resolve()
    else:
        resolved_path = resolved_path.resolve()
    allowed_write_roots = {
        _resolve_permission_root(item, project_root=project_root, definition=definition)
        for item in definition.permissions.write_paths
    }
    if not any(_is_relative_to(resolved_path, root) for root in allowed_write_roots):
        raise ValidationError(
            f'Skill "{definition.name}" artifact path "{resolved_path}" is outside the allowed write roots.'
        )
    role = str(payload.get("role") or "primary")
    if role not in {"primary", "preview", "intermediate", "report"}:
        raise ValidationError(f'Skill "{definition.name}" returned an artifact with invalid role "{role}".')
    return SkillArtifact(
        name=str(payload.get("name") or resolved_path.name),
        path=str(resolved_path),
        media_type=str(payload.get("media_type") or "") or None,
        role=role,  # type: ignore[arg-type]
        description=str(payload.get("description") or "") or None,
        metadata=dict(payload.get("metadata") or {}),
    )


def _skill_dir_from_input(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        return resolved
    if resolved.name == "SKILL.md":
        return resolved.parent
    raise ValidationError(f'Expected a skill directory or SKILL.md path, got "{path}".')


def _hash_skill_directory(path: Path) -> str:
    _validate_skill_tree(path)
    digest = hashlib.sha256()
    for file_path in sorted(
        item for item in path.rglob("*") if item.is_file() and not _is_runtime_cache_path(item, root=path)
    ):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_package_component(value: str, *, field: str) -> str:
    pattern = _PACKAGE_NAME if field == "name" else _PACKAGE_VERSION
    if not pattern.fullmatch(value):
        raise ValidationError(f'Skill package {field} "{value}" contains unsafe characters.')
    return value


def _toml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _definition_skill_root(definition: SkillDefinition) -> Path:
    if definition.install_path:
        return Path(definition.install_path).expanduser().resolve()
    if definition.package_manifest_path:
        return Path(definition.package_manifest_path).expanduser().resolve().parent
    if definition.path:
        return Path(definition.path).expanduser().resolve().parent
    if definition.source:
        return Path(definition.source).expanduser().resolve()
    raise ValidationError(f'Skill "{definition.name}" does not have a resolvable package root.')


def _validate_skill_definition(definition: SkillDefinition) -> None:
    skill_root = _definition_skill_root(definition)
    for entrypoint in definition.entrypoints:
        script = Path(str(entrypoint.script or "")).expanduser()
        if script.is_absolute() or ".." in script.parts:
            raise ValidationError(
                f'Skill "{definition.name}" entrypoint "{entrypoint.name}" script must remain inside the skill root.'
            )
        resolved_script = (skill_root / script).resolve()
        if not _is_relative_to(resolved_script, skill_root) or not resolved_script.is_file():
            raise ValidationError(
                f'Skill "{definition.name}" entrypoint "{entrypoint.name}" script must be a file inside the skill root.'
            )
    for resource_value in definition.resources:
        resource = Path(resource_value).expanduser().resolve()
        if not _is_relative_to(resource, skill_root):
            raise ValidationError(f'Skill "{definition.name}" resource "{resource}" is outside the skill root.')
    for permission_path in [*definition.permissions.read_paths, *definition.permissions.write_paths]:
        declared = Path(permission_path).expanduser()
        if declared.is_absolute() or ".." in declared.parts:
            raise ValidationError(
                f'Skill "{definition.name}" permission path "{permission_path}" must remain inside the project root.'
            )


def _resolve_permission_root(value: str, *, project_root: Path, definition: SkillDefinition) -> Path:
    declared = Path(value).expanduser()
    if declared.is_absolute():
        raise ValidationError(
            f'Skill "{definition.name}" permission path "{value}" must be relative to the project root.'
        )
    resolved = (project_root / declared).resolve()
    if not _is_relative_to(resolved, project_root):
        raise ValidationError(f'Skill "{definition.name}" permission path "{value}" escapes the project root.')
    return resolved


def _validate_skill_tree(path: Path) -> None:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ValidationError(f'Skill package root "{root}" is not a directory.')
    file_count = 0
    total_size = 0
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValidationError(f'Skill package contains unsupported symbolic link "{item}".')
        resolved = item.resolve()
        if not _is_relative_to(resolved, root):
            raise ValidationError(f'Skill package path "{item}" escapes its package root.')
        if not item.is_file():
            continue
        if _is_runtime_cache_path(item, root=root):
            continue
        file_count += 1
        if file_count > _MAX_SKILL_ARCHIVE_MEMBERS:
            raise ValidationError(f"Skill package contains more than {_MAX_SKILL_ARCHIVE_MEMBERS} files.")
        size = item.stat().st_size
        if size > _MAX_SKILL_MEMBER_BYTES:
            raise ValidationError(f'Skill package file "{item}" exceeds the per-file size limit.')
        total_size += size
        if total_size > _MAX_SKILL_EXTRACTED_BYTES:
            raise ValidationError("Skill package exceeds the total uncompressed-size limit.")


def _validate_remote_url(value: str, *, purpose: str) -> str:
    try:
        parsed = urlsplit(value)
        parsed_port = parsed.port
    except ValueError as error:
        raise ValidationError(f"Skill {purpose} URL is malformed.") from error
    if parsed.username or parsed.password or parsed.fragment:
        raise ValidationError(f"Skill {purpose} URL must not contain credentials or a fragment.")
    host = (parsed.hostname or "").lower()
    is_loopback = host in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback):
        raise ValidationError(f"Skill {purpose} URL must use HTTPS (HTTP is only allowed for loopback development).")
    if not host:
        raise ValidationError(f"Skill {purpose} URL must include a hostname.")
    if parsed_port is not None and not 1 <= parsed_port <= 65_535:
        raise ValidationError(f"Skill {purpose} URL has an invalid port.")
    return value


def _resolve_registry_artifact_url(registry_url: str, artifact_url: str) -> str:
    resolved = _validate_remote_url(urljoin(registry_url, artifact_url), purpose="artifact")
    if _url_origin(resolved) != _url_origin(registry_url):
        raise ValidationError("Skill artifact URL must use the same origin as its registry index.")
    return resolved


def _url_origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    default_port = 443 if scheme == "https" else 80
    return scheme, (parsed.hostname or "").lower(), parsed.port or default_port


def _is_remote_skill_source(value: str) -> bool:
    return urlsplit(value.split("#", 1)[0]).scheme.lower() in {"http", "https"}


def _download_url(url: str, *, timeout: float, max_bytes: int) -> bytes:
    body = bytearray()
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=False) as response:
            if 300 <= response.status_code < 400:
                raise ValidationError(f'Skill download from "{url}" refused an HTTP redirect.')
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError as error:
                    raise ValidationError(f'Skill download from "{url}" returned an invalid Content-Length.') from error
                if declared_size > max_bytes:
                    raise ValidationError(f'Skill download from "{url}" exceeds the {max_bytes}-byte limit.')
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise ValidationError(f'Skill download from "{url}" exceeds the {max_bytes}-byte limit.')
    except ValidationError:
        raise
    except httpx.HTTPError as error:
        raise ValidationError(f'Skill download from "{url}" failed: {error.__class__.__name__}.') from error
    return bytes(body)


def _is_runtime_cache_path(path: Path, *, root: Path) -> bool:
    relative = path.relative_to(root)
    return "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}


def _skill_tar_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(member.name).parts
    if "__pycache__" in parts or Path(member.name).suffix in {".pyc", ".pyo"}:
        return None
    return member
