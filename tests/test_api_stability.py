from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, ForwardRef, TypeVar, get_args, get_origin, get_type_hints
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zhivex_ai
from zhivex_ai.api_stability import (
    API_STABILITY,
    BETA_EXPORTS,
    EXPERIMENTAL_EXPORTS,
    STABLE_EXPORTS,
)


KNOWN_LEVELS = {"stable", "beta", "experimental"}
KNOWN_CATEGORIES = {
    "agent",
    "catalog",
    "errors",
    "foundation",
    "gateway",
    "messages",
    "middleware",
    "observability",
    "provider",
    "provider-support",
    "protocol",
    "realtime",
    "safety",
    "skills",
    "transport",
    "types",
    "ui",
    "workflow",
}

WORKFLOW_ERROR_EXPORTS = {
    "WorkflowConflictError",
    "WorkflowDefinitionMismatchError",
    "WorkflowInterruptError",
    "WorkflowLeaseLostError",
    "WorkflowRunNotFoundError",
}


def _public_callables(value: type[Any]) -> Iterable[tuple[str, Any]]:
    init = vars(value).get("__init__")
    if init is not None and init is not object.__init__:
        yield "__init__", init
    call = vars(value).get("__call__")
    if call is not None:
        yield "__call__", call
    for name, member in vars(value).items():
        if name.startswith("_"):
            continue
        if isinstance(member, (classmethod, staticmethod)):
            yield name, member.__func__
        elif isinstance(member, property) and member.fget is not None:
            yield name, member.fget
        elif inspect.isfunction(member):
            yield name, member


def _resolved_hints(value: Any) -> dict[str, Any]:
    module = sys.modules.get(getattr(value, "__module__", ""))
    globalns = dict(vars(module)) if module is not None else {}
    globalns.update({name: getattr(zhivex_ai, name) for name in zhivex_ai.__all__})
    localns = vars(value) if inspect.isclass(value) else None
    return get_type_hints(
        value,
        globalns=globalns,
        localns=localns,
        include_extras=True,
    )


def _annotation_members(annotation: Any) -> Iterable[Any]:
    yield annotation
    if isinstance(annotation, TypeVar):
        if annotation.__bound__ is not None:
            yield from _annotation_members(annotation.__bound__)
        for constraint in annotation.__constraints__:
            yield from _annotation_members(constraint)
        return
    if isinstance(annotation, ForwardRef):
        return
    if get_origin(annotation) is Annotated:
        annotated_type, *metadata = get_args(annotation)
        if any(level in {"beta", "experimental"} for level in metadata):
            return
        yield from _annotation_members(annotated_type)
        return
    for argument in get_args(annotation):
        yield from _annotation_members(argument)


def _stable_dependency_violations(root_names: Iterable[str]) -> list[str]:
    export_names_by_id: dict[int, set[str]] = {}
    exported_objects: dict[str, Any] = {}
    for export_name in zhivex_ai.__all__:
        exported = getattr(zhivex_ai, export_name)
        exported_objects[export_name] = exported
        export_names_by_id.setdefault(id(exported), set()).add(export_name)

    unstable = BETA_EXPORTS | EXPERIMENTAL_EXPORTS
    violations: set[str] = set()
    pending = list(root_names)
    visited: set[str] = set()

    def check_annotation(root_name: str, surface: str, annotation: Any) -> None:
        for dependency in _annotation_members(annotation):
            for dependency_name in export_names_by_id.get(id(dependency), ()):
                if dependency_name in unstable:
                    violations.add(
                        f"{root_name}.{surface} -> {dependency_name} "
                        f"({API_STABILITY[dependency_name].level})"
                    )

    def check_hints(root_name: str, surface: str, value: Any) -> None:
        if not getattr(value, "__annotations__", None):
            return
        try:
            hints = _resolved_hints(value)
        except (NameError, TypeError) as error:
            violations.add(f"{root_name}.{surface} has unresolved annotations: {error}")
            return
        for annotation in hints.values():
            check_annotation(root_name, surface, annotation)

    while pending:
        root_name = pending.pop()
        if root_name in visited:
            continue
        visited.add(root_name)
        root = exported_objects[root_name]
        if get_origin(root) is not None:
            check_annotation(root_name, "alias", root)
        check_hints(root_name, "annotations", root)
        if inspect.isclass(root):
            for member_name, member in _public_callables(root):
                check_hints(root_name, member_name, member)
    return sorted(violations)


class ApiStabilityTests(TestCase):
    def test_non_portable_provider_factories_are_experimental(self) -> None:
        factories = {"create_bedrock", "create_ollama", "create_openrouter"}

        self.assertTrue(factories.issubset(EXPERIMENTAL_EXPORTS))
        self.assertFalse(factories & BETA_EXPORTS)

    def test_meta_factory_is_stable_tier_1_while_native_helpers_remain_beta(self) -> None:
        self.assertNotIn("create_meta", BETA_EXPORTS)
        self.assertIn("create_meta", STABLE_EXPORTS)
        self.assertNotIn("create_meta", EXPERIMENTAL_EXPORTS)
        self.assertEqual(API_STABILITY["create_meta"].category, "provider")
        self.assertIn("tier-1 provider", API_STABILITY["create_meta"].notes)
        for helper in ("meta_hosted_tool", "meta_tool_search_tool", "meta_web_search_tool"):
            self.assertIn(helper, BETA_EXPORTS)
            self.assertNotIn(helper, STABLE_EXPORTS)

    def test_stable_workflow_surface_only_references_stable_contracts(self) -> None:
        workflow_stable_exports = {
            name
            for name in STABLE_EXPORTS
            if API_STABILITY[name].category == "workflow"
        } | WORKFLOW_ERROR_EXPORTS

        self.assertEqual(_stable_dependency_violations(workflow_stable_exports), [])

    def test_workflow_errors_are_stable_error_entries(self) -> None:
        for name in WORKFLOW_ERROR_EXPORTS:
            self.assertEqual(API_STABILITY[name].level, "stable")
            self.assertEqual(API_STABILITY[name].category, "errors")

    def test_named_engine_adapter_factories_remain_beta(self) -> None:
        factories = {
            "create_dbos_workflow_adapter",
            "create_prefect_workflow_adapter",
            "create_restate_workflow_adapter",
            "create_temporal_workflow_adapter",
        }
        self.assertTrue(factories.issubset(BETA_EXPORTS))
        self.assertFalse(factories & STABLE_EXPORTS)

    def test_every_public_export_has_stability_metadata(self) -> None:
        self.assertEqual(set(zhivex_ai.__all__), set(API_STABILITY))

    def test_manifest_sets_are_explicit_and_non_overlapping(self) -> None:
        stable = set(STABLE_EXPORTS)
        beta = set(BETA_EXPORTS)
        experimental = set(EXPERIMENTAL_EXPORTS)

        self.assertFalse(stable & beta)
        self.assertFalse(stable & experimental)
        self.assertFalse(beta & experimental)
        self.assertEqual(set(zhivex_ai.__all__), stable | beta | experimental)

    def test_manifest_uses_known_levels_and_categories(self) -> None:
        for entry in API_STABILITY.values():
            self.assertIn(entry.level, KNOWN_LEVELS)
            self.assertIn(entry.category, KNOWN_CATEGORIES)

    def test_manifest_does_not_mask_missing_module_mappings(self) -> None:
        for name, entry in API_STABILITY.items():
            self.assertEqual(entry.name, name)
            self.assertIn(name, zhivex_ai._EXPORTS)

    def test_stable_exports_are_documented_in_stability_doc(self) -> None:
        stability = (ROOT / "STABILITY.md").read_text("utf-8")
        for symbol in sorted(STABLE_EXPORTS):
            self.assertIn(f"`{symbol}`", stability)

    def test_experimental_exports_are_not_listed_as_stable(self) -> None:
        stability = (ROOT / "STABILITY.md").read_text("utf-8")
        stable_section = stability.split("## Beta", 1)[0]

        for symbol in sorted(EXPERIMENTAL_EXPORTS):
            self.assertNotIn(f"`{symbol}`", stable_section)

    def test_stability_doc_points_to_manifest_as_drift_gate(self) -> None:
        stability = (ROOT / "STABILITY.md").read_text("utf-8")
        versioning = (ROOT / "VERSIONING.md").read_text("utf-8")

        self.assertIn("api_stability.py", stability)
        self.assertIn("api_stability.py", versioning)
