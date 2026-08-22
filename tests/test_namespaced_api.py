from __future__ import annotations

import unittest
from types import ModuleType

import zhivex_ai
from zhivex_ai import evals, experimental, integrations, workflows
from zhivex_ai.api_stability import EXPERIMENTAL_EXPORTS
from zhivex_ai.experimental import providers as experimental_providers
from zhivex_ai.experimental import realtime as experimental_realtime
from zhivex_ai.integrations import protocols as integration_protocols
from zhivex_ai.integrations import responses as integration_responses


class NamespacedApiTests(unittest.TestCase):
    def assert_compatibility_namespace(
        self,
        namespace: ModuleType,
        *,
        source_modules: set[str],
        companion_exports: set[str] | None = None,
    ) -> None:
        expected = {
            name
            for name, source_module in zhivex_ai._EXPORTS.items()
            if source_module in source_modules
        }
        expected.update(companion_exports or ())

        self.assertEqual(set(namespace.__all__), expected)
        for name in expected:
            self.assertIs(getattr(namespace, name), getattr(zhivex_ai, name))

    def test_evals_groups_the_existing_evaluation_contract(self) -> None:
        self.assert_compatibility_namespace(
            evals,
            source_modules={"agent_evaluation"},
            companion_exports={"GenerateResult"},
        )

    def test_workflows_groups_all_existing_workflow_contracts(self) -> None:
        self.assert_compatibility_namespace(
            workflows,
            source_modules={
                "workflow",
                "workflow_adapters",
                "workflow_graph",
                "workflow_state",
            },
            companion_exports={
                "WorkflowConflictError",
                "WorkflowDefinitionMismatchError",
                "WorkflowInterruptError",
                "WorkflowLeaseLostError",
                "WorkflowRunNotFoundError",
            },
        )

    def test_integrations_groups_protocol_and_responses_contracts(self) -> None:
        self.assert_compatibility_namespace(
            integrations,
            source_modules={"protocols", "responses_host"},
        )
        self.assert_compatibility_namespace(
            integration_protocols,
            source_modules={"protocols"},
        )
        self.assert_compatibility_namespace(
            integration_responses,
            source_modules={"responses_host"},
        )

    def test_experimental_namespace_matches_the_stability_manifest(self) -> None:
        self.assertEqual(set(experimental.__all__), set(EXPERIMENTAL_EXPORTS))
        for name in EXPERIMENTAL_EXPORTS:
            self.assertIs(getattr(experimental, name), getattr(zhivex_ai, name))

        self.assertEqual(
            set(experimental_providers.__all__),
            {
                "create_bedrock",
                "create_ollama",
                "create_openrouter",
                "openai_local_shell_tool",
                "openai_shell_environment",
                "openai_shell_tool",
            },
        )
        self.assertEqual(
            set(experimental_realtime.__all__),
            set(EXPERIMENTAL_EXPORTS) - set(experimental_providers.__all__),
        )

    def test_namespaces_do_not_expand_the_legacy_root_wildcard_contract(self) -> None:
        self.assertTrue(
            {"evals", "experimental", "integrations", "workflows"}.isdisjoint(
                zhivex_ai.__all__
            )
        )

    def test_namespace_modules_explain_their_compatibility_role(self) -> None:
        for namespace in (
            evals,
            workflows,
            integrations,
            integration_protocols,
            integration_responses,
            experimental,
            experimental_providers,
            experimental_realtime,
        ):
            self.assertIsNotNone(namespace.__doc__)
            self.assertTrue(namespace.__doc__.strip())


if __name__ == "__main__":
    unittest.main()
