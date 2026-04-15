from __future__ import annotations

from collections.abc import AsyncIterable
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    AgentRuntime,
    AgentSkillActivatedEvent,
    AgentSkillSkippedEvent,
    clear_agent_session_skills,
    create_agent_session,
    create_in_memory_agent_memory_store,
    create_text_message,
    get_agent_session_skills,
    resume_agent,
    run_agent,
    set_agent_session_skills,
    skill,
    tool,
)
from zhivex_ai.skills import SkillDependency, discover_skills, load_skill  # noqa: E402
from zhivex_ai.types import GenerateResult, ModelCapabilities, ModelGenerateInput, ModelMessage, ToolCall, ToolCallPart  # noqa: E402


BASE_CAPABILITIES = ModelCapabilities(
    streaming=False,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=False,
    vision=False,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=False,
    web_search=False,
)


class SkillEchoModel:
    capabilities = BASE_CAPABILITIES

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.model_id = "skill-echo"

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        system_text = "\n".join(
            "".join(part.text for part in message.parts if part.type == "text")
            for message in input.messages
            if message.role == "system"
        )
        if "[Active skill: release-notes]" in system_text:
            return GenerateResult(messages=[create_text_message("assistant", f"{self.provider}:skill-on")], text=f"{self.provider}:skill-on")
        return GenerateResult(messages=[create_text_message("assistant", f"{self.provider}:skill-off")], text=f"{self.provider}:skill-off")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        raise NotImplementedError


class ToolUsingSkillModel:
    provider = "openai"
    model_id = "skill-tool"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        has_tool_message = any(message.role == "tool" for message in input.messages)
        if not has_tool_message:
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="call_1", name="docs_lookup", input={"topic": "skills"}))],
                    )
                ]
            )
        return GenerateResult(messages=[create_text_message("assistant", "dependency-loaded")], text="dependency-loaded")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        raise NotImplementedError


class ConflictToolSkillModel:
    provider = "openai"
    model_id = "skill-conflict"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        system_text = "\n".join(
            "".join(part.text for part in message.parts if part.type == "text")
            for message in input.messages
            if message.role == "system"
        )
        if "[Active skill: strong-skill]" in system_text and "[Active skill: weak-skill]" not in system_text:
            return GenerateResult(messages=[create_text_message("assistant", "priority-ok")], text="priority-ok")
        return GenerateResult(messages=[create_text_message("assistant", "priority-bad")], text="priority-bad")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        raise NotImplementedError


class SkillRuntimeTests(IsolatedAsyncioTestCase):
    async def test_explicit_skill_activation_is_provider_agnostic(self) -> None:
        release_skill = skill(
            name="release-notes",
            description="Use when writing release notes or changelog summaries.",
            instructions="Summarize user-visible changes and migrations.",
        )
        for provider in ("openai", "azure-openai", "anthropic", "gemini", "vertex"):
            agent = Agent(name="assistant", model=SkillEchoModel(provider), skills={"release-notes": release_skill})
            result = await run_agent(agent=agent, prompt="$release-notes summarize the latest changes")
            self.assertEqual(result.text, f"{provider}:skill-on")
            self.assertEqual(result.session.metadata["active_skills"][0]["name"], "release-notes")

    async def test_implicit_skill_activation_matches_description(self) -> None:
        release_skill = skill(
            name="release-notes",
            description="Use when summarizing changelog entries and release note requests.",
            instructions="Focus on user-visible behavior.",
        )
        agent = Agent(name="assistant", model=SkillEchoModel("openai"), skills={"release-notes": release_skill})
        result = await run_agent(agent=agent, prompt="Please summarize the changelog into release notes.")
        self.assertEqual(result.text, "openai:skill-on")

    async def test_implicit_skill_activation_respects_policy(self) -> None:
        release_skill = skill(
            name="release-notes",
            description="Use when summarizing changelog entries and release note requests.",
            instructions="Focus on user-visible behavior.",
            allow_implicit_invocation=False,
        )
        agent = Agent(name="assistant", model=SkillEchoModel("openai"), skills={"release-notes": release_skill})
        result = await run_agent(agent=agent, prompt="Please summarize the changelog into release notes.")
        self.assertEqual(result.text, "openai:skill-off")

    async def test_skill_activation_honors_triggers_anti_triggers_and_provider_rules(self) -> None:
        release_skill = skill(
            name="release-notes",
            description="Use when summarizing changelog entries and release note requests.",
            instructions="Focus on user-visible behavior.",
            triggers=["release notes"],
            anti_triggers=["draft contract"],
            allowed_providers=["openai"],
            allowed_models=["skill-*"],
        )
        openai_agent = Agent(name="assistant", model=SkillEchoModel("openai"), skills={"release-notes": release_skill})
        blocked_provider = Agent(name="assistant", model=SkillEchoModel("gemini"), skills={"release-notes": release_skill})

        active = await run_agent(agent=openai_agent, prompt="Please write release notes for this SDK.")
        anti_triggered = await run_agent(agent=openai_agent, prompt="Please write release notes for this draft contract.")
        blocked = await run_agent(agent=blocked_provider, prompt="Please write release notes for this SDK.")

        self.assertEqual(active.text, "openai:skill-on")
        self.assertEqual(anti_triggered.text, "openai:skill-off")
        self.assertEqual(blocked.text, "gemini:skill-off")

    async def test_skill_dependency_tools_are_merged_for_agent_runs(self) -> None:
        docs_skill = skill(
            name="docs",
            description="Use when looking up docs.",
            instructions="Use the docs lookup tool when needed.",
            dependencies=[SkillDependency(type="mcp", value="docs", url="https://mcp.example.com")],
        )

        async def fake_discover_mcp_tools(*args, **kwargs):
            return {
                "docs_lookup": tool(
                    name="docs_lookup",
                    description="Lookup docs",
                    schema=dict[str, str],
                    execute=lambda input: {"topic": input["topic"], "ok": True},
                )
            }

        with patch("zhivex_ai.agent.discover_mcp_tools", side_effect=fake_discover_mcp_tools):
            agent = Agent(name="assistant", model=ToolUsingSkillModel(), skills={"docs": docs_skill})
            result = await run_agent(agent=agent, prompt="$docs explain skills")

        self.assertEqual(result.text, "dependency-loaded")
        self.assertEqual(result.tool_results[0].tool_name, "docs_lookup")
        self.assertFalse(result.tool_results[0].is_error)

    async def test_implicit_skill_dependency_failures_are_skipped(self) -> None:
        docs_skill = skill(
            name="docs",
            description="Use when looking up docs.",
            instructions="Use the docs lookup tool when needed.",
            triggers=["skills"],
            dependencies=[SkillDependency(type="mcp", value="docs", url="https://mcp.example.com")],
        )
        runtime = AgentRuntime()
        events: list[object] = []

        async def collect(event: object) -> None:
            events.append(event)

        with patch("zhivex_ai.agent.discover_mcp_tools", side_effect=RuntimeError("MCP offline")):
            agent = Agent(name="assistant", model=SkillEchoModel("openai"), skills={"docs": docs_skill})
            result = await runtime.run(agent=agent, prompt="Explain skills", emit=collect)

        self.assertEqual(result.text, "openai:skill-off")
        skipped = [event for event in events if isinstance(event, AgentSkillSkippedEvent)]
        self.assertEqual(len(skipped), 1)
        self.assertIn("dependency resolution failed", skipped[0].reason)

    async def test_explicit_skill_dependency_failures_raise(self) -> None:
        docs_skill = skill(
            name="docs",
            description="Use when looking up docs.",
            instructions="Use the docs lookup tool when needed.",
            dependency_failure_mode="fail",
            dependencies=[SkillDependency(type="mcp", value="docs", url="https://mcp.example.com")],
        )
        with patch("zhivex_ai.agent.discover_mcp_tools", side_effect=RuntimeError("MCP offline")):
            agent = Agent(name="assistant", model=SkillEchoModel("openai"), skills={"docs": docs_skill})
            with self.assertRaisesRegex(Exception, 'Skill "docs" could not be activated'):
                await run_agent(agent=agent, prompt="$docs explain skills")

    async def test_skill_priority_breaks_conflicts_in_favor_of_higher_priority_skill(self) -> None:
        shared_tool_a = tool(name="shared_lookup", schema=dict[str, str], execute=lambda input: {"tool": "a"})
        shared_tool_b = tool(name="shared_lookup", schema=dict[str, str], execute=lambda input: {"tool": "b"})
        strong = skill(
            name="strong-skill",
            description="Use when triaging priority conflicts.",
            instructions="Handle the request.",
            triggers=["priority"],
            priority=10,
            tools={"shared_lookup": shared_tool_a},
        )
        weak = skill(
            name="weak-skill",
            description="Use when triaging priority conflicts.",
            instructions="Handle the request.",
            triggers=["priority"],
            priority=1,
            tools={"shared_lookup": shared_tool_b},
        )
        runtime = AgentRuntime()
        events: list[object] = []

        async def collect(event: object) -> None:
            events.append(event)

        agent = Agent(
            name="assistant",
            model=ConflictToolSkillModel(),
            skills={"strong-skill": strong, "weak-skill": weak},
        )
        result = await runtime.run(agent=agent, prompt="Resolve this priority issue.", emit=collect)

        self.assertEqual(result.text, "priority-ok")
        activated = [event for event in events if isinstance(event, AgentSkillActivatedEvent)]
        skipped = [event for event in events if isinstance(event, AgentSkillSkippedEvent)]
        self.assertEqual([event.skill_name for event in activated], ["strong-skill"])
        self.assertEqual(skipped[0].skill_name, "weak-skill")
        self.assertIn("conflicting tools", skipped[0].reason)

    async def test_active_skills_stick_to_the_session_and_resume_agent(self) -> None:
        memory = create_in_memory_agent_memory_store()
        release_skill = skill(
            name="release-notes",
            description="Use when writing release notes or changelog summaries.",
            instructions="Summarize user-visible changes and migrations.",
        )
        agent = Agent(
            name="assistant",
            model=SkillEchoModel("openai"),
            skills={"release-notes": release_skill},
            memory=memory,
        )
        session = create_agent_session()

        first = await run_agent(agent=agent, session=session, prompt="$release-notes summarize the latest changes")
        second = await run_agent(agent=agent, session=session, prompt="Keep going.")
        resumed = await resume_agent(agent=agent, session_id=session.id, prompt="One more update.")

        self.assertEqual(first.text, "openai:skill-on")
        self.assertEqual(second.text, "openai:skill-on")
        self.assertEqual(resumed.text, "openai:skill-on")
        self.assertEqual(first.session.metadata["sticky_skills"], ["release-notes"])

    async def test_agent_runtime_emits_skill_activation_and_skip_events(self) -> None:
        release_skill = skill(
            name="release-notes",
            description="Use when writing release notes or changelog summaries.",
            instructions="Summarize user-visible changes and migrations.",
        )
        agent = Agent(name="assistant", model=SkillEchoModel("openai"), skills={"release-notes": release_skill})
        runtime = AgentRuntime()
        events: list[object] = []
        session = create_agent_session()

        async def collect(event: object) -> None:
            events.append(event)

        await runtime.run(agent=agent, session=session, prompt="$release-notes summarize the latest changes", emit=collect)
        await runtime.run(agent=agent, session=session, prompt="Keep going.", emit=collect)

        activated = [event for event in events if isinstance(event, AgentSkillActivatedEvent)]
        self.assertEqual([event.activation for event in activated], ["explicit", "sticky"])
        self.assertEqual([event.skill_name for event in activated], ["release-notes", "release-notes"])

        missing_session = create_agent_session(metadata={"sticky_skills": ["missing-skill"]})
        skipped_events: list[object] = []
        async def collect_skipped(event: object) -> None:
            skipped_events.append(event)

        await runtime.run(agent=agent, session=missing_session, prompt="hello", emit=collect_skipped)
        skipped = [event for event in skipped_events if isinstance(event, AgentSkillSkippedEvent)]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].skill_name, "missing-skill")
        self.assertEqual(skipped[0].activation, "sticky")

    async def test_set_agent_session_skills_replaces_sticky_skills(self) -> None:
        release_skill = skill(
            name="release-notes",
            description="Use when writing release notes or changelog summaries.",
            instructions="Summarize user-visible changes and migrations.",
        )
        agent = Agent(name="assistant", model=SkillEchoModel("openai"), skills={"release-notes": release_skill})
        session = create_agent_session()

        set_agent_session_skills(session, "release-notes", "release-notes")
        result = await run_agent(agent=agent, session=session, prompt="Keep going.")

        self.assertEqual(result.text, "openai:skill-on")
        self.assertEqual(get_agent_session_skills(result.session), ["release-notes"])
        self.assertEqual(result.session.metadata["active_skills"][0]["activation"], "sticky")

    async def test_clear_agent_session_skills_disables_sticky_reactivation(self) -> None:
        release_skill = skill(
            name="release-notes",
            description="Use when writing release notes or changelog summaries.",
            instructions="Summarize user-visible changes and migrations.",
        )
        agent = Agent(name="assistant", model=SkillEchoModel("openai"), skills={"release-notes": release_skill})
        session = create_agent_session()

        await run_agent(agent=agent, session=session, prompt="$release-notes summarize the latest changes")
        clear_agent_session_skills(session)
        result = await run_agent(agent=agent, session=session, prompt="Keep going.")

        self.assertEqual(result.text, "openai:skill-off")
        self.assertEqual(result.session.metadata["sticky_skills"], [])
        self.assertEqual(result.session.metadata["active_skills"], [])

    async def test_non_persistent_skill_does_not_stick_to_session(self) -> None:
        transient = skill(
            name="release-notes",
            description="Use when writing release notes or changelog summaries.",
            instructions="Summarize user-visible changes and migrations.",
            persist_to_session=False,
        )
        agent = Agent(name="assistant", model=SkillEchoModel("openai"), skills={"transient": transient})
        session = create_agent_session()

        await run_agent(agent=agent, session=session, prompt="$transient summarize the latest changes")

        self.assertEqual(get_agent_session_skills(session), [])


class SkillDiscoveryTests(IsolatedAsyncioTestCase):
    def test_load_skill_parses_frontmatter_and_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "release-notes"
            (skill_dir / "agents").mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: release-notes\ndescription: Summarize changelog updates.\n---\n\nUse concise release note sections.\n",
                "utf-8",
            )
            (skill_dir / "agents" / "openai.yaml").write_text(
                'interface:\n'
                '  display_name: "Release Notes"\n'
                '  default_prompt: "Draft release notes"\n'
                "policy:\n"
                "  allow_implicit_invocation: false\n"
                "  priority: 7\n"
                "  triggers:\n"
                '    - "release notes"\n'
                "  anti_triggers:\n"
                '    - "draft contract"\n'
                "  allowed_providers:\n"
                '    - "openai"\n'
                "  allowed_models:\n"
                '    - "gpt-*"\n'
                "  persist_to_session: false\n"
                '  dependency_failure_mode: "fail"\n'
                "dependencies:\n"
                "  tools:\n"
                '    - type: "mcp"\n'
                '      value: "docs"\n'
                '      transport: "streamable_http"\n'
                '      url: "https://mcp.example.com"\n',
                "utf-8",
            )

            definition = load_skill(skill_dir)

        self.assertEqual(definition.name, "release-notes")
        self.assertEqual(definition.display_name, "Release Notes")
        self.assertEqual(definition.default_prompt, "Draft release notes")
        self.assertFalse(definition.allow_implicit_invocation)
        self.assertEqual(definition.priority, 7)
        self.assertEqual(definition.triggers, ["release notes"])
        self.assertEqual(definition.anti_triggers, ["draft contract"])
        self.assertEqual(definition.allowed_providers, ["openai"])
        self.assertEqual(definition.allowed_models, ["gpt-*"])
        self.assertFalse(definition.persist_to_session)
        self.assertEqual(definition.dependency_failure_mode, "fail")
        self.assertEqual(definition.dependencies[0].url, "https://mcp.example.com")

    def test_discover_skills_scans_repo_parent_skill_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            root_skill = root / ".agents" / "skills" / "release-notes"
            nested_skill = root / "services" / "billing" / ".agents" / "skills" / "billing"
            nested_skill.mkdir(parents=True)
            root_skill.mkdir(parents=True)
            (root_skill / "SKILL.md").write_text(
                "---\nname: release-notes\ndescription: Summarize releases.\n---\n\nUse release-note formatting.\n",
                "utf-8",
            )
            (nested_skill / "SKILL.md").write_text(
                "---\nname: billing\ndescription: Help with billing domain requests.\n---\n\nUse billing language.\n",
                "utf-8",
            )

            discovered = discover_skills(cwd=root / "services" / "billing")

        self.assertEqual(set(discovered), {"billing", "release-notes"})
