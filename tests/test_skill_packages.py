from __future__ import annotations

from collections.abc import AsyncIterable
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, skipUnless
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    AgentRuntime,
    AgentSkillArtifactCreatedEvent,
    AgentSkillExecutionFinishEvent,
    AgentSkillExecutionStartEvent,
    AgentSkillResolvedEvent,
    ValidationError,
    install_skill,
    list_installed_skills,
    load_skill,
    load_skill_package,
    publish_skill,
    run_skill,
)
from zhivex_ai.skill_cli import main as skill_cli_main  # noqa: E402
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


def _write_skill_package(root: Path, *, name: str = "writer", version: str = "1.0.0") -> Path:
    skill_dir = root / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "resources").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use this skill for writing files.\n---\n\nCreate real files when needed.\n",
        "utf-8",
    )
    (skill_dir / "skill.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"name: {name}",
                f"version: {version}",
                "description: Use this skill for writing files.",
                "entrypoints:",
                "  - name: create",
                "    description: Create a text file.",
                "    script: scripts/create.py",
                "    default: true",
                "    input_schema:",
                "      type: object",
                "      required:",
                "        - output_path",
                "        - content",
                "      properties:",
                "        output_path:",
                "          type: string",
                "        content:",
                "          type: string",
                "permissions:",
                "  allow_network: false",
                "  write_paths:",
                "    - .",
                "resources:",
                "  - resources",
            ]
        )
        + "\n",
        "utf-8",
    )
    (skill_dir / "scripts" / "create.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from pathlib import Path",
                "",
                "def run(payload, context):",
                "    project_root = Path(context['project_root'])",
                "    output_path = Path(payload['output_path'])",
                "    if not output_path.is_absolute():",
                "        output_path = (project_root / output_path).resolve()",
                "    output_path.parent.mkdir(parents=True, exist_ok=True)",
                "    output_path.write_text(str(payload['content']), 'utf-8')",
                "    return {",
                "        'output': {'path': str(output_path), 'content': str(payload['content'])},",
                "        'artifacts': [",
                "            {",
                "                'name': output_path.name,",
                "                'path': str(output_path),",
                "                'media_type': 'text/plain',",
                "                'role': 'primary',",
                "            }",
                "        ],",
                "    }",
            ]
        )
        + "\n",
        "utf-8",
    )
    return skill_dir


class PackageToolModel:
    provider = "openai"
    model_id = "package-tool"
    capabilities = BASE_CAPABILITIES

    def __init__(self, output_path: str = "note.txt") -> None:
        self.output_path = output_path

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        has_tool_message = any(message.role == "tool" for message in input.messages)
        if not has_tool_message:
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[
                            ToolCallPart(
                                tool_call=ToolCall(
                                    id="call_1",
                                    name="writer_create",
                                    input={"output_path": self.output_path, "content": "hello"},
                                )
                            )
                        ],
                    )
                ]
            )
        return GenerateResult(text="created", messages=[])

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        raise NotImplementedError


class SkillPackageTests(IsolatedAsyncioTestCase):
    async def test_load_skill_package_merges_skill_yaml(self) -> None:
        with TemporaryDirectory() as tmp:
            skill_dir = _write_skill_package(Path(tmp))
            definition = load_skill_package(skill_dir)

        self.assertEqual(definition.name, "writer")
        self.assertEqual(definition.version, "1.0.0")
        self.assertEqual([entry.name for entry in definition.entrypoints], ["create"])
        self.assertTrue(definition.resources)

    async def test_run_skill_executes_entrypoint_and_returns_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(Path(tmp))

            result = await run_skill(str(skill_dir), input={"output_path": "out/note.txt", "content": "hello"}, project_root=project_root)

            self.assertEqual(result.skill_name, "writer")
            self.assertEqual(result.entrypoint, "create")
            self.assertEqual(result.output["content"], "hello")
            self.assertEqual(len(result.artifacts), 1)
            self.assertTrue((project_root / "out" / "note.txt").exists())

    async def test_run_skill_blocks_output_outside_project_root(self) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(Path(tmp))
            with self.assertRaisesRegex(ValidationError, "outside the allowed write roots"):
                await run_skill(str(skill_dir), input={"output_path": "../escape.txt", "content": "hello"}, project_root=project_root)

    async def test_install_skill_from_path_updates_manifest_and_lockfile(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(root)

            installed = install_skill(skill_dir, project_root=project_root)
            items = list_installed_skills(project_root=project_root)

            self.assertEqual(installed.name, "writer")
            self.assertEqual(len(items), 1)
            self.assertTrue((project_root / ".agents" / "skills.toml").exists())
            self.assertTrue((project_root / ".agents" / "skills.lock.toml").exists())

    async def test_publish_and_install_from_http_registry(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_root = root / "registry"
            skill_dir = _write_skill_package(root)
            index = publish_skill(skill_dir, registry_dir=registry_root)
            artifact_name = next(iter(index.skills["writer"]["1.0.0"]["artifact_url"].split("/")[-1:]))
            artifact_path = registry_root / "artifacts" / artifact_name
            project_root = root / "project"
            project_root.mkdir()

            registry_url = "https://skills.example.test/index.json"

            def fake_get(url: str, timeout: float):
                if url == registry_url:
                    return httpx.Response(200, json={"skills": index.skills}, request=httpx.Request("GET", url))
                if url == "https://skills.example.test/artifacts/writer-1.0.0.tar.gz":
                    return httpx.Response(200, content=artifact_path.read_bytes(), request=httpx.Request("GET", url))
                raise AssertionError(f"Unexpected URL {url}")

            with patch("zhivex_ai.skillpacks.httpx.get", side_effect=fake_get):
                installed = install_skill("writer@1.0.0", project_root=project_root, registry_url=registry_url)

            self.assertEqual(installed.name, "writer")
            self.assertTrue(Path(installed.install_path).exists())

    async def test_agent_runtime_exposes_package_skill_tools_and_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(Path(tmp))
            definition = load_skill(skill_dir)
            runtime = AgentRuntime()
            events: list[object] = []

            async def collect(event: object) -> None:
                events.append(event)

            agent = Agent(name="assistant", model=PackageToolModel(str(project_root / "note.txt")), skills={"writer": definition})
            result = await runtime.run(agent=agent, prompt="$writer create a file", emit=collect)

            self.assertEqual(result.text, "created")
            self.assertEqual(len(result.artifacts), 1)
            self.assertEqual(result.artifacts[0].name, "note.txt")
            self.assertTrue(any(isinstance(event, AgentSkillResolvedEvent) for event in events))
            self.assertTrue(any(isinstance(event, AgentSkillExecutionStartEvent) for event in events))
            self.assertTrue(any(isinstance(event, AgentSkillExecutionFinishEvent) for event in events))
            self.assertTrue(any(isinstance(event, AgentSkillArtifactCreatedEvent) for event in events))


class SkillCLITests(TestCase):
    def test_cli_validate_and_run_commands(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                validate_exit = skill_cli_main(["validate", str(skill_dir)])
                run_exit = skill_cli_main(
                    [
                        "run",
                        str(skill_dir),
                        "--project-root",
                        str(project_root),
                        "--input",
                        '{"output_path":"cli.txt","content":"from-cli"}',
                    ]
                )

            output = stdout.getvalue()
            self.assertEqual(validate_exit, 0)
            self.assertEqual(run_exit, 0)
            self.assertIn('"name": "writer"', output)
            self.assertIn('"skill_name": "writer"', output)


class DocxSkillTests(IsolatedAsyncioTestCase):
    async def test_docx_skill_missing_dependency_fails_clearly(self) -> None:
        original_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str, package: str | None = None):
            if name == "docx":
                return None
            return original_find_spec(name, package)

        with patch("zhivex_ai.skillpacks.importlib.util.find_spec", side_effect=fake_find_spec):
            with self.assertRaisesRegex(RuntimeError, "python-docx"):
                await run_skill("docx", entrypoint="create", input={"output_path": "demo.docx"}, project_root=ROOT)

    @skipUnless(importlib.util.find_spec("docx") is not None, "python-docx is not installed")
    async def test_docx_skill_create_edit_and_analyze(self) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            created = await run_skill(
                "docx",
                entrypoint="create",
                input={
                    "output_path": "report.docx",
                    "title": "Quarterly Review",
                    "subtitle": "Q1 FY26",
                    "paragraphs": ["Opening paragraph."],
                    "sections": [{"heading": "Summary", "body": "All systems nominal.", "bullet_list": ["Alpha", "Beta"]}],
                    "tables": [{"title": "Status Table", "rows": [["Name", "Value"], ["Status", "OK"]]}],
                    "properties": {"author": "Zhivex", "subject": "Review"},
                },
                project_root=project_root,
            )
            edited = await run_skill(
                "docx",
                entrypoint="edit",
                input={
                    "input_path": "report.docx",
                    "output_path": "report-edited.docx",
                    "append_paragraphs": ["Follow-up paragraph."],
                    "append_sections": [{"heading": "Next Steps", "paragraphs": ["Ship it."]}],
                    "replace_text": [{"old": "nominal", "new": "green"}],
                    "append_tables": [{"rows": [["Owner", "Team"], ["SDK", "Platform"]]}],
                },
                project_root=project_root,
            )
            analyzed = await run_skill(
                "docx",
                entrypoint="analyze",
                input={"input_path": "report-edited.docx", "include_paragraphs": True},
                project_root=project_root,
            )

            self.assertTrue((project_root / "report.docx").exists())
            self.assertTrue((project_root / "report-edited.docx").exists())
            self.assertEqual(len(created.artifacts), 1)
            self.assertEqual(len(edited.artifacts), 1)
            self.assertIn("Quarterly Review", analyzed.output["text"])
            self.assertIn("green", analyzed.output["text"])
            self.assertGreaterEqual(analyzed.output["table_count"], 2)
            self.assertGreaterEqual(analyzed.output["heading_count"], 2)
            self.assertGreaterEqual(analyzed.output["paragraph_count"], 4)
            self.assertEqual(analyzed.output["properties"]["author"], "Zhivex")
            self.assertTrue(analyzed.output["paragraphs"])
