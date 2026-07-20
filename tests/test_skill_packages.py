from __future__ import annotations

from collections.abc import AsyncIterable
from contextlib import redirect_stdout
import importlib.util
import io
import json
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
from zhivex_ai.skillpacks import _download_url, build_skill_entrypoint_tools
from zhivex_ai.skill_cli import main as skill_cli_main  # noqa: E402
from zhivex_ai.types import (
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    ToolCall,
    ToolCallPart,
)  # noqa: E402


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
                                    input={
                                        "output_path": self.output_path,
                                        "content": "hello",
                                    },
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

            result = await run_skill(
                str(skill_dir),
                input={"output_path": "out/note.txt", "content": "hello"},
                project_root=project_root,
            )

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
                await run_skill(
                    str(skill_dir),
                    input={"output_path": "../escape.txt", "content": "hello"},
                    project_root=project_root,
                )

    async def test_package_entrypoint_tool_rejects_absolute_output_outside_project_root(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(root)
            definition = load_skill(skill_dir)
            tool = build_skill_entrypoint_tools(definition, project_root=project_root)["writer_create"]
            assert tool.execute is not None
            escape_path = root / "escape.txt"

            with self.assertRaisesRegex(ValidationError, "outside the allowed write roots"):
                await tool.execute({"output_path": str(escape_path), "content": "hello"})

            self.assertFalse(escape_path.exists())

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

            def fake_download(url: str, *, timeout: float, max_bytes: int) -> bytes:
                if url == registry_url:
                    return json.dumps({"skills": index.skills}).encode("utf-8")
                if url == "https://skills.example.test/artifacts/writer-1.0.0.tar.gz":
                    return artifact_path.read_bytes()
                raise AssertionError(f"Unexpected URL {url}")

            with patch("zhivex_ai.skillpacks._download_url", side_effect=fake_download):
                installed = install_skill(
                    "writer@1.0.0",
                    project_root=project_root,
                    registry_url=registry_url,
                    trust_remote_code=True,
                )

            self.assertEqual(installed.name, "writer")
            self.assertTrue(Path(installed.install_path).exists())
            result = await run_skill(
                "writer",
                input={"output_path": "registry-note.txt", "content": "trusted"},
                project_root=project_root,
            )
            self.assertEqual(result.output["content"], "trusted")
            repeated = await run_skill(
                "writer",
                input={
                    "output_path": "registry-note-2.txt",
                    "content": "still-trusted",
                },
                project_root=project_root,
            )
            self.assertEqual(repeated.output["content"], "still-trusted")

            lock_path = project_root / ".agents" / "skills.lock.toml"
            legacy_lock = "\n".join(
                line for line in lock_path.read_text("utf-8").splitlines() if not line.startswith("content_checksum =")
            )
            lock_path.write_text(legacy_lock + "\n", "utf-8")
            with self.assertRaisesRegex(ValidationError, "legacy lock entry"):
                await run_skill(
                    "writer",
                    input={"output_path": "legacy.txt", "content": "blocked"},
                    project_root=project_root,
                )

    async def test_registry_install_requires_explicit_remote_code_trust(self) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            with self.assertRaisesRegex(ValidationError, "trust_remote_code=True"):
                install_skill(
                    "writer@1.0.0",
                    project_root=project_root,
                    registry_url="https://skills.example.test/index.json",
                )

    async def test_registry_install_rejects_insecure_remote_transport(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValidationError, "must use HTTPS"):
                install_skill(
                    "writer@1.0.0",
                    project_root=Path(tmp),
                    registry_url="http://skills.example.test/index.json",
                    trust_remote_code=True,
                )

    async def test_registry_artifact_must_keep_registry_origin(self) -> None:
        registry_url = "https://skills.example.test/index.json"
        index = {
            "skills": {
                "writer": {
                    "1.0.0": {
                        "artifact_url": "https://internal.example.test/writer.tar.gz",
                        "checksum": "0" * 64,
                    }
                }
            }
        }

        with TemporaryDirectory() as tmp:
            with patch("zhivex_ai.skillpacks._download_url", return_value=json.dumps(index).encode("utf-8")):
                with self.assertRaisesRegex(ValidationError, "same origin"):
                    install_skill(
                        "writer@1.0.0",
                        project_root=Path(tmp),
                        registry_url=registry_url,
                        trust_remote_code=True,
                    )

    async def test_download_url_enforces_incremental_limit(self) -> None:
        url = "https://skills.example.test/artifact.tar.gz"
        response = httpx.Response(
            200,
            content=b"x" * 17,
            headers={"content-length": "17"},
            request=httpx.Request("GET", url),
        )

        class StreamContext:
            def __enter__(self):
                return response

            def __exit__(self, exc_type, exc, tb):
                return None

        with patch("zhivex_ai.skillpacks.httpx.stream", return_value=StreamContext()):
            with self.assertRaisesRegex(ValidationError, "16-byte limit"):
                _download_url(url, timeout=1.0, max_bytes=16)

    async def test_entrypoint_import_runs_under_network_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(root)
            (skill_dir / "scripts" / "create.py").write_text(
                "import socket\nsocket.socket()\n\ndef run(payload, context):\n    return {}\n",
                "utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Network access is disabled"):
                await run_skill(
                    str(skill_dir),
                    input={"output_path": "note.txt", "content": "hello"},
                    project_root=project_root,
                )

    async def test_entrypoint_script_cannot_escape_skill_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(root)
            manifest = (skill_dir / "skill.yaml").read_text("utf-8")
            (skill_dir / "skill.yaml").write_text(
                manifest.replace("script: scripts/create.py", "script: ../outside.py"),
                "utf-8",
            )
            (root / "outside.py").write_text("def run(payload, context): return {}\n", "utf-8")

            with self.assertRaisesRegex(ValidationError, "escapes the skill root"):
                await run_skill(
                    str(skill_dir),
                    input={"output_path": "note.txt", "content": "hello"},
                    project_root=project_root,
                )

    async def test_local_install_rejects_symlinks_in_package_tree(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(root)
            (root / "outside.txt").write_text("secret", "utf-8")
            (skill_dir / "resources" / "link.txt").symlink_to(root / "outside.txt")

            with self.assertRaisesRegex(ValidationError, "symbolic link"):
                install_skill(skill_dir, project_root=project_root)

    async def test_local_install_rejects_permission_path_escape(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(root)
            manifest_path = skill_dir / "skill.yaml"
            manifest_path.write_text(
                manifest_path.read_text("utf-8").replace("  write_paths:\n    - .", "  write_paths:\n    - ../outside"),
                "utf-8",
            )

            with self.assertRaisesRegex(ValidationError, "inside the project root"):
                install_skill(skill_dir, project_root=project_root)

    async def test_installed_skill_checksum_is_verified_before_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(root)
            installed = install_skill(skill_dir, project_root=project_root)
            Path(installed.install_path, "scripts", "create.py").write_text(
                "def run(payload, context): return {}\n", "utf-8"
            )

            with self.assertRaisesRegex(ValidationError, "checksum verification"):
                await run_skill(
                    "writer",
                    input={"output_path": "note.txt", "content": "hello"},
                    project_root=project_root,
                )

    async def test_legacy_local_lock_checksum_remains_compatible(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(root)
            install_skill(skill_dir, project_root=project_root)
            lock_path = project_root / ".agents" / "skills.lock.toml"
            legacy_lock = "\n".join(
                line for line in lock_path.read_text("utf-8").splitlines() if not line.startswith("content_checksum =")
            )
            lock_path.write_text(legacy_lock + "\n", "utf-8")

            result = await run_skill(
                "writer",
                input={"output_path": "legacy-local.txt", "content": "compatible"},
                project_root=project_root,
            )
            self.assertEqual(result.output["content"], "compatible")

    async def test_agent_runtime_exposes_package_skill_tools_and_artifacts(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(Path(tmp))
            definition = load_skill(skill_dir)
            runtime = AgentRuntime()
            events: list[object] = []

            async def collect(event: object) -> None:
                events.append(event)

            agent = Agent(
                name="assistant",
                model=PackageToolModel("note.txt"),
                skills={"writer": definition},
                approval_policy=lambda request: True,
            )
            with patch("zhivex_ai.skillpacks.Path.cwd", return_value=project_root):
                result = await runtime.run(agent=agent, prompt="$writer create a file", emit=collect)

            self.assertEqual(result.text, "created")
            self.assertEqual(len(result.artifacts), 1)
            self.assertEqual(result.artifacts[0].name, "note.txt")
            self.assertTrue(any(isinstance(event, AgentSkillResolvedEvent) for event in events))
            self.assertTrue(any(isinstance(event, AgentSkillExecutionStartEvent) for event in events))
            self.assertTrue(any(isinstance(event, AgentSkillExecutionFinishEvent) for event in events))
            self.assertTrue(any(isinstance(event, AgentSkillArtifactCreatedEvent) for event in events))

    async def test_agent_package_skill_fails_closed_without_approval_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(Path(tmp))
            definition = load_skill(skill_dir)
            agent = Agent(
                name="assistant",
                model=PackageToolModel("blocked.txt"),
                skills={"writer": definition},
            )

            with patch("zhivex_ai.skillpacks.Path.cwd", return_value=project_root):
                result = await AgentRuntime().run(agent=agent, prompt="$writer create a file")

            self.assertFalse((project_root / "blocked.txt").exists())
            self.assertEqual(len(result.artifacts), 0)
            self.assertTrue(result.tool_results[0].is_error)
            self.assertIn("approval_policy", result.tool_results[0].error.message)

    async def test_package_skill_permissions_are_passed_to_approval_policy(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            project_root.mkdir()
            skill_dir = _write_skill_package(Path(tmp))
            definition = load_skill(skill_dir)
            requests: list[object] = []

            async def approval_policy(request: object) -> bool:
                requests.append(request)
                return True

            agent = Agent(
                name="assistant",
                model=PackageToolModel("note.txt"),
                skills={"writer": definition},
                approval_policy=approval_policy,
            )
            with patch("zhivex_ai.skillpacks.Path.cwd", return_value=project_root):
                result = await AgentRuntime().run(agent=agent, prompt="$writer create a file")

            self.assertEqual(result.text, "created")
            self.assertTrue(requests)
            request = requests[0]
            self.assertEqual(getattr(request, "tool_name"), "writer_create")
            self.assertIn("filesystem", getattr(request, "tool_permissions"))
            self.assertIn("write", getattr(request, "tool_permissions"))
            self.assertIn("code-execution", getattr(request, "tool_permissions"))
            self.assertTrue(getattr(request, "tool_metadata")["skill_package"])


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
                await run_skill(
                    "docx",
                    entrypoint="create",
                    input={"output_path": "demo.docx"},
                    project_root=ROOT,
                )

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
                    "sections": [
                        {
                            "heading": "Summary",
                            "body": "All systems nominal.",
                            "bullet_list": ["Alpha", "Beta"],
                        }
                    ],
                    "tables": [
                        {
                            "title": "Status Table",
                            "rows": [["Name", "Value"], ["Status", "OK"]],
                        }
                    ],
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
