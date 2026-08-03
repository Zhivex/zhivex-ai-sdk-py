from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from zhivex_ai import Agent, create_mock_language_model
from zhivex_ai.cli import build_parser, main
from zhivex_ai.messages import create_text_message
from zhivex_ai.types import GenerateResult


class ZhivexCliTests(unittest.TestCase):
    def test_parser_exposes_general_workflows(self) -> None:
        parser = build_parser()
        for command in ("inspect", "run", "eval", "serve", "playground"):
            with self.subTest(command=command):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args([command, "--help"])
                self.assertEqual(raised.exception.code, 0)

    def test_invalid_agent_reference_returns_usage_error(self) -> None:
        self.assertEqual(main(["inspect", "missing-reference"]), 2)

    def test_playground_rejects_non_loopback_hosts(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            status = main(["playground", "my_app:agent", "--host", "0.0.0.0"])

        self.assertEqual(status, 2)
        self.assertIn("restricted to a loopback host", stderr.getvalue())

    def test_run_command_loads_trusted_agent_and_prints_result(self) -> None:
        agent: Agent[Any, Any] = Agent(
            name="assistant",
            model=create_mock_language_model(
                responses=[
                    GenerateResult(
                        text="hello",
                        message=create_text_message("assistant", "hello"),
                        finish_reason="stop",
                    )
                ]
            ),
        )
        output = StringIO()
        with patch("zhivex_ai.cli.importlib.import_module", return_value=SimpleNamespace(agent=agent)):
            with redirect_stdout(output):
                status = main(["run", "my_app:agent", "--prompt", "hi"])

        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "hello\n")

    def test_eval_writes_atomic_json_and_junit_artifacts_with_repetitions(self) -> None:
        agent: Agent[Any, Any] = Agent(
            name="assistant",
            model=create_mock_language_model(
                responses=[
                    GenerateResult(
                        text="hello",
                        message=create_text_message("assistant", "hello"),
                        finish_reason="stop",
                    )
                    for _ in range(2)
                ]
            ),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            json_output = root / "artifacts" / "evaluation.json"
            junit_output = root / "artifacts" / "evaluation.xml"
            dataset.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "greeting",
                                "prompt": "Say hello",
                                "expectations": {"output_equals": "hello"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stdout = StringIO()
            with patch("zhivex_ai.cli.importlib.import_module", return_value=SimpleNamespace(agent=agent)):
                with redirect_stdout(stdout):
                    status = main(
                        [
                            "eval",
                            "my_app:agent",
                            "--dataset",
                            str(dataset),
                            "--repetitions",
                            "2",
                            "--max-concurrency",
                            "2",
                            "--output-json",
                            str(json_output),
                            "--output-junit",
                            str(junit_output),
                            "--min-pass-rate",
                            "1",
                            "--max-mean-latency-ms",
                            "1000",
                        ]
                    )

            self.assertEqual(status, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["trial_total"], 2)
            self.assertEqual(json.loads(json_output.read_text("utf-8")), payload)
            junit = ET.fromstring(junit_output.read_text("utf-8"))
            self.assertEqual(junit.attrib["tests"], "2")
            self.assertFalse(list((root / "artifacts").glob("*.tmp")))

    def test_eval_gate_failure_returns_one_and_invalid_configuration_returns_two(self) -> None:
        agent: Agent[Any, Any] = Agent(
            name="assistant",
            model=create_mock_language_model(
                responses=[
                    GenerateResult(
                        text="wrong",
                        message=create_text_message("assistant", "wrong"),
                        finish_reason="stop",
                    )
                ]
            ),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps([{"name": "case", "expectations": {"output_equals": "expected"}}]),
                encoding="utf-8",
            )
            stdout = StringIO()
            with patch("zhivex_ai.cli.importlib.import_module", return_value=SimpleNamespace(agent=agent)):
                with redirect_stdout(stdout):
                    self.assertEqual(main(["eval", "my_app:agent", "--dataset", str(dataset)]), 1)
            self.assertFalse(json.loads(stdout.getvalue())["ok"])

            empty_dataset = root / "empty.json"
            empty_dataset.write_text("[]", encoding="utf-8")
            stderr = StringIO()
            with patch("zhivex_ai.cli.importlib.import_module", return_value=SimpleNamespace(agent=agent)):
                with redirect_stderr(stderr):
                    self.assertEqual(
                        main(["eval", "my_app:agent", "--dataset", str(empty_dataset)]),
                        2,
                    )
            self.assertIn("cannot be empty", stderr.getvalue())

            with patch("zhivex_ai.cli.importlib.import_module", return_value=SimpleNamespace(agent=agent)):
                with redirect_stderr(StringIO()):
                    self.assertEqual(
                        main(
                            [
                                "eval",
                                "my_app:agent",
                                "--dataset",
                                str(dataset),
                                "--repetitions",
                                "0",
                            ]
                        ),
                        2,
                    )
