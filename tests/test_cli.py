from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
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

    def test_run_command_loads_trusted_agent_and_prints_result(self) -> None:
        agent = Agent(
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
