from unittest import TestCase

import pytest

pytest.importorskip("google.auth", reason="Vertex runner tests require the vertex extra")
pytest.importorskip("google.auth.transport.requests", reason="Vertex ADC requires requests")

from scripts.verify_vertex_mistral_integration import validate_fim, validate_ocr


class MistralNativeRunnerTests(TestCase):
    def test_fim_checks_completion_without_executing_code(self):
        validate_fim(" 58 ", "stop")
        for text, finish in (("57", "stop"), ("58", "length"), ("__import__('os').getcwd()", "stop"), ("58 or 1", "stop")):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                validate_fim(text, finish)

    def test_ocr_requires_actual_document_markers_in_pages(self):
        validate_ocr({"pages": [{"index": 0, "markdown": "ORCHID-7284 turquoise"}]})
        for response in ({}, {"pages": []}, {"pages": [{"markdown": "unrelated"}], "metadata": "ORCHID-7284 turquoise"}, {"pages": [{"markdown": 5}]}):
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                validate_ocr(response)
