"""Characterization captured from main 67f56fc before HU11 extraction."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from zhivex_ai.providers import _openai_responses_normalization as normalization

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/openai_responses_normalization.json').read_text())


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['name'])
def test_recorded_response_shapes(case):
    payload = deepcopy(case['payload'])
    assert asdict(normalization._parse_responses_message(payload, case['provider'])) == case['message']
    usage = normalization._parse_responses_usage(payload)
    assert (asdict(usage) if usage is not None else None) == case['usage']
    assert list(normalization._parse_response_finish_reason(payload)) == case['finish']
    assert payload == case['payload']
