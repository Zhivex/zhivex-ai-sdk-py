"""Verify receipt of a synthetic trace by both local OTLP backends."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.request import urlopen


def verify(trace_id: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{32}", trace_id):
        raise ValueError("Expected the trace ID emitted by this recipe run")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            with urlopen('http://127.0.0.1:16686/api/traces/' + trace_id, timeout=3) as response:
                traces = json.load(response)['data']
            if traces:
                trace = traces[0]
                names = {span['operationName'] for span in trace['spans']}
                required = {'application.request', 'zhivex.generation', 'zhivex.gateway.attempt', 'zhivex.agent.run'}
                if not required <= names:
                    raise ValueError('incomplete trace')
                with urlopen('http://127.0.0.1:13200/api/traces/' + trace['traceID'], timeout=3) as response:
                    if response.status == 200:
                        return {'schema_version': 1, 'status': 'passed', 'backends': ['jaeger', 'tempo'],
                                'span_count': len(trace['spans']), 'operations': sorted(required)}
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(1)
    return {'schema_version': 1, 'status': 'failed', 'reason': 'trace_not_received_by_both_backends'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.receipt.read_text())['trace_id'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
