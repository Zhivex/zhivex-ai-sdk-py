"""Vertex text smoke with Express API keys or explicit ADC.

python examples/integrations/vertex.py --model gemini-3.8-flash
python examples/integrations/vertex.py --adc --project YOUR_PROJECT --location global
python examples/integrations/vertex.py --adc --project YOUR_PROJECT --responses --model xai/grok-4.6
python examples/integrations/vertex.py --adc --project YOUR_PROJECT --normalized-responses --model xai/grok-4.6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available
from zhivex_ai import create_vertex, generate_text


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adc", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--location", default="global")
    parser.add_argument("--model")
    responses_mode = parser.add_mutually_exclusive_group()
    responses_mode.add_argument("--responses", action="store_true", help="Use the Beta native Responses endpoint with an accessible compatible model")
    responses_mode.add_argument("--normalized-responses", action="store_true", help="Normalize the Beta Responses route through generate_text")
    args = parser.parse_args()
    if (args.responses or args.normalized_responses) and not args.adc:
        parser.error("Native Responses requires --adc and a Google Cloud project.")
    model_id = args.model or ("xai/grok-4.6" if args.responses or args.normalized_responses else "gemini-3.8-flash")
    load_dotenv_if_available()
    credentials = None
    if args.adc:
        import google.auth

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
            quota_project_id=args.project,
        )
    vertex = create_vertex(
        credentials=credentials,
        project_id=args.project,
        location=args.location,
        express_mode=not args.adc,
    )
    if args.responses:
        async with asyncio.timeout(30):
            response = await vertex.native.model_garden().responses(
                {"model": model_id, "input": "Explain asynchronous programming in one sentence.", "max_output_tokens": 512, "store": False},
            )
        print(json.dumps(response, indent=2))
        return
    result = await generate_text(
        model=vertex.native.model_garden().responses_model(model_id) if args.normalized_responses else vertex(model_id),
        prompt="Explain asynchronous programming in one sentence.",
        max_tokens=512,
        timeout_ms=30000,
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
