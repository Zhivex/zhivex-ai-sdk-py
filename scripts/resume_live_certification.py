"""Resume a synthetic live approval in a fresh process using only durable state."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import traceback
from pathlib import Path

from zhivex_ai import Agent, RunLimits, create_gemini, create_openai, create_sqlite_agent_run_store, resume_agent_run


if __package__:
    from .live_certification_tools import recovery_tool
else:
    from live_certification_tools import recovery_tool


async def resume(*, database: str, effects: str, provider: str, model: str, approved: bool):
    store = create_sqlite_agent_run_store(database)
    state = await store.find_by_idempotency_key("probe")
    if state is None:
        return {"status": "failed", "reason": "missing_state"}

    factory = create_openai if provider == "openai" else create_gemini
    agent = Agent(
        name="live-certification", model=factory()(model), run_store=store,
        instructions="Answer briefly from the tool result. If denied, acknowledge the denial. Do not request tools again.",
        tools={"lookup": recovery_tool(effects)},
        run_limits=RunLimits(max_steps=2, max_tool_calls=1, max_wall_time_ms=30_000),
    )
    try:
        async with asyncio.timeout(35):
            result = await resume_agent_run(agent=agent, run_id=state.run_id, approved=approved,
                                            tool_choice="none", max_tokens=256, max_retries=0, timeout_ms=25_000)
        return {"status": "passed", "output_nonempty": bool(result.text), "durable_status": result.state.status}
    except Exception as error:
        body = getattr(error, "response_body", None) or ""
        try:
            provider_error = json.loads(body).get("error", {})
        except (ValueError, AttributeError):
            provider_error = {}
        codes = {key: value for key, value in provider_error.items()
                 if key in {"code", "param", "type", "status"} and isinstance(value, (str, int))
                 and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,100}", str(value))}
        codes["http_status"] = getattr(error, "status", None)
        codes["hints"] = [hint for hint in ("thought_signature", "thought signature", "model", "function_call", "call_id", "tool", "max_output_tokens") if hint in body.lower()]
        # A losing concurrent claimant is reported separately from successful recovery.
        return {"status": "failed", "error_type": type(error).__name__, "diagnostics": codes,
                "failure_frames": [f"{Path(frame.filename).name}:{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(error.__traceback__)[-4:]]}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--effects", required=True)
    parser.add_argument("--provider", required=True, choices=["openai", "gemini"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--approved", choices=["yes", "no"], required=True)
    args = parser.parse_args()
    print(json.dumps(await resume(database=args.database, effects=args.effects, provider=args.provider,
                                  model=args.model, approved=args.approved == "yes")))


if __name__ == "__main__":
    asyncio.run(main())
