from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    create_agent_run_snapshot,
    create_agent_trace_artifact,
    create_in_memory_agent_run_store,
    create_mock_language_model,
    replay_agent_run,
    run_agent,
    summarize_agent_trace,
)


async def main() -> None:
    store = create_in_memory_agent_run_store()
    agent = Agent(name="assistant", model=create_mock_language_model(), run_store=store)
    result = await run_agent(agent=agent, prompt="Draft an update", idempotency_key="update-1")
    state = await store.load(result.run_id)
    if state is None:
        raise RuntimeError("run state was not persisted")

    print(create_agent_run_snapshot(state).output_text)
    print(replay_agent_run(state).timeline[0].type)
    print(create_agent_trace_artifact(state).status)
    print(summarize_agent_trace(state).steps)


if __name__ == "__main__":
    asyncio.run(main())
