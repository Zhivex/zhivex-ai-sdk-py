from __future__ import annotations

import asyncio
from dataclasses import dataclass
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    GenerateResult,
    create_agent_session,
    create_mock_language_model,
    create_sqlite_agent_run_store,
    create_sqlite_checkpoint_store,
    create_text_message,
    replay_agent_run,
    resume_agent,
    run_agent,
)


@dataclass(frozen=True, slots=True)
class WorkerSummary:
    first_run_id: str
    reused_run_id: str
    resumed_text: str
    replay_status: str


async def run_worker_resume_demo(path: str) -> WorkerSummary:
    run_store = create_sqlite_agent_run_store(path, namespace="production-worker")
    checkpoint_store = create_sqlite_checkpoint_store(path, namespace="production-worker")
    agent = Agent(
        name="worker_assistant",
        model=create_mock_language_model(
            responses=[
                GenerateResult(text="queued job accepted", messages=[create_text_message("assistant", "queued job accepted")]),
                GenerateResult(text="resumed job completed", messages=[create_text_message("assistant", "resumed job completed")]),
            ]
        ),
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        metadata={"worker": "production-worker"},
    )
    session = create_agent_session()
    first = await run_agent(
        agent=agent,
        session=session,
        prompt="Process queued job.",
        idempotency_key="job-123",
    )
    reused = await run_agent(
        agent=agent,
        session=session,
        prompt="Duplicate delivery should reuse the stored run.",
        idempotency_key="job-123",
    )
    resumed = await resume_agent(agent=agent, session_id=session.id, prompt="Continue from the checkpoint.")
    replay = replay_agent_run(first.state)
    return WorkerSummary(
        first_run_id=first.run_id,
        reused_run_id=reused.run_id,
        resumed_text=resumed.text,
        replay_status=replay.snapshot.status,
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        summary = await run_worker_resume_demo(str(Path(directory) / "worker.sqlite3"))
        print(summary)


if __name__ == "__main__":
    asyncio.run(main())
