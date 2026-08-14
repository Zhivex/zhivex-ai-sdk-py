from __future__ import annotations

import asyncio
import tempfile
import sys
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
    create_sqlite_agent_memory_store,
    create_sqlite_checkpoint_store,
    create_text_message,
    resume_agent,
    run_agent,
)


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "agent.sqlite3")
        agent = Agent(
            name="assistant",
            model=create_mock_language_model(
                responses=[
                    GenerateResult(text="remembered", messages=[create_text_message("assistant", "remembered")]),
                    GenerateResult(text="Apollo", messages=[create_text_message("assistant", "Apollo")]),
                ]
            ),
            memory=create_sqlite_agent_memory_store(path),
            checkpoint_store=create_sqlite_checkpoint_store(path),
        )
        session = create_agent_session()
        await run_agent(agent=agent, session=session, prompt="Remember Apollo.")
        resumed = await resume_agent(agent=agent, session_id=session.id, prompt="What did I ask you to remember?")
        print(resumed.text)
        print(resumed.resumed_from_checkpoint is not None)


if __name__ == "__main__":
    asyncio.run(main())
