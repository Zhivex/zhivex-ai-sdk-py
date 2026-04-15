import asyncio
from pathlib import Path

from zhivex_ai import (
    Agent,
    create_openai,
    create_sqlite_agent_memory_store,
    create_sqlite_checkpoint_store,
    create_agent_session,
    resume_agent,
    run_agent,
)


async def main() -> None:
    db_path = str(Path(__file__).with_name(".tmp-agent-state.sqlite3").resolve())
    openai = create_openai()
    agent = Agent(
        name="assistant",
        instructions="Remember prior turns.",
        model=openai("gpt-5.4-mini"),
        memory=create_sqlite_agent_memory_store(db_path),
        checkpoint_store=create_sqlite_checkpoint_store(db_path),
    )

    session = create_agent_session()
    await run_agent(agent=agent, session=session, prompt="Remember that project Apollo is important.")

    resumed = await resume_agent(
        agent=agent,
        session_id=session.id,
        prompt="What project did I mention?",
    )
    print(resumed.text)
    print(resumed.session.metadata.get("resumed_from_checkpoint"))


if __name__ == "__main__":
    asyncio.run(main())
