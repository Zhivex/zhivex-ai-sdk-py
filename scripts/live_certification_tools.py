"""Identical tool definition reconstructed on either side of a process boundary."""
from functools import partial
from pathlib import Path

from zhivex_ai import ToolDefinition


def lookup(_input, *, effects):
    with Path(effects).open("a") as log:
        log.write("lookup\n")
    return {"marker": "ORBIT_SEVEN"}


def recovery_tool(effects):
    return ToolDefinition(
        name="lookup", description="Returns the marker required to answer the user.",
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        execute=partial(lookup, effects=effects), requires_approval=True,
    )
