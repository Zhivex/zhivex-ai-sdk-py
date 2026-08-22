"""Experimental provider factories and provider-native execution helpers.

All objects are compatibility re-exports of the existing public contracts.
"""

from __future__ import annotations

from ..providers.bedrock import create_bedrock
from ..providers.ollama import create_ollama
from ..providers.openai import openai_local_shell_tool, openai_shell_environment, openai_shell_tool
from ..providers.openrouter import create_openrouter

__all__ = [
    "create_bedrock",
    "create_ollama",
    "create_openrouter",
    "openai_local_shell_tool",
    "openai_shell_environment",
    "openai_shell_tool",
]
