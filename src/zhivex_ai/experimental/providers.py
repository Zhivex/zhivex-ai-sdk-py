"""Experimental provider factories and provider-native execution helpers.

All objects are compatibility re-exports of the existing public contracts.
"""

from __future__ import annotations

from ..providers.bedrock import create_bedrock
from ..providers.ollama import create_ollama
from ..providers.openai import openai_local_shell_tool, openai_shell_environment, openai_shell_tool
from ..providers.openai_compatible import create_groq_compatible, create_mistral_compatible, openai_compatible
from ..providers.openrouter import create_openrouter
from ..providers.xai import (
    create_xai,
    xai_code_execution_tool,
    xai_file_search_tool,
    xai_web_search_tool,
    xai_x_search_tool,
)
from ..providers.zai import create_zai

__all__ = [
    "create_bedrock",
    "create_ollama",
    "create_openrouter",
    "create_xai",
    "create_zai",
    "openai_compatible",
    "create_mistral_compatible",
    "create_groq_compatible",
    "openai_local_shell_tool",
    "openai_shell_environment",
    "openai_shell_tool",
    "xai_code_execution_tool",
    "xai_file_search_tool",
    "xai_web_search_tool",
    "xai_x_search_tool",
]
