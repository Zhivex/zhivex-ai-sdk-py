from __future__ import annotations

from importlib import import_module

__all__ = [
    "create_bedrock",
    "create_anthropic",
    "create_azure_openai",
    "create_deepseek",
    "create_gemini",
    "KIMI_OFFICIAL_TOOL_URIS",
    "KimiFormulaClient",
    "create_kimi",
    "kimi_formula_toolset",
    "create_meta",
    "meta_hosted_tool",
    "meta_tool_search_tool",
    "meta_web_search_tool",
    "create_ollama",
    "create_openai",
    "create_openrouter",
    "create_qwen",
    "create_vllm",
    "create_vertex",
]

_EXPORTS = {
    "create_bedrock": ".bedrock",
    "create_anthropic": ".anthropic",
    "create_azure_openai": ".azure_openai",
    "create_deepseek": ".deepseek",
    "create_gemini": ".gemini",
    "KIMI_OFFICIAL_TOOL_URIS": ".kimi",
    "KimiFormulaClient": ".kimi",
    "create_kimi": ".kimi",
    "kimi_formula_toolset": ".kimi",
    "create_meta": ".meta",
    "meta_hosted_tool": ".meta",
    "meta_tool_search_tool": ".meta",
    "meta_web_search_tool": ".meta",
    "create_ollama": ".ollama",
    "create_openai": ".openai",
    "create_openrouter": ".openrouter",
    "create_qwen": ".qwen",
    "create_vllm": ".vllm",
    "create_vertex": ".vertex",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
