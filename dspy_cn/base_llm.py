from __future__ import annotations

from typing import Any, Mapping

import dspy

from .model_config import ModelConfig


def _dspy_model_name(config: ModelConfig) -> str:
    prefixes = {
        "openai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama_chat",
        "local": "ollama_chat",
        "local_llama": "ollama_chat",
        "together": "together_ai",
        "groq": "groq",
    }
    prefix = prefixes.get(config.provider)
    if not prefix or config.model.startswith(f"{prefix}/"):
        return config.model
    return f"{prefix}/{config.model}"


def configure_dspy_lm(
    config: ModelConfig | Mapping[str, Any],
    *,
    adapter: Any | None = None,
    enforce_json: bool = False,
):
    if not isinstance(config, ModelConfig):
        config = ModelConfig.from_mapping(config)

    kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        **config.extra,
    }
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.api_key_env:
        kwargs["api_key"] = config.api_key(required=True)

    lm = dspy.LM(_dspy_model_name(config), **kwargs)
    settings: dict[str, Any] = {"lm": lm}
    if adapter is not None:
        settings["adapter"] = adapter
    if enforce_json:
        settings["enforce_json"] = True
    dspy.settings.configure(**settings)
    return lm
