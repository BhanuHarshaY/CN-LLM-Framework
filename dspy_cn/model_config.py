from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_key_env: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 300
    timeout: float = 120.0
    keep_alive: str = "30m"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelConfig":
        if not data:
            raise ValueError("Model configuration cannot be empty.")

        provider = str(data.get("provider", "")).strip().lower()
        model = str(data.get("model", "")).strip()
        if not provider:
            raise ValueError("Model configuration requires 'provider'.")
        if not model:
            raise ValueError("Model configuration requires 'model'.")

        api_base = data.get("api_base")
        if provider == "openai_compatible" and not api_base:
            raise ValueError("Provider 'openai_compatible' requires 'api_base'.")

        known = {
            "provider", "model", "api_key_env", "api_base", "temperature",
            "max_tokens", "timeout", "keep_alive", "extra",
        }
        extra = dict(data.get("extra") or {})
        extra.update({key: value for key, value in data.items() if key not in known})

        return cls(
            provider=provider,
            model=model,
            api_key_env=data.get("api_key_env"),
            api_base=api_base,
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 300)),
            timeout=float(data.get("timeout", 120.0)),
            keep_alive=str(data.get("keep_alive", "30m")),
            extra=extra,
        )

    def api_key(self, required: bool = True) -> Optional[str]:
        if not self.api_key_env:
            if required:
                raise EnvironmentError(
                    f"Provider '{self.provider}' requires 'api_key_env' in config.yaml."
                )
            return None

        value = os.getenv(self.api_key_env)
        if not value and required:
            raise EnvironmentError(
                f"Environment variable '{self.api_key_env}' is required "
                f"for provider '{self.provider}'."
            )
        return value


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_model_config(section: str, path: str | Path | None = None) -> ModelConfig:
    config = load_config(path)
    if section not in config:
        raise KeyError(f"Missing configuration section: {section}")
    return ModelConfig.from_mapping(config[section])


def resolve_package_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PACKAGE_DIR / path
