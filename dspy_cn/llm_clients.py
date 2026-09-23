from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Type

from .model_config import ModelConfig

SYSTEM_PROMPT = (
    "You are an impartial counter-narrative evaluation judge. "
    "Return valid JSON only."
)


def parse_json_object(content: str) -> Dict[str, Any]:
    cleaned = str(content).replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON:\n{cleaned}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON response must be an object.")
    return parsed


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def generate_batch_json(
        self,
        counter_narratives: List[str],
        rubric: str,
    ) -> List[Dict[str, Any]]:
        if not counter_narratives:
            return []

        items = [
            {"item_id": index, "counter_narrative": text}
            for index, text in enumerate(counter_narratives)
        ]
        prompt = f"""
Evaluate every counter-narrative independently.

{rubric}

Return ONLY valid JSON in this exact structure:
{{
  "results": [
    {{
      "item_id": 0,
      "PRS": 0.0,
      "QS": 0.0,
      "SAFE": 0.0,
      "EMP": 0.0,
      "CON": 0.0,
      "PERS": 0.0,
      "SPEC": 0.0,
      "FLU": 0.0,
      "TOX": 0.0
    }}
  ]
}}

Rules:
- Return exactly one result for every input item.
- Preserve each item_id.
- Every metric score must be a decimal number from 0.0 to 5.0.
- Use exactly one decimal place for every metric score.
- Intermediate scores such as 2.6, 3.4, and 4.7 are allowed.
- Use decimal precision when performance falls between rubric levels.
- Score every metric independently.
- Use 5.0 only for exceptional performance.
- Return no reasoning, Markdown, or text outside the JSON object.

INPUT ITEMS:
{json.dumps(items, ensure_ascii=False)}
""".strip()

        parsed = self.generate_json(prompt)
        results = parsed.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Batch response does not contain a results list.")

        expected_ids = set(range(len(counter_narratives)))
        returned: Dict[int, Dict[str, Any]] = {}
        for result in results:
            if not isinstance(result, dict):
                continue
            try:
                item_id = int(result["item_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if item_id in expected_ids and item_id not in returned:
                returned[item_id] = result

        if set(returned) != expected_ids:
            missing = sorted(expected_ids - set(returned))
            raise ValueError(f"Missing batch item IDs: {missing}")
        return [returned[item_id] for item_id in range(len(counter_narratives))]


_CLIENT_REGISTRY: Dict[str, Type[BaseLLMClient]] = {}


def register_client(*provider_names: str):
    def decorator(cls: Type[BaseLLMClient]) -> Type[BaseLLMClient]:
        for name in provider_names:
            _CLIENT_REGISTRY[name.lower().strip()] = cls
        return cls
    return decorator


@register_client("ollama", "local", "local_llama")
class OllamaLLMClient(BaseLLMClient):
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        try:
            import ollama
        except ImportError as exc:
            raise ImportError("Install Ollama support with: pip install ollama") from exc

        response = ollama.chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
            stream=False,
            keep_alive=self.config.keep_alive,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                **self.config.extra,
            },
        )
        return parse_json_object(response["message"]["content"])


@register_client("openai")
class OpenAILLMClient(BaseLLMClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install OpenAI support with: pip install openai") from exc
        self.client = OpenAI(
            api_key=config.api_key(required=True),
            base_url=config.api_base,
            timeout=config.timeout,
        )

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            **self.config.extra,
        )
        return parse_json_object(response.choices[0].message.content or "")


@register_client("openai_compatible", "vllm", "lmstudio", "together", "groq")
class OpenAICompatibleLLMClient(BaseLLMClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install OpenAI-compatible support with: pip install openai") from exc
        self.client = OpenAI(
            api_key=config.api_key(required=False) or "not-required",
            base_url=config.api_base,
            timeout=config.timeout,
        )

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra,
        )
        return parse_json_object(response.choices[0].message.content or "")


@register_client("anthropic", "claude")
class AnthropicLLMClient(BaseLLMClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("Install Anthropic support with: pip install anthropic") from exc
        self.client = Anthropic(
            api_key=config.api_key(required=True),
            base_url=config.api_base,
            timeout=config.timeout,
        )

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=self.config.model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra,
        )
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return parse_json_object(text)


def get_llm_client(config: ModelConfig | Mapping[str, Any]) -> BaseLLMClient:
    if not isinstance(config, ModelConfig):
        config = ModelConfig.from_mapping(config)
    client_cls = _CLIENT_REGISTRY.get(config.provider)
    if client_cls is None:
        supported = ", ".join(sorted(_CLIENT_REGISTRY))
        raise ValueError(
            f"Unsupported provider '{config.provider}'. Supported providers: {supported}"
        )
    return client_cls(config)


# Backward-compatible name for existing code.
def get_judge_client(config: ModelConfig | Mapping[str, Any]) -> BaseLLMClient:
    return get_llm_client(config)
