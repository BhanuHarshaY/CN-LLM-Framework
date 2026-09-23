# Pluggable LLM Framework

The generation model and evaluation judge are selected only through `config.yaml`.
No evaluator, metric, batching, or CSV code needs to change when switching models.

## Run from the directory containing `dspy_cn`

```bash
python -m dspy_cn.quick_test
python -m dspy_cn.evaluate_csv
python -m dspy_cn.generate_predictions
python -m dspy_cn.dspy_optimize
```

## Ollama

```yaml
judge_llm:
  provider: ollama
  model: llama3.1:8b
```

```bash
ollama pull llama3.1:8b
```

## OpenAI

```yaml
judge_llm:
  provider: openai
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
```

```bash
export OPENAI_API_KEY="..."
```

## Anthropic

```yaml
judge_llm:
  provider: anthropic
  model: claude-sonnet-4-5
  api_key_env: ANTHROPIC_API_KEY
```

## OpenAI-compatible servers

Use `openai_compatible`, `vllm`, `lmstudio`, `together`, or `groq` and provide `api_base` where required.

## Adding another provider

Create a class extending `BaseLLMClient`, decorate it with `@register_client("provider_name")`, and implement `generate_json`. The rest of the framework remains unchanged.
