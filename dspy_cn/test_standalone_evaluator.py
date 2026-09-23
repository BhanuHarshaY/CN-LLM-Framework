import json

from .model_config import ModelConfig, load_config
from .standalone_evaluator import evaluate_counter_narrative

config = load_config()
judge_config = ModelConfig.from_mapping(config["judge_llm"])

counter_narrative = (
    "Everyone deserves to be treated with dignity and respect. "
    "A person's identity does not make them less valuable, and "
    "respectful dialogue can help us understand one another."
)

result = evaluate_counter_narrative(
    counter_narrative,
    judge_config=judge_config,
)

print(json.dumps(result, indent=2))