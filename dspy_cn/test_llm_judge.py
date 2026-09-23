from .llm_judge import score_counter_narrative
from .model_config import ModelConfig, load_config

config = load_config()
judge_config = ModelConfig.from_mapping(config["judge_llm"])

cn = (
    "Everyone deserves dignity and respect. "
    "Differences in identity do not make anyone less valuable."
)

score = score_counter_narrative(
    cn,
    config=judge_config,
)

print(score)