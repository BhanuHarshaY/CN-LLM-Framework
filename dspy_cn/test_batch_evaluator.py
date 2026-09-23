import json

from .standalone_evaluator import (
    StandaloneCounterNarrativeEvaluator,
)


counter_narratives = [
    (
        "Everyone deserves to be treated with dignity and "
        "respect. A person's identity does not make them "
        "less valuable."
    ),
    (
        "Mocking people for who they are does not create a "
        "better conversation. Listening respectfully can help "
        "us understand experiences different from our own."
    ),
    (
        "People have different identities and experiences. "
        "Those differences should not be used to deny anyone "
        "equal respect."
    ),
    (
        "You are wrong and should just stop talking."
    ),
]


from .model_config import ModelConfig, load_config

config = load_config()
judge_config = ModelConfig.from_mapping(config["judge_llm"])

evaluator = StandaloneCounterNarrativeEvaluator(
    judge_config=judge_config,
)

results = evaluator.score_batch(
    counter_narratives
)

print(f"Returned {len(results)} results")

for index, result in enumerate(results):
    print("\n" + "=" * 60)
    print(f"ITEM {index}")
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )