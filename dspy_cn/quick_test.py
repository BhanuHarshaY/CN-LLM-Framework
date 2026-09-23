from .base_llm import configure_dspy_lm
from .dspy_program import EnglishCNProgram
from .model_config import (
    ModelConfig,
    load_config,
)
from .standalone_evaluator import (
    StandaloneCounterNarrativeEvaluator,
)


def main() -> None:
    print("[1/4] Loading configuration")

    config = load_config()

    print("[2/4] Configuring DSPy generation model")

    configure_dspy_lm(
        config["base_llm"]
    )

    print("[3/4] Testing counter-narrative generation")

    program = EnglishCNProgram()

    hate_speech = (
        "People from that community should not "
        "have the same rights as everyone else."
    )

    prediction = program(
        hate_speech=hate_speech
    )

    counter_narrative = str(
        prediction.counter_narrative
    ).strip()

    print("\nGenerated counter-narrative:")
    print(counter_narrative)

    print("\n[4/4] Testing reward evaluator")

    judge_config = ModelConfig.from_mapping(
        config["judge_llm"]
    )

    evaluator = (
        StandaloneCounterNarrativeEvaluator(
            judge_config=judge_config
        )
    )

    reward_scores = (
        evaluator.score_rewards_only(
            counter_narrative
        )
    )

    print("\nReward scores:")

    for metric, score in reward_scores.items():
        print(
            f"{metric}: "
            f"{score:.4f}"
        )

    print("\n[OK] Quick test completed")


if __name__ == "__main__":
    main()