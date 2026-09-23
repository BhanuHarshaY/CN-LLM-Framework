"""
Shared utilities for reference-free semantic reward functions.

The zero-shot classifier is loaded once and reused across rewards.
The default model supports multilingual NLI-style classification.
"""

from functools import lru_cache
from typing import Iterable, List

from transformers import pipeline


DEFAULT_ZERO_SHOT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@lru_cache(maxsize=2)
def get_zero_shot_classifier(
    model_name: str = DEFAULT_ZERO_SHOT_MODEL,
):
    return pipeline(
        task="zero-shot-classification",
        model=model_name,
    )


def zero_shot_positive_score(
    text: str,
    positive_labels: Iterable[str],
    negative_labels: Iterable[str],
    model_name: str = DEFAULT_ZERO_SHOT_MODEL,
    hypothesis_template: str = "This text is {}.",
) -> float:
    """
    Returns a normalized score in [0, 1].

    Positive and negative labels are evaluated together. The score is:
        positive_probability /
        (positive_probability + negative_probability)
    """

    text = str(text).strip()

    if not text:
        return 0.0

    positive_labels = list(positive_labels)
    negative_labels = list(negative_labels)
    labels: List[str] = positive_labels + negative_labels

    classifier = get_zero_shot_classifier(model_name)

    result = classifier(
        text,
        candidate_labels=labels,
        hypothesis_template=hypothesis_template,
        multi_label=True,
    )

    score_map = {
        str(label): float(score)
        for label, score in zip(
            result["labels"],
            result["scores"],
        )
    }

    positive_score = sum(
        score_map.get(label, 0.0)
        for label in positive_labels
    ) / max(1, len(positive_labels))

    negative_score = sum(
        score_map.get(label, 0.0)
        for label in negative_labels
    ) / max(1, len(negative_labels))

    denominator = positive_score + negative_score

    if denominator <= 0:
        return 0.0

    return clamp01(positive_score / denominator)
