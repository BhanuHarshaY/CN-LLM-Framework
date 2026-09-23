"""
Independent non-confrontational tone reward.

Returns a score from 0.0 to 1.0.
This score is reported separately as NON_CONFRONTATIONAL.
It is not merged into PRS or EMP.
"""

import re

from transformers import pipeline

from .base import RewardInput


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


class NonConfrontationalTone:
    """
    Penalizes anger, disgust, threats, ridicule, and escalation.
    """

    ESCALATION_PATTERNS = (
        r"\bshut up\b",
        r"\bfight\b",
        r"\battack\b",
        r"\bdestroy\b",
        r"\bhate you\b",
        r"\byou deserve\b",
        r"\bmake you pay\b",
        r"\bget lost\b",
        r"\bgo away\b",
        r"\bhow dare you\b",
    )

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    ):
        self.classifier = pipeline(
            task="text-classification",
            model=model_name,
            top_k=None,
        )

    def score(self, inp: RewardInput) -> float:
        text = str(inp.counter_narrative).strip()

        if not text:
            return 0.0

        predictions = self.classifier(text)

        if predictions and isinstance(predictions[0], list):
            predictions = predictions[0]

        emotion_scores = {
            str(item["label"]).lower(): float(item["score"])
            for item in predictions
        }

        hostile_emotion = (
            emotion_scores.get("anger", 0.0)
            + emotion_scores.get("disgust", 0.0)
        )

        normalized_text = text.lower()
        escalation_hits = sum(
            bool(re.search(pattern, normalized_text))
            for pattern in self.ESCALATION_PATTERNS
        )
        escalation_penalty = min(1.0, escalation_hits * 0.35)

        score = 1.0 - (0.70 * hostile_emotion) - (0.30 * escalation_penalty)

        return _clamp01(score)
