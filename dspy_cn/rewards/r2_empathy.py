"""
Independent empathy reward.

Returns a score from 0.0 to 1.0.
This score is reported as EMP and is not included inside PRS.
"""

from transformers import pipeline

from .base import RewardInput


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


class EmpathyReward:
    """
    Estimates empathetic emotional framing.

    Note:
    This remains an emotion-based proxy. It should be validated against
    human empathy annotations before being treated as a definitive metric.
    """

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

        positive_signal = (
            emotion_scores.get("joy", 0.0)
            + 0.50 * emotion_scores.get("sadness", 0.0)
            + 0.25 * emotion_scores.get("neutral", 0.0)
        )

        hostile_signal = (
            emotion_scores.get("anger", 0.0)
            + emotion_scores.get("disgust", 0.0)
        )

        empathy_score = 0.50 + positive_signal - hostile_signal

        return _clamp01(empathy_score)
