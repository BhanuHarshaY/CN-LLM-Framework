from functools import lru_cache

import torch
from transformers import pipeline


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@lru_cache(maxsize=1)
def get_emotion_classifier():
    return pipeline(
        task="text-classification",
        model=(
            "j-hartmann/"
            "emotion-english-distilroberta-base"
        ),
        top_k=None,
        device=get_device(),
    )


def analyze_emotions(
    text: str,
) -> dict[str, float]:
    text = str(text).strip()

    if not text:
        return {}

    classifier = get_emotion_classifier()

    with torch.inference_mode():
        predictions = classifier(text)

    if (
        predictions
        and isinstance(predictions[0], list)
    ):
        predictions = predictions[0]

    return {
        str(item["label"]).lower(): float(
            item["score"]
        )
        for item in predictions
    }


def calculate_empathy(
    emotions: dict[str, float],
) -> float:
    positive_signal = (
        emotions.get("joy", 0.0)
        + 0.50 * emotions.get("sadness", 0.0)
        + 0.25 * emotions.get("neutral", 0.0)
    )

    hostile_signal = (
        emotions.get("anger", 0.0)
        + emotions.get("disgust", 0.0)
    )

    score = (
        0.50
        + positive_signal
        - hostile_signal
    )

    return float(
        max(0.0, min(1.0, score))
    )


def calculate_non_confrontational(
    emotions: dict[str, float],
) -> float:
    hostile_signal = (
        emotions.get("anger", 0.0)
        + emotions.get("disgust", 0.0)
    )

    return float(
        max(
            0.0,
            min(1.0, 1.0 - hostile_signal),
        )
    )