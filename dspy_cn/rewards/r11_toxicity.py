from functools import lru_cache
from typing import Dict

from detoxify import Detoxify


@lru_cache(maxsize=1)
def get_detoxify_model():
    """
    Loads one shared Detoxify model for the entire process.
    """

    return Detoxify("original")


def analyze_toxicity(
    text: str,
) -> Dict[str, float]:
    text = str(text).strip()

    if not text:
        return {}

    model = get_detoxify_model()
    result = model.predict(text)

    return {
        key: float(value)
        for key, value in result.items()
    }


def calculate_non_toxicity(
    toxicity_result: Dict[str, float],
) -> float:
    if not toxicity_result:
        return 0.0

    toxicity_risk = max(
        toxicity_result.get("toxicity", 0.0),
        toxicity_result.get(
            "severe_toxicity",
            0.0,
        ),
        toxicity_result.get("insult", 0.0),
        toxicity_result.get("threat", 0.0),
        toxicity_result.get(
            "identity_attack",
            0.0,
        ),
    )

    return float(
        max(
            0.0,
            min(1.0, 1.0 - toxicity_risk),
        )
    )