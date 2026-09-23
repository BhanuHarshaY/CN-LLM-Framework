"""
FLU: Reference-free fluency reward.

Returns a separate FLU score from 0.0 to 1.0.
Uses a multilingual zero-shot NLI model.
"""

import re

from .base import RewardInput
from .semantic_reward_utils import (
    clamp01,
    zero_shot_positive_score,
)


class FluencyReward:
    """
    Evaluates grammaticality, naturalness, readability,
    and sentence completeness.
    """

    POSITIVE_LABELS = (
        "fluent and natural",
        "grammatically correct",
        "easy to read",
    )

    NEGATIVE_LABELS = (
        "disfluent or unnatural",
        "grammatically incorrect",
        "difficult to read",
    )

    def score(self, inp: RewardInput) -> float:
        text = str(inp.counter_narrative).strip()

        if not text:
            return 0.0

        semantic_score = zero_shot_positive_score(
            text=text,
            positive_labels=self.POSITIVE_LABELS,
            negative_labels=self.NEGATIVE_LABELS,
        )

        malformed_spacing = bool(
            re.search(r"\s{3,}", text)
        )
        repeated_punctuation = bool(
            re.search(r"([!?.,])\1{2,}", text)
        )
        incomplete_ending = not bool(
            re.search(r"[.!?।]\s*$", text)
        )

        mechanical_penalty = (
            0.20 * malformed_spacing
            + 0.20 * repeated_punctuation
            + 0.10 * incomplete_ending
        )

        return clamp01(
            semantic_score - mechanical_penalty
        )
