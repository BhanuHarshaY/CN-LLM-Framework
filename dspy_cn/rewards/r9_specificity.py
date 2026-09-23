"""
SPEC: Reference-free specificity reward.

Returns a separate SPEC score from 0.0 to 1.0.

Without the original hate speech, this measures standalone concreteness
and non-genericness, not context-specific relevance.
"""

import re

from .base import RewardInput
from .semantic_reward_utils import (
    clamp01,
    zero_shot_positive_score,
)


class SpecificityReward:
    """
    Combines semantic concreteness with a lightweight genericness penalty.
    """

    GENERIC_PATTERNS = (
        r"^\s*everyone deserves dignity and respect[.!]?\s*$",
        r"^\s*we should respect everyone[.!]?\s*$",
        r"^\s*people should be kind to each other[.!]?\s*$",
        r"^\s*everyone has the right to be themselves[.!]?\s*$",
    )

    POSITIVE_LABELS = (
        "specific and concrete",
        "contains a meaningful reason or detail",
        "informative rather than generic",
    )

    NEGATIVE_LABELS = (
        "vague and generic",
        "empty or formulaic",
        "lacks meaningful detail",
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

        normalized = text.lower()

        exact_generic_match = any(
            re.match(pattern, normalized)
            for pattern in self.GENERIC_PATTERNS
        )

        word_count = len(text.split())

        if exact_generic_match:
            genericness_score = 0.0
        elif word_count < 8:
            genericness_score = word_count / 8.0
        else:
            genericness_score = 1.0

        return clamp01(
            0.75 * semantic_score
            + 0.25 * genericness_score
        )
