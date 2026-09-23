"""
CON: Reference-free constructiveness reward.

Returns a separate CON score from 0.0 to 1.0.
"""

from .base import RewardInput
from .semantic_reward_utils import zero_shot_positive_score


class ConstructivenessReward:
    """
    Evaluates whether the response encourages reflection,
    respectful discussion, understanding, or a positive alternative.
    """

    POSITIVE_LABELS = (
        "constructive",
        "encourages respectful reflection",
        "supports positive dialogue",
    )

    NEGATIVE_LABELS = (
        "unconstructive",
        "dismissive without offering reflection",
        "likely to worsen the conversation",
    )

    def score(self, inp: RewardInput) -> float:
        return zero_shot_positive_score(
            text=inp.counter_narrative,
            positive_labels=self.POSITIVE_LABELS,
            negative_labels=self.NEGATIVE_LABELS,
        )
