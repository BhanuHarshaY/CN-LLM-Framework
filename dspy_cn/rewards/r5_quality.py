"""
QS: Standalone overall quality reward.

Returns a separate QS score from 0.0 to 1.0.
No hate speech or ground truth is required.
"""

from .base import RewardInput
from .semantic_reward_utils import zero_shot_positive_score


class QualityReward:
    """
    Evaluates whether the counter-narrative is clear, meaningful,
    coherent, well-written, and useful as a standalone response.
    """

    POSITIVE_LABELS = (
        "clear and well written",
        "meaningful and useful",
        "coherent and complete",
    )

    NEGATIVE_LABELS = (
        "unclear and poorly written",
        "empty or meaningless",
        "incoherent or incomplete",
    )

    def score(self, inp: RewardInput) -> float:
        return zero_shot_positive_score(
            text=inp.counter_narrative,
            positive_labels=self.POSITIVE_LABELS,
            negative_labels=self.NEGATIVE_LABELS,
        )