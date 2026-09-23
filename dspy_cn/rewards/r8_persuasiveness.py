"""
PERS: Reference-free persuasive-quality reward.

Returns a separate PERS score from 0.0 to 1.0.

This measures standalone persuasive quality, not whether the response
successfully rebuts a particular hate-speech claim.
"""

from .base import RewardInput
from .semantic_reward_utils import zero_shot_positive_score


class PersuasivenessReward:
    """
    Evaluates whether the text gives a credible reason, invites
    reconsideration, and is likely to encourage reflection.
    """

    POSITIVE_LABELS = (
        "persuasive",
        "gives a credible reason",
        "encourages reconsideration",
    )

    NEGATIVE_LABELS = (
        "unpersuasive",
        "unsupported or merely assertive",
        "unlikely to encourage reflection",
    )

    def score(self, inp: RewardInput) -> float:
        return zero_shot_positive_score(
            text=inp.counter_narrative,
            positive_labels=self.POSITIVE_LABELS,
            negative_labels=self.NEGATIVE_LABELS,
        )
