"""
Independent politeness and respectfulness reward.

Returns a score from 0.0 to 1.0.
This score is reported directly as PRS.
"""

import re

from .base import RewardInput


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


class PolitenessRespectReward:
    """
    Reference-free politeness reward.

    It rewards respectful wording and penalizes insults, commands,
    dismissiveness, ridicule, and aggressive second-person framing.
    """

    RESPECTFUL_PATTERNS = (
        r"\bplease\b",
        r"\brespect\b",
        r"\brespectful\b",
        r"\bdignity\b",
        r"\bunderstand\b",
        r"\bconsider\b",
        r"\bkindness\b",
        r"\bcompassion\b",
        r"\bdeserve(?:s|d)?\b",
        r"\bappreciate\b",
        r"\bmay want to\b",
        r"\bcould\b",
    )

    IMPOLITE_PATTERNS = (
        r"\bshut up\b",
        r"\bstupid\b",
        r"\bidiot\b",
        r"\bmoron\b",
        r"\bdumb\b",
        r"\bpathetic\b",
        r"\bworthless\b",
        r"\bdisgusting\b",
        r"\bignorant\b",
        r"\bgo away\b",
        r"\bno one cares\b",
        r"\byou people\b",
    )

    AGGRESSIVE_OPENINGS = (
        r"^\s*you are\b",
        r"^\s*you're\b",
        r"^\s*stop\b",
        r"^\s*never\b",
        r"^\s*obviously\b",
    )

    def score(self, inp: RewardInput) -> float:
        text = str(inp.counter_narrative).strip().lower()

        if not text:
            return 0.0

        respectful_hits = sum(
            bool(re.search(pattern, text))
            for pattern in self.RESPECTFUL_PATTERNS
        )

        impolite_hits = sum(
            bool(re.search(pattern, text))
            for pattern in self.IMPOLITE_PATTERNS
        )

        aggressive_hits = sum(
            bool(re.search(pattern, text))
            for pattern in self.AGGRESSIVE_OPENINGS
        )

        # A neutral, non-hostile response starts at 0.65.
        score = 0.65
        score += min(0.30, respectful_hits * 0.10)
        score -= min(0.80, impolite_hits * 0.30)
        score -= min(0.30, aggressive_hits * 0.15)

        return _clamp01(score)
