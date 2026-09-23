"""
Author: Preethi Gajawada
Description: Distinct-2 metric measuring lexical diversity of generated counter-narratives.
"""

# dspy_cn/rewards/distinct2.py
# Distinct-2: Measures bigram diversity of generated counter-narratives.
# Used by shared task organizers as a reference-based evaluation metric.
# Higher = more diverse vocabulary usage.
from .base import RewardInput


class Distinct2Score:
    def score(self, inp: RewardInput) -> float:
        tokens = inp.counter_narrative.lower().split()
        if len(tokens) < 2:
            return 0.0
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        if not bigrams:
            return 0.0
        return len(set(bigrams)) / len(bigrams)

    def score_batch(self, counter_narratives: list) -> float:
        """Compute Distinct-2 across a batch of CNs (set-level diversity)."""
        all_bigrams = []
        for cn in counter_narratives:
            tokens = cn.lower().split()
            for i in range(len(tokens) - 1):
                all_bigrams.append((tokens[i], tokens[i + 1]))
        if not all_bigrams:
            return 0.0
        return len(set(all_bigrams)) / len(all_bigrams)
