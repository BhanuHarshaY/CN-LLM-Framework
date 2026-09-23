"""
Author: Preethi Gajawada
Description: Safety and non-toxicity reward using Detoxify.
"""

# dspy_cn/rewards/r1_safety.py
# R1: Safety & Non-Toxicity — maps to TS (Toxicity Score)
# Model: Detoxify 'original' variant
from detoxify import Detoxify
from .base import RewardInput


class SafetyNonToxicity:
    def __init__(self):
        self.model = Detoxify("original")

    def score(self, inp: RewardInput) -> float:
        res = self.model.predict(inp.counter_narrative)
        tox = max(
            res.get("toxicity", 0),
            res.get("severe_toxicity", 0),
            res.get("insult", 0),
        )
        return float(1.0 - tox)
