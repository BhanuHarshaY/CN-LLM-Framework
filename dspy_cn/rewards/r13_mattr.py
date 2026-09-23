import re

from .base import RewardInput


class MATTRReward:
    """
    Moving-Average Type-Token Ratio.

    More stable than simple type-token ratio for different text lengths.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)

    def score(self, inp: RewardInput) -> float:
        tokens = self._tokenize(inp.counter_narrative)

        if not tokens:
            return 0.0

        if len(tokens) <= self.window_size:
            return float(len(set(tokens)) / len(tokens))

        ratios = []

        for start in range(len(tokens) - self.window_size + 1):
            window = tokens[start:start + self.window_size]
            ratios.append(len(set(window)) / self.window_size)

        return float(sum(ratios) / len(ratios))