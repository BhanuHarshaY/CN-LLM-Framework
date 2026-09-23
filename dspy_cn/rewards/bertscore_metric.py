"""
Author: Preethi Gajawada
Description: BERTScore semantic similarity metric for CN vs ground truth.
"""

# dspy_cn/rewards/bertscore_metric.py
# BERTScore: Token-level semantic similarity between generated CN and ground truth.
# Used by shared task organizers as a reference-based evaluation metric.
from bert_score import score as bert_score
from .base import RewardInput


class BERTScoreMetric:
    def score(self, inp: RewardInput) -> float:
        """Compute BERTScore F1 for a single CN vs ground truth."""
        if not inp.ground_truth:
            return 0.0
        _, _, f1 = bert_score(
            [inp.counter_narrative],
            [inp.ground_truth],
            lang="en",
            verbose=False,
        )
        return float(f1.mean().item())

    def score_batch(self, counter_narratives: list, ground_truths: list) -> dict:
        """Compute BERTScore for a batch. Returns per-item F1 and average."""
        if not ground_truths or not counter_narratives:
            return {"per_item": [], "avg_f1": 0.0}
        _, _, f1 = bert_score(
            counter_narratives,
            ground_truths,
            lang="en",
            verbose=False,
        )
        per_item = [float(x) for x in f1.tolist()]
        return {"per_item": per_item, "avg_f1": float(f1.mean().item())}
