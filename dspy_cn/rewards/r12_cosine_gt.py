"""
Author: Preethi Gajawada
Description: Ground-truth alignment reward using cosine similarity.
"""

# dspy_cn/rewards/r12_cosine_gt.py
# R12: Alignment with Ground Truth via Cosine Similarity — maps to CCNC
# Model: sentence-transformers/all-MiniLM-L6-v2
from sentence_transformers import SentenceTransformer, util
from .base import RewardInput


class AlignWithGTCosine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, inp: RewardInput) -> float:
        if not inp.ground_truth:
            return 0.0
        a = self.model.encode(inp.counter_narrative, convert_to_tensor=True)
        b = self.model.encode(inp.ground_truth, convert_to_tensor=True)
        return float(util.cos_sim(a, b).item())
