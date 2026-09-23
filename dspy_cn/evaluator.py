"""
Author: Preethi Gajawada
Institution: Northeastern University
Project: Multilingual Counter-Narrative Generation using DSPy
Description: Reward-based evaluator computing PRS, CCNC, and QS metrics.
Year: 2026
"""

# dspy_cn/evaluator.py
# Evaluator combining subset reward functions into PRS, CCNC, QS
# Plus reference-based metrics: Distinct-2 and BERTScore

from typing import List, Optional, Dict, Any
from dspy_cn.rewards import (
    RewardInput,
    SafetyNonToxicity,
    EmpathyReward,
    InputOutputSemanticGrounding,
    NonConfrontationalTone,
    MNLIContradiction,
    AlignWithGTCosine,
    Distinct2Score,
    BERTScoreMetric,
)


# 🔥 -------- CHUNKING UTILITIES --------

def chunk_text(text: str, max_words: int = 200):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def chunked_score(fn, base_input):
    """Chunk only counter narrative (used for MNLI R8)."""
    cn_chunks = chunk_text(base_input.counter_narrative)

    scores = []
    for cn in cn_chunks:
        new_input = RewardInput(
            hate_speech=base_input.hate_speech,
            counter_narrative=cn,
            ground_truth=base_input.ground_truth,
        )
        scores.append(fn.score(new_input))

    return sum(scores) / len(scores) if scores else 0.0


def dual_chunk_score(fn, base_input):
    """Chunk both HS and CN (used for R3, R12)."""
    hs_chunks = chunk_text(base_input.hate_speech)
    cn_chunks = chunk_text(base_input.counter_narrative)

    scores = []
    for hs in hs_chunks:
        for cn in cn_chunks:
            new_input = RewardInput(
                hate_speech=hs,
                counter_narrative=cn,
                ground_truth=base_input.ground_truth,
            )
            scores.append(fn.score(new_input))

    return sum(scores) / len(scores) if scores else 0.0


# 🔥 -------- MAIN EVALUATOR --------

class RewardEvaluator:
    """Evaluates CNs using the subset reward functions mapped to PRS, CCNC, QS."""

    def __init__(self):
        print("[INFO] Loading reward models...")
        self.r1 = SafetyNonToxicity()
        self.r2 = EmpathyReward()
        self.r3 = InputOutputSemanticGrounding()
        self.r4 = NonConfrontationalTone()
        self.r8 = MNLIContradiction()
        self.r12 = AlignWithGTCosine()
        self.distinct2 = Distinct2Score()
        self.bertscore = BERTScoreMetric()
        print("[OK] All reward models loaded.")

    def score_single(self, hs: str, cn: str, gt: Optional[str] = None):

        hs_chunks = chunk_text(hs)
        cn_chunks = chunk_text(cn)

        scores = []

        for h, c in zip(hs_chunks, cn_chunks):
            inp = RewardInput(
                hate_speech=h,
                counter_narrative=c,
                ground_truth=gt,
            )

            try:
                r1 = self.r1.score(inp)
                r2 = self.r2.score(inp)
                r3 = self.r3.score(inp)
                r4 = self.r4.score(inp)
                r8 = self.r8.score(inp)
                r12 = self.r12.score(inp) if gt else 0.0

                scores.append((r1, r2, r3, r4, r8, r12))

            except Exception:
                continue

        # fallback
        if not scores:
            return {
                "R1": 0, "R2": 0, "R3": 0, "R4": 0,
                "R8": 0, "R12": 0,
                "PRS": 0, "CCNC": 0, "QS": 0,
                "Distinct2": 0,
                "combined": 0,
            }

        # ✅ aggregate (mean)
        r1 = sum(s[0] for s in scores) / len(scores)
        r2 = sum(s[1] for s in scores) / len(scores)
        r3 = sum(s[2] for s in scores) / len(scores)
        r4 = sum(s[3] for s in scores) / len(scores)
        r8 = sum(s[4] for s in scores) / len(scores)
        r12 = sum(s[5] for s in scores) / len(scores)

        # composites
        prs = 0.45 * r4 + 0.40 * r2 + 0.15 * r1
        ccnc = 0.45 * r3 + 0.35 * r8 + 0.20 * r12
        qs = (0.50 * r1) + (0.40 * r4) + (0.10 * r3)

        d2 = self.distinct2.score(
            RewardInput(hate_speech=hs, counter_narrative=cn, ground_truth=gt)
        )

        return {
            "R1": r1, "R2": r2, "R3": r3, "R4": r4,
            "R8": r8, "R12": r12,
            "PRS": prs,
            "CCNC": ccnc,
            "QS": qs,
            "Distinct2": d2,
            "combined": (prs + ccnc + qs) / 3,
        }
    def score_batch(self, hs_list: List[str], cn_list: List[str],
                    gt_list: Optional[List[str]] = None) -> Dict[str, Any]:

        n = len(cn_list)
        gt_list = gt_list or [None] * n

        all_scores = []
        for i in range(n):
            scores = self.score_single(hs_list[i], cn_list[i], gt_list[i])
            all_scores.append(scores)

        # Batch-level Distinct-2
        batch_d2 = self.distinct2.score_batch(cn_list)

        # BERTScore
        valid_gt = [g for g in gt_list if g is not None]
        if valid_gt and len(valid_gt) == n:
            bs_result = self.bertscore.score_batch(cn_list, gt_list)
            avg_bertscore = bs_result["avg_f1"]
            per_item_bertscore = bs_result["per_item"]
        else:
            avg_bertscore = 0.0
            per_item_bertscore = [0.0] * n

        for i, s in enumerate(all_scores):
            s["BERTScore"] = per_item_bertscore[i]

        # Averages
        keys = ["R1", "R2", "R3", "R4", "R8", "R12", "PRS", "CCNC", "QS",
                "Distinct2", "combined", "BERTScore"]

        avgs = {}
        for k in keys:
            vals = [s[k] for s in all_scores]
            avgs[f"avg_{k}"] = sum(vals) / len(vals) if vals else 0.0

        avgs["batch_Distinct2"] = batch_d2
        avgs["avg_BERTScore"] = avg_bertscore

        return {
            "per_item": all_scores,
            "averages": avgs,
            "n": n,
        }