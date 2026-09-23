"""
Author: Preethi Gajawada
Description: MNLI contradiction reward measuring how well the CN challenges the hate speech.
"""

# dspy_cn/rewards/r8_mnli_contradiction.py
from transformers import AutoTokenizer, pipeline
from .base import RewardInput


class MNLIContradiction:
    def __init__(self):
        self.nli = pipeline(
            "text-classification",
            model="roberta-large-mnli",
            top_k=None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

    def score(self, inp: RewardInput) -> float:
        hs = inp.hate_speech
        cn = inp.counter_narrative

        # Truncate to fit within 512 tokens
        combined = hs + "</s></s>" + cn
        tokens = self.tokenizer.encode(combined, add_special_tokens=False)
        if len(tokens) > 500:
            # Truncate CN side to fit
            hs_tokens = self.tokenizer.encode(hs, add_special_tokens=False)[:200]
            cn_tokens = self.tokenizer.encode(cn, add_special_tokens=False)[:200]
            hs = self.tokenizer.decode(hs_tokens, skip_special_tokens=True)
            cn = self.tokenizer.decode(cn_tokens, skip_special_tokens=True)

        text = f"{hs}</s></s>{cn}"

        try:
            preds = self.nli(text, truncation=True, max_length=512)
            if preds and isinstance(preds[0], list):
                preds = preds[0]

            label_map = {p["label"].upper(): p["score"] for p in preds}
            contradiction_prob = label_map.get("CONTRADICTION", 0.0)
            entailment_prob = label_map.get("ENTAILMENT", 0.0)

            non_entailment = 1.0 - entailment_prob
            return float(0.6 * non_entailment + 0.4 * contradiction_prob)
        except Exception as e:
            print(f"[WARN] R8 error: {e}")
            return 0.5
