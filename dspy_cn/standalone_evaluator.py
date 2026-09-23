from typing import Any, Dict, List

from .llm_judge import CounterNarrativeJudge
from .model_config import ModelConfig
from .rewards.base import RewardInput
from .rewards.r3_politeness import PolitenessRespectReward
from .rewards.r5_quality import QualityReward
from .rewards.r6_safety import SafetyDeEscalationReward
from .rewards.r7_constructiveness import ConstructivenessReward
from .rewards.r8_persuasiveness import PersuasivenessReward
from .rewards.r9_specificity import SpecificityReward
from .rewards.r10_fluency import FluencyReward
from .rewards.emotion_utils import (
    analyze_emotions,
    calculate_empathy,
    calculate_non_confrontational,
)
from .rewards.r11_toxicity import (
    analyze_toxicity,
    calculate_non_toxicity,
)


REQUESTED_METRICS = (
    "PRS",
    "QS",
    "SAFE",
    "EMP",
    "CON",
    "PERS",
    "SPEC",
    "FLU",
    "TOX",
)


class StandaloneCounterNarrativeEvaluator:
    """
    Optimized reference-free evaluator.

    Improvements:
    - scores-only Llama judging;
    - one shared emotion-model pass for EMP and non-confrontational tone;
    - one shared Detoxify pass for TOX;
    - independent deterministic PRS;
    - no per-item Distinct-2.
    """

    def __init__(
        self,
        judge_config: ModelConfig | dict,
        llm_weight: float = 0.5,
        reward_weight: float = 0.5,
    ):
        if llm_weight < 0 or reward_weight < 0:
            raise ValueError(
                "Score weights must be non-negative"
            )

        if llm_weight + reward_weight == 0:
            raise ValueError(
                "At least one score weight must be positive"
            )

        total = llm_weight + reward_weight
        self.llm_weight = llm_weight / total
        self.reward_weight = reward_weight / total

        self.judge = CounterNarrativeJudge(config=judge_config)

        self.politeness = PolitenessRespectReward()
        self.quality = QualityReward()
        self.safety = SafetyDeEscalationReward()
        self.constructiveness = ConstructivenessReward()
        self.persuasiveness = PersuasivenessReward()
        self.specificity = SpecificityReward()
        self.fluency = FluencyReward()

    @staticmethod
    def _safe_score(
        reward,
        reward_input: RewardInput,
    ) -> float:
        try:
            value = float(
                reward.score(reward_input)
            )
        except Exception as exc:
            print(
                f"[WARN] "
                f"{reward.__class__.__name__} failed: {exc}"
            )
            return 0.0

        return max(0.0, min(1.0, value))

    def score_rewards_only(
        self,
        counter_narrative: str,
    ) -> Dict[str, float]:
        text = str(counter_narrative).strip()

        if not text:
            raise ValueError(
                "counter_narrative cannot be empty"
            )

        reward_input = RewardInput(
            hate_speech="",
            counter_narrative=text,
            ground_truth=None,
        )

        scores: Dict[str, float] = {}

        # Stage 5: independent deterministic PRS.
        scores["PRS"] = self._safe_score(
            self.politeness,
            reward_input,
        )

        # Existing semantic rewards.
        scores["QS"] = self._safe_score(
            self.quality,
            reward_input,
        )
        scores["SAFE"] = self._safe_score(
            self.safety,
            reward_input,
        )
        scores["CON"] = self._safe_score(
            self.constructiveness,
            reward_input,
        )
        scores["PERS"] = self._safe_score(
            self.persuasiveness,
            reward_input,
        )
        scores["SPEC"] = self._safe_score(
            self.specificity,
            reward_input,
        )
        scores["FLU"] = self._safe_score(
            self.fluency,
            reward_input,
        )

        # Stage 3: one emotion inference.
        emotion_result = analyze_emotions(text)

        scores["EMP"] = calculate_empathy(
            emotion_result
        )
        scores["NON_CONFRONTATIONAL"] = (
            calculate_non_confrontational(
                emotion_result
            )
        )

        # Stage 4: one Detoxify inference.
        toxicity_result = analyze_toxicity(text)

        scores["TOX"] = calculate_non_toxicity(
            toxicity_result
        )

        scores["combined_reward_score"] = round(
            sum(
                scores[metric]
                for metric in REQUESTED_METRICS
            ) / len(REQUESTED_METRICS),
            6,
        )

        return scores

    def _merge_result(
        self,
        text: str,
        llm_result: Dict[str, Any],
        reward_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        llm_percentage = float(
            llm_result["scores"]["percentage"]
        )
        reward_percentage = round(
            reward_scores[
                "combined_reward_score"
            ] * 100.0,
            2,
        )

        final_score_percentage = round(
            self.llm_weight * llm_percentage
            + self.reward_weight * reward_percentage,
            2,
        )

        return {
            "counter_narrative": text,
            "llm_judge": llm_result,
            "reward_functions": reward_scores,
            "final_summary": {
                "llm_percentage": llm_percentage,
                "reward_percentage": reward_percentage,
                "final_score_percentage": (
                    final_score_percentage
                ),
                "llm_weight": self.llm_weight,
                "reward_weight": self.reward_weight,
            },
        }

    def score(
        self,
        counter_narrative: str,
    ) -> Dict[str, Any]:
        text = str(counter_narrative).strip()

        if not text:
            raise ValueError(
                "counter_narrative cannot be empty"
            )

        llm_result = self.judge.score(text)
        reward_scores = self.score_rewards_only(
            text
        )

        return self._merge_result(
            text,
            llm_result,
            reward_scores,
        )

    def score_batch(
        self,
        counter_narratives: List[str],
    ) -> List[Dict[str, Any]]:
        cleaned = [
            str(value).strip()
            for value in counter_narratives
        ]

        if not cleaned:
            return []

        if any(not value for value in cleaned):
            raise ValueError(
                "counter_narratives cannot contain empty values"
            )

        llm_results = self.judge.score_batch(
            cleaned
        )

        if len(llm_results) != len(cleaned):
            raise ValueError(
                "Judge returned an unexpected batch size."
            )

        completed = []

        for text, llm_result in zip(
            cleaned,
            llm_results,
        ):
            reward_scores = self.score_rewards_only(
                text
            )

            completed.append(
                self._merge_result(
                    text,
                    llm_result,
                    reward_scores,
                )
            )

        return completed


def evaluate_counter_narrative(
    counter_narrative: str,
    judge_config: ModelConfig | dict,
) -> Dict[str, Any]:
    evaluator = StandaloneCounterNarrativeEvaluator(judge_config=judge_config)
    return evaluator.score(counter_narrative)
