from functools import lru_cache

from .llm_clients import get_judge_client
from .model_config import ModelConfig, load_config, DEFAULT_CONFIG_PATH

PRS_RUBRIC = """Criterion: Does the generated counter-narrative reflect a polite and respectful tone relevant to LGBTQ+ individuals and communities?
0 = No polite or respectful language used in the counter-narrative framing
1 = Attempts polite and respectful framing, but the tone is inconsistent
2 = Clearly and consistently reflects a polite and respectful tone"""

CCNC_RUBRIC = """Criterion: Does the counter-narrative remain coherent, relevant, and contextually appropriate in response to the specific harmful or offensive speech?
0 = The counter-narrative is off-topic, incoherent, or fails to address the context
1 = Partially coherent or contextually relevant but lacks clarity or consistency
2 = Clearly and consistently coherent, relevant, and context-aware throughout the response"""

QS_RUBRIC = """Criterion: Does the counter-narrative demonstrate richness, grammatical correctness, and strong relatedness to the harmful speech?
0 = Poor quality with little depth, major grammatical issues, or weak/no connection to the harmful speech
1 = Some depth and generally understandable, but may contain minor grammatical issues or partial relatedness
2 = Clear, well-written, rich in content, and strongly related to the harmful speech"""


@lru_cache(maxsize=4)
def _judge_client(config_path: str = str(DEFAULT_CONFIG_PATH)):
    config = load_config(config_path)
    return get_judge_client(ModelConfig.from_mapping(config["judge_llm"]))


def llm_judge(
    hate_speech: str,
    counter_narrative: str,
    config_path: str = str(DEFAULT_CONFIG_PATH),
) -> dict:
    prompt = f"""You are an impartial judge. Score the counter-narrative on THREE criteria.

HATE SPEECH:
{hate_speech}

COUNTER-NARRATIVE:
{counter_narrative}

--- CRITERION 1: Politeness and Respectful Score (PRS) ---
{PRS_RUBRIC}

--- CRITERION 2: Contextual Counter-Narrative Coherence Score (CCNC) ---
{CCNC_RUBRIC}

--- CRITERION 3: Quality Score (QS) ---
{QS_RUBRIC}

Return ONLY valid JSON with integer scores:
{{"PRS": 0, "CCNC": 0, "QS": 0}}"""

    try:
        scores = _judge_client(config_path).generate_json(prompt)
        prs = max(0, min(2, int(scores.get("PRS", 0))))
        ccnc = max(0, min(2, int(scores.get("CCNC", 0))))
        qs = max(0, min(2, int(scores.get("QS", 0))))
    except Exception as exc:
        print(f"[WARN] Judge error: {exc}")
        prs, ccnc, qs = 1, 1, 1

    total = prs + ccnc + qs
    return {"PRS": prs, "CCNC": ccnc, "QS": qs, "total": total, "pct": total / 6 * 100}


def cn_metric_llm_judge(example, prediction, trace=None) -> float:
    scores = llm_judge(str(example.hate_speech), str(prediction.counter_narrative))
    return scores["total"] / 6.0


def cn_metric_rewards(example, prediction, trace=None) -> float:
    global _reward_evaluator
    if "_reward_evaluator" not in globals() or _reward_evaluator is None:
        from dspy_cn.evaluator import RewardEvaluator
        globals()["_reward_evaluator"] = RewardEvaluator()

    ground_truth = str(example.ground_truth) if hasattr(example, "ground_truth") else None
    scores = _reward_evaluator.score_single(
        str(example.hate_speech),
        str(prediction.counter_narrative),
        ground_truth,
    )
    return float(scores["combined"])
