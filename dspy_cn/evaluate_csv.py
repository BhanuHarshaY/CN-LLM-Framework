import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .model_config import ModelConfig, load_config, resolve_package_path
from .standalone_evaluator import StandaloneCounterNarrativeEvaluator
from .rewards.distinct2 import Distinct2Score

SCORE_METRICS = ("PRS", "QS", "SAFE", "EMP", "CON", "PERS", "SPEC", "FLU", "TOX")
EMPTY_TEXT_ERROR = "Counter-narrative is empty."


def make_batches(dataframe: pd.DataFrame, batch_size: int) -> Iterable[Tuple[int, pd.DataFrame]]:
    for start in range(0, len(dataframe), batch_size):
        yield start, dataframe.iloc[start:start + batch_size]


def flatten_evaluation(row: pd.Series, evaluation: Dict[str, Any], error: str = "") -> Dict[str, Any]:
    output = row.to_dict()
    llm = evaluation["llm_judge"]
    scores = llm["scores"]
    reward = evaluation["reward_functions"]
    final = evaluation["final_summary"]

    output.update({
        "JUDGE_PROVIDER": llm["provider"],
        "JUDGE_MODEL": llm["model"],
        **{f"LLM_{name}": scores[name] for name in SCORE_METRICS},
        "LLM_TOTAL": scores["total"],
        "LLM_MAX_SCORE": scores["max_score"],
        "LLM_PERCENTAGE": scores["percentage"],
        **{f"REWARD_{name}": reward[name] for name in SCORE_METRICS},
        "REWARD_NON_CONFRONTATIONAL": reward["NON_CONFRONTATIONAL"],
        "COMBINED_REWARD_SCORE": reward["combined_reward_score"],
        "REWARD_PERCENTAGE": final["reward_percentage"],
        "FINAL_SCORE_PERCENTAGE": final["final_score_percentage"],
        "ERROR": error,
    })
    return output


def error_row(row: pd.Series, message: str) -> Dict[str, Any]:
    output = row.to_dict()
    output["ERROR"] = message
    return output


def save_checkpoint(results: Dict[int, Dict[str, Any]], output_file: Path) -> None:
    ordered = [results[position] for position in sorted(results)]
    pd.DataFrame(ordered).to_csv(output_file, index=False)


def load_existing_results(output_file: Path) -> Dict[int, Dict[str, Any]]:
    """Reload a prior checkpoint and keep only rows that don't need rescoring.

    A row is "done" if it has a real LLM_TOTAL, or if it permanently failed
    because the counter-narrative was empty (rescoring won't change that).
    Any other error (timeout, malformed judge response, etc.) is retried.
    """
    if not output_file.exists():
        return {}

    existing = pd.read_csv(output_file)
    results: Dict[int, Dict[str, Any]] = {}
    for position, row in existing.iterrows():
        record = row.to_dict()
        error_value = record.get("ERROR")
        error_text = str(error_value).strip() if pd.notna(error_value) else ""
        if pd.notna(record.get("LLM_TOTAL")) or error_text == EMPTY_TEXT_ERROR:
            results[position] = record
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score counter-narratives with the decimal (0.0-5.0) LLM-as-judge + reward-function pipeline."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset key defined under evaluation.datasets in config.yaml (e.g. baseline, enhanced)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to dspy_cn/config.yaml)")
    parser.add_argument("--model", default=None, help="Override judge_llm.model from config.yaml")
    parser.add_argument("--max-rows", type=int, default=None, help="Override evaluation.max_rows for a smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    judge_settings = dict(config["judge_llm"])
    if args.model:
        judge_settings["model"] = args.model
    judge_config = ModelConfig.from_mapping(judge_settings)

    eval_config = config.get("evaluation", {})
    datasets_config = eval_config.get("datasets", {})
    if args.dataset not in datasets_config:
        raise KeyError(f"Unknown dataset '{args.dataset}'. Available: {sorted(datasets_config)}")
    dataset_config = datasets_config[args.dataset]

    input_file = resolve_package_path(dataset_config["input_file"])
    output_file = resolve_package_path(dataset_config["output_file"])
    summary_file = resolve_package_path(dataset_config["summary_file"])
    cn_column = str(dataset_config["counter_narrative_column"])

    max_rows = args.max_rows if args.max_rows is not None else eval_config.get("max_rows")
    batch_size = int(eval_config.get("batch_size", 8))
    checkpoint_every = int(eval_config.get("checkpoint_every_batches", 10))

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    dataframe = pd.read_csv(input_file)
    if cn_column not in dataframe.columns:
        raise ValueError(f"Column '{cn_column}' not found. Available: {dataframe.columns.tolist()}")
    if max_rows is not None:
        dataframe = dataframe.head(int(max_rows)).copy()
    dataframe = dataframe.reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    evaluator = StandaloneCounterNarrativeEvaluator(
        judge_config=judge_config,
        llm_weight=float(eval_config.get("llm_weight", 0.5)),
        reward_weight=float(eval_config.get("reward_weight", 0.5)),
    )

    results: Dict[int, Dict[str, Any]] = load_existing_results(output_file)
    if results:
        print(f"Resuming from checkpoint: {len(results)}/{len(dataframe)} rows already scored in {output_file}")

    print(f"Dataset: {args.dataset} ({input_file})")
    print(f"Rows to evaluate: {len(dataframe)}")
    print(f"Judge: {judge_config.provider}/{judge_config.model}")
    print(f"Batch size: {batch_size}")

    total_batches = (len(dataframe) + batch_size - 1) // batch_size
    batch_failures = 0

    for batch_number, (batch_start, batch_df) in enumerate(make_batches(dataframe, batch_size), start=1):
        valid_positions: List[int] = []
        valid_rows: List[pd.Series] = []
        valid_texts: List[str] = []

        for offset, (_, row) in enumerate(batch_df.iterrows()):
            position = batch_start + offset
            if position in results:
                continue
            raw = row[cn_column]
            text = "" if pd.isna(raw) else str(raw).strip()
            if not text:
                results[position] = error_row(row, EMPTY_TEXT_ERROR)
            else:
                valid_positions.append(position)
                valid_rows.append(row)
                valid_texts.append(text)

        if not valid_texts:
            continue

        print(f"Batch {batch_number}/{total_batches}: rows {batch_start + 1}-{batch_start + len(batch_df)}")
        try:
            evaluations = evaluator.score_batch(valid_texts)
            for position, row, evaluation in zip(valid_positions, valid_rows, evaluations):
                results[position] = flatten_evaluation(row, evaluation)
        except Exception as batch_exc:
            batch_failures += 1
            print(f"[WARN] Batch failed; using row fallback: {batch_exc}")
            for position, row, text in zip(valid_positions, valid_rows, valid_texts):
                try:
                    results[position] = flatten_evaluation(
                        row,
                        evaluator.score(text),
                        error="Batch failed; single-row fallback succeeded.",
                    )
                except Exception as row_exc:
                    results[position] = error_row(row, str(row_exc))

        if checkpoint_every > 0 and batch_number % checkpoint_every == 0:
            save_checkpoint(results, output_file)

    save_checkpoint(results, output_file)

    texts = dataframe[cn_column].dropna().astype(str).map(str.strip)
    texts = [text for text in texts if text]
    pd.DataFrame([{
        "dataset": args.dataset,
        "input_file": str(input_file),
        "rows": len(dataframe),
        "judge_provider": judge_config.provider,
        "judge_model": judge_config.model,
        "batch_size": batch_size,
        "dataset_distinct2": Distinct2Score().score_batch(texts),
        "batch_failures": batch_failures,
    }]).to_csv(summary_file, index=False)

    print(f"Evaluation complete. Output: {output_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
