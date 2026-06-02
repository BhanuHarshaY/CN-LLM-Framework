import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-4o"
ASPECTS = ["Opposition", "Relatedness", "Specificity", "Toxicity", "Fluency"]
OUTPUT_COLUMNS = [
    "hate_speech",
    "counter_narrative",
    "opposition",
    "relatedness",
    "specificity",
    "toxicity",
    "fluency",
]
ASPECT_COLUMNS = [aspect.lower() for aspect in ASPECTS]
OUTPUT_TOKENS_PER_CALL_ESTIMATE = 100
INPUT_COST_PER_1M = 2.50
OUTPUT_COST_PER_1M = 10.00

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "src" / "data"
RESULTS_DIR = ROOT_DIR / "results"
PROMPTS_PATH = ROOT_DIR / "src" / "evaluation" / "prompts.csv"

DATASETS = {
    "baseline": {
        "label": "Multitarget-CONAN",
        "path": DATA_DIR / "Multitarget-CONAN.csv",
        "hate_speech_column": "HATE_SPEECH",
        "counter_narrative_column": "COUNTER_NARRATIVE",
        "output_path": RESULTS_DIR / "multitarget_conan_scores.csv",
        "output_columns": OUTPUT_COLUMNS,
    },
    "enhanced": {
        "label": "Generate_CN enhanced dataset",
        "path": DATA_DIR / "Generate_CN.csv",
        "hate_speech_column": "HATE_SPEECH",
        "counter_narrative_column": "GENERATED_COUNTER_NARRATIVE",
        "target_column": "TARGET",
        "output_path": RESULTS_DIR / "generate_cn_scores.csv",
        "output_columns": ["hate_speech", "counter_narrative", "target", *ASPECT_COLUMNS],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate counter narratives with gpt-4o across five aspects."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "baseline", "enhanced"],
        default="all",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit per selected dataset for sample runs.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print row, token, and cost estimates without making API calls.",
    )
    return parser.parse_args()


def selected_dataset_keys(dataset_arg):
    if dataset_arg == "all":
        return ["baseline", "enhanced"]
    return [dataset_arg]


def load_prompts():
    prompts = pd.read_csv(PROMPTS_PATH)
    missing = [aspect for aspect in ASPECTS if aspect not in prompts.columns]
    if missing:
        raise ValueError(f"Missing prompt columns in {PROMPTS_PATH}: {missing}")
    return {aspect: str(prompts[aspect].iloc[0]) for aspect in ASPECTS}


def load_dataset(config, max_rows=None):
    df = pd.read_csv(config["path"])
    required = [config["hate_speech_column"], config["counter_narrative_column"]]
    if config.get("target_column"):
        required.append(config["target_column"])
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {config['path']}: {missing}")

    if max_rows is not None:
        df = df.head(max_rows)

    normalized = pd.DataFrame(
        {
            "hate_speech": df[config["hate_speech_column"]].fillna("").astype(str),
            "counter_narrative": df[config["counter_narrative_column"]]
            .fillna("")
            .astype(str),
        }
    )
    if config.get("target_column"):
        normalized["target"] = df[config["target_column"]].fillna("").astype(str)

    return normalized


def user_content(hate_speech, counter_narrative):
    return (
        f"Hate Speech: {hate_speech}\n"
        f"Counter Narrative: {counter_narrative}\n"
        "Scores: "
    )


def batch_system_prompt(prompts):
    rubric_text = "\n\n".join(
        f"{aspect.upper()} RUBRIC:\n{prompts[aspect]}" for aspect in ASPECTS
    )
    return (
        "You are evaluating a counter narrative response to hate speech. "
        "Score all five aspects using the rubrics below. "
        "Return only valid JSON with exactly these lowercase integer keys: "
        "opposition, relatedness, specificity, toxicity, fluency. "
        "Each value must be an integer from 1 to 5. Do not include explanations.\n\n"
        f"{rubric_text}"
    )


def estimate_tokens(text):
    return len(text) / 4


def estimate_cost(dataset_keys, prompts, max_rows=None):
    total_rows = 0
    total_calls = 0
    total_input_tokens = 0

    print("Cost estimate for gpt-4o")
    print(f"Pricing assumption: ${INPUT_COST_PER_1M:.2f}/1M input tokens, "
          f"${OUTPUT_COST_PER_1M:.2f}/1M output tokens")
    print("Batch mode: one API call scores all five aspects for one row")
    print(f"Output estimate: {OUTPUT_TOKENS_PER_CALL_ESTIMATE} tokens per row call")
    print()
    system_prompt = batch_system_prompt(prompts)

    for key in dataset_keys:
        config = DATASETS[key]
        df = load_dataset(config, max_rows=max_rows)
        rows = len(df)
        calls = rows
        input_tokens = 0

        for row in df.itertuples(index=False):
            content = user_content(row.hate_speech, row.counter_narrative)
            input_tokens += estimate_tokens(system_prompt + content)

        total_rows += rows
        total_calls += calls
        total_input_tokens += input_tokens

        print(
            f"{config['label']}: rows={rows:,}, calls={calls:,}, "
            f"estimated_input_tokens={input_tokens:,.0f}"
        )

    total_output_tokens = total_calls * OUTPUT_TOKENS_PER_CALL_ESTIMATE
    input_cost = total_input_tokens / 1_000_000 * INPUT_COST_PER_1M
    output_cost = total_output_tokens / 1_000_000 * OUTPUT_COST_PER_1M

    print()
    print(f"Total rows: {total_rows:,}")
    print(f"Total calls: {total_calls:,}")
    print(f"Estimated input tokens: {total_input_tokens:,.0f}")
    print(f"Estimated output tokens: {total_output_tokens:,.0f}")
    print(f"Estimated input cost: ${input_cost:,.2f}")
    print(f"Estimated output cost: ${output_cost:,.2f}")
    print(f"Estimated total cost: ${input_cost + output_cost:,.2f}")


def load_existing_results(config, source_df):
    output_path = config["output_path"]
    output_columns = config["output_columns"]
    if output_path.exists():
        existing = pd.read_csv(output_path)
        for column in output_columns:
            if column not in existing.columns:
                existing[column] = pd.NA
        existing = existing[output_columns]
    else:
        existing = pd.DataFrame(columns=output_columns)

    if len(existing) < len(source_df):
        missing = source_df.iloc[len(existing) :].copy()
        for aspect in ASPECT_COLUMNS:
            missing[aspect] = pd.NA
        existing = pd.concat([existing, missing[output_columns]], ignore_index=True)

    existing = existing.head(len(source_df)).copy()

    metadata_columns = ["hate_speech", "counter_narrative"]
    if "target" in output_columns:
        metadata_columns.append("target")
    for column in metadata_columns:
        existing[column] = source_df[column].values

    return existing[output_columns]


def parse_batch_scores(response_text):
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse JSON scores from response: {response_text}")
        parsed = json.loads(match.group(0))

    scores = {}
    for aspect in ASPECT_COLUMNS:
        value = parsed.get(aspect)
        if value is None:
            raise ValueError(f"Missing '{aspect}' in response: {response_text}")
        try:
            score = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid '{aspect}' score in response: {response_text}") from exc
        if score < 1 or score > 5:
            raise ValueError(f"Out-of-range '{aspect}' score in response: {response_text}")
        scores[aspect] = score

    return scores


def judge_row(client, system_prompt, hate_speech, counter_narrative):
    content = user_content(hate_speech, counter_narrative)
    last_error = None

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content or ""
            return parse_batch_scores(response_text)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt * 2)

    raise RuntimeError(f"Failed after 3 attempts: {last_error}") from last_error


def save_results(output_path, results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)


def evaluate_dataset(client, key, prompts, max_rows=None):
    config = DATASETS[key]
    source_df = load_dataset(config, max_rows=max_rows)
    results = load_existing_results(config, source_df)
    system_prompt = batch_system_prompt(prompts)

    print(f"Evaluating {config['label']} with {MODEL}: {len(source_df):,} rows")
    print(f"Output: {config['output_path']}")
    print("Batch mode: one API call per row returns all five scores")

    for row_index, row in results.iterrows():
        if not row[ASPECT_COLUMNS].isna().any():
            continue

        print(f"Row {row_index + 1:,}/{len(results):,} - {config['label']}")
        scores = judge_row(
            client,
            system_prompt,
            row["hate_speech"],
            row["counter_narrative"],
        )
        for aspect_column, score in scores.items():
            results.at[row_index, aspect_column] = score
        time.sleep(0.2)

        if (row_index + 1) % 10 == 0:
            save_results(config["output_path"], results)
            print(f"Checkpoint saved after {row_index + 1:,} rows")

    save_results(config["output_path"], results)

    incomplete = results[ASPECT_COLUMNS].isna().any(axis=1).sum()
    if incomplete:
        print(f"Finished with {incomplete:,} incomplete rows in {config['output_path']}")
    else:
        print(f"Finished {config['label']}; all selected rows have scores.")


def main():
    args = parse_args()
    prompts = load_prompts()
    dataset_keys = selected_dataset_keys(args.dataset)

    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be a positive integer")

    if args.estimate_only:
        estimate_cost(dataset_keys, prompts, max_rows=args.max_rows)
        return

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = OpenAI()
    for key in dataset_keys:
        evaluate_dataset(client, key, prompts, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
