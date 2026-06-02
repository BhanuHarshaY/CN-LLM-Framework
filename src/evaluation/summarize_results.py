from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
ENHANCED_SOURCE_PATH = (
    ROOT_DIR / "src" / "data" / "Generate_CN.csv"
)

BASELINE_PATH = RESULTS_DIR / "multitarget_conan_scores.csv"
ENHANCED_PATH = RESULTS_DIR / "generate_cn_scores.csv"
SUMMARY_PATH = RESULTS_DIR / "comparison_summary.csv"
TARGET_SUMMARY_PATH = RESULTS_DIR / "enhanced_target_breakdown.csv"

ASPECTS = ["opposition", "relatedness", "specificity", "toxicity", "fluency"]


def load_scores(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    df = pd.read_csv(path)
    missing = [aspect for aspect in ASPECTS if aspect not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns in {path}: {missing}")

    return df


def ensure_enhanced_target(enhanced):
    if "target" in enhanced.columns:
        return enhanced

    source = pd.read_csv(ENHANCED_SOURCE_PATH)
    if "TARGET" not in source.columns:
        raise ValueError(f"Missing TARGET column in {ENHANCED_SOURCE_PATH}")
    if len(source) < len(enhanced):
        raise ValueError(
            "Enhanced source file has fewer rows than enhanced result file; "
            "cannot backfill target values."
        )

    enhanced = enhanced.copy()
    enhanced.insert(2, "target", source["TARGET"].head(len(enhanced)).fillna("").astype(str))
    enhanced.to_csv(ENHANCED_PATH, index=False)
    return enhanced


def build_summary(baseline, enhanced):
    rows = []

    for aspect in ASPECTS:
        baseline_mean = pd.to_numeric(baseline[aspect], errors="coerce").mean()
        enhanced_mean = pd.to_numeric(enhanced[aspect], errors="coerce").mean()
        rows.append(
            {
                "aspect": aspect,
                "multitarget_conan_mean": baseline_mean,
                "enhanced_dataset_mean": enhanced_mean,
                "gap_enhanced_minus_multitarget": enhanced_mean - baseline_mean,
                "absolute_gap": abs(enhanced_mean - baseline_mean),
            }
        )

    return pd.DataFrame(rows).sort_values("absolute_gap", ascending=False)


def build_target_summary(enhanced):
    if "target" not in enhanced.columns:
        raise ValueError("Enhanced result file is missing the 'target' column")

    scored = enhanced.copy()
    for aspect in ASPECTS:
        scored[aspect] = pd.to_numeric(scored[aspect], errors="coerce")

    return (
        scored.groupby("target", dropna=False)[ASPECTS]
        .mean()
        .reset_index()
        .assign(row_count=scored.groupby("target", dropna=False).size().values)
        [["target", "row_count", *ASPECTS]]
        .sort_values("row_count", ascending=False)
    )


def main():
    baseline = load_scores(BASELINE_PATH)
    enhanced = load_scores(ENHANCED_PATH)
    enhanced = ensure_enhanced_target(enhanced)
    summary = build_summary(baseline, enhanced)
    target_summary = build_target_summary(enhanced)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    target_summary.to_csv(TARGET_SUMMARY_PATH, index=False)

    print("Comparison summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    print("Enhanced dataset per-target breakdown")
    print(target_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    print("Largest gaps")
    for row in summary.itertuples(index=False):
        direction = "enhanced higher" if row.gap_enhanced_minus_multitarget > 0 else "baseline higher"
        print(f"{row.aspect}: {row.absolute_gap:.3f} ({direction})")
    print()
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved target breakdown to {TARGET_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
