from pathlib import Path
import os

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MPLCONFIG_DIR = PLOTS_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASELINE_PATH = RESULTS_DIR / "multitarget_conan_scores.csv"
ENHANCED_PATH = RESULTS_DIR / "generate_cn_scores.csv"
SUMMARY_PATH = RESULTS_DIR / "comparison_summary.csv"
TARGET_PATH = RESULTS_DIR / "enhanced_target_breakdown.csv"

ASPECTS = ["opposition", "relatedness", "specificity", "toxicity", "fluency"]
ASPECT_LABELS = {
    "opposition": "Opposition",
    "relatedness": "Relatedness",
    "specificity": "Specificity",
    "toxicity": "Toxicity",
    "fluency": "Fluency",
}

BASELINE_COLOR = "#2f6fbb"
ENHANCED_COLOR = "#d76f30"
GAP_COLOR = "#7a4aa0"
GRID_COLOR = "#d8d8d8"


def load_inputs():
    baseline = pd.read_csv(BASELINE_PATH)
    enhanced = pd.read_csv(ENHANCED_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    target = pd.read_csv(TARGET_PATH)
    return baseline, enhanced, summary, target


def style_axes(ax, title=None, ylabel=None, xlabel=None):
    if title:
        ax.set_title(title, fontsize=15, weight="bold", pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_mean_scores(summary):
    labels = [ASPECT_LABELS[aspect] for aspect in summary["aspect"]]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 6))
    baseline = ax.bar(
        x - width / 2,
        summary["multitarget_conan_mean"],
        width,
        label="Multitarget-CONAN",
        color=BASELINE_COLOR,
    )
    enhanced = ax.bar(
        x + width / 2,
        summary["enhanced_dataset_mean"],
        width,
        label="Enhanced dataset",
        color=ENHANCED_COLOR,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 5.4)
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax, "Mean GPT-4o Scores by Aspect", "Mean score (1-5)")

    for bars in [baseline, enhanced]:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    save_figure(fig, "mean_scores_by_aspect.png")


def plot_score_gaps(summary):
    ordered = summary.sort_values("gap_enhanced_minus_multitarget")
    labels = [ASPECT_LABELS[aspect] for aspect in ordered["aspect"]]
    values = ordered["gap_enhanced_minus_multitarget"]
    colors = [ENHANCED_COLOR if value > 0 else BASELINE_COLOR for value in values]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    style_axes(
        ax,
        "Enhanced Minus Baseline Score Gap",
        "Aspect",
        "Mean score difference",
    )
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.grid(axis="y", visible=False)

    for bar, value in zip(bars, values):
        x_position = value - 0.05 if value < 0 else value + 0.05
        alignment = "right" if value < 0 else "left"
        ax.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=alignment,
            fontsize=10,
        )

    save_figure(fig, "enhanced_vs_baseline_gaps.png")


def plot_target_heatmap(target_summary):
    target_summary = target_summary.sort_values("target")
    data = target_summary[ASPECTS].to_numpy()
    labels = [ASPECT_LABELS[aspect] for aspect in ASPECTS]
    targets = target_summary["target"].tolist()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    image = ax.imshow(data, cmap="YlGnBu", vmin=1, vmax=5, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_title("Enhanced Dataset Mean Scores by Target", fontsize=15, weight="bold", pad=14)

    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            ax.text(
                column,
                row,
                f"{data[row, column]:.2f}",
                ha="center",
                va="center",
                color="#111111",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean score (1-5)")
    save_figure(fig, "enhanced_target_score_heatmap.png")


def plot_target_counts(target_summary):
    ordered = target_summary.sort_values("row_count")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.barh(ordered["target"], ordered["row_count"], color="#4f9d8f")
    style_axes(ax, "Enhanced Dataset Rows by Target", "Target", "Rows")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.grid(axis="y", visible=False)
    ax.bar_label(bars, padding=3, fontsize=9)

    save_figure(fig, "enhanced_target_row_counts.png")


def plot_score_distributions(baseline, enhanced):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    axes = axes.flatten()
    bins = np.arange(0.5, 6.5, 1)

    for index, aspect in enumerate(ASPECTS):
        ax = axes[index]
        ax.hist(
            baseline[aspect],
            bins=bins,
            alpha=0.72,
            label="Multitarget-CONAN",
            color=BASELINE_COLOR,
            edgecolor="white",
        )
        ax.hist(
            enhanced[aspect],
            bins=bins,
            alpha=0.72,
            label="Enhanced",
            color=ENHANCED_COLOR,
            edgecolor="white",
        )
        ax.set_title(ASPECT_LABELS[aspect], fontsize=12, weight="bold")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Score")
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower right", bbox_to_anchor=(0.92, 0.12))
    fig.suptitle("Score Distributions by Aspect", fontsize=16, weight="bold", y=0.98)

    save_figure(fig, "score_distributions_by_aspect.png")


def main():
    baseline, enhanced, summary, target_summary = load_inputs()
    plot_mean_scores(summary)
    plot_score_gaps(summary)
    plot_target_heatmap(target_summary)
    plot_target_counts(target_summary)
    plot_score_distributions(baseline, enhanced)
    print(f"All plots saved in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
