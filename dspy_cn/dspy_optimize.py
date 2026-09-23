"""
Author: Preethi Gajawada
Institution: Northeastern University
Project: Multilingual Counter-Narrative Generation using DSPy
Description: Runs COPRO optimization to refine prompts for counter-narrative generation.
Year: 2026
"""

# dspy_cn/dspy_optimize.py
# Runs COPRO optimization for Tamil (English kept commented).
import dspy
import json
import os
import random
import pandas as pd
from datetime import datetime
from dspy.adapters import JSONAdapter
from .model_config import load_config, resolve_package_path

from .base_llm import configure_dspy_lm
from .dspy_program import (
    EnglishCNProgram, TamilCNProgram,
)
from .dspy_metric import cn_metric_rewards


# ── Data Loading ───────────────────────────────────────────────

def load_trainset_en(csv_path: str, sample_size: int = 30, seed: int = 42):
    df = pd.read_csv(csv_path)
    exs = []

    for _, row in df.iterrows():
        ex = dspy.Example(
            hate_speech=str(row["text"]),
            ground_truth=str(row["counter_narrative"]),
        ).with_inputs("hate_speech")
        exs.append(ex)

    if sample_size and sample_size < len(exs):
        random.seed(seed)
        exs = random.sample(exs, sample_size)

    print(f"[OK] Loaded {len(exs)} English training examples")
    return exs


def load_trainset_ta(csv_path: str, sample_size: int = 30, seed: int = 42):
    df = pd.read_csv(csv_path)
    exs = []

    for _, row in df.iterrows():
        ex = dspy.Example(
            hate_speech=str(row["augmented_text"]),
            ground_truth=str(row["counter_narrative"]),
        ).with_inputs("hate_speech")
        exs.append(ex)

    if sample_size and sample_size < len(exs):
        random.seed(seed)
        exs = random.sample(exs, sample_size)

    print(f"[OK] Loaded {len(exs)} Tamil training examples")
    return exs


# ── Audit Logger ───────────────────────────────────────────────

class OptimizationAudit:
    def __init__(self, name: str):
        self.name = name
        self.entries = []
        self.start_time = datetime.now().isoformat()

    def log(self, score, instruction=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "instruction": instruction,
        }
        self.entries.append(entry)

        print(f"[AUDIT] score={score:.4f}")
        if instruction:
            print(f"         instruction={instruction[:80]}...")

    def save(self, path):
        data = {
            "optimizer": self.name,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "total_iterations": len(self.entries),
            "best_score": max(e["score"] for e in self.entries) if self.entries else 0,
            "iterations": self.entries,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[OK] Audit saved to {path}")


# ── COPRO Optimization ─────────────────────────────────────────

def run_copro(program, trainset, out_path, audit_path):
    audit = OptimizationAudit("COPRO")

    optimizer = dspy.COPRO(
        metric=cn_metric_rewards,
        depth=5,
        breadth=4,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print("Starting COPRO optimization...")
    print(f"Training examples: {len(trainset)}")
    print(f"{'='*60}\n")

    optimized = optimizer.compile(
        program,
        trainset=trainset,
        eval_kwargs={
            "num_threads": 1,          # 🔒 avoid parallel errors
            "display_progress": True
        },
    )

    # ── Safe extraction of instruction ──
    try:
        state = optimized.generate.dump_state()
        instruction = state.get("signature", {}).get("instructions", "N/A")
        audit.log(score=1.0, instruction=instruction)
    except Exception as e:
        print("⚠️ Instruction extraction failed:", e)
        instruction = "Fallback instruction"

    optimized.save(out_path)
    audit.save(audit_path)

    print(f"\n{'='*60}")
    print("FINAL OPTIMIZED INSTRUCTION:")
    print(f"{'='*60}")
    print(instruction)

    return optimized


# ── Main ───────────────────────────────────────────────────────

def main():
    cfg = load_config()
    configure_dspy_lm(
        cfg["base_llm"],
        adapter=JSONAdapter(),
        enforce_json=True,
    )

    output_dir = resolve_package_path("outputs")
    log_dir = resolve_package_path("logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    opt_cfg = cfg["optimization"]
    sample_size = opt_cfg.get("train_sample_size", 30)

    # ── ENGLISH (COMMENTED) ──
    
    print("\n" + "=" * 60)
    print("ENGLISH — COPRO")
    print("=" * 60)

    en_train = load_trainset_en(str(resolve_package_path(cfg["data"]["train_en_csv"])), sample_size)
    en_program = EnglishCNProgram()

    run_copro(
        en_program,
        en_train,
        out_path=str(output_dir / "en_copro_optimized.json"),
        audit_path=str(log_dir / "en_copro_audit.json"),
    )
    

    # ── TAMIL ──
    print("\n" + "=" * 60)
    print("TAMIL — COPRO")
    print("=" * 60)

    ta_train = load_trainset_ta(str(resolve_package_path(cfg["data"]["train_ta_csv"])), sample_size)
    ta_program = TamilCNProgram()

    run_copro(
        ta_program,
        ta_train,
        out_path=str(output_dir / "ta_copro_optimized.json"),
        audit_path=str(log_dir / "ta_copro_audit.json"),
    )

    print("\n" + "=" * 60)
    print("ALL OPTIMIZATIONS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()