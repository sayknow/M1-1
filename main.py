"""Run the complete ECOS collection, cleaning, and analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from src.analyzer import run_analysis
from src.cleaner import build_processed_data
from src.collector import collect_all


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"


def main() -> None:
    print("[1/3] Collecting official ECOS data (2015-01 to 2025-12)")
    collect_all(RAW_DIR, env_file=ROOT / ".env")

    print("[2/3] Cleaning, aligning, and engineering monthly variables")
    data, processed_path = build_processed_data(RAW_DIR, PROCESSED_DIR)
    print(f"  processed {len(data)} months -> {processed_path}")

    print("[3/3] Calculating correlations and creating figures")
    findings = run_analysis(processed_path, FIGURES_DIR, OUTPUT_DIR)
    print(json.dumps(findings, ensure_ascii=False, indent=2))
    print(f"  figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()

