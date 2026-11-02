"""Run every internship task end to end and report whether each one passed.

Usage:
    python src/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
TASKS = [
    ("Task 1", "Netflix Content Recommendation System", "task1_recommendation.py"),
    ("Task 2", "Content Type Prediction Model", "task2_content_type.py"),
    ("Task 4", "Netflix Content Segmentation", "task4_segmentation.py"),
    ("Task 6", "Content Success Analytics Engine", "task6_analytics_engine.py"),
]


def main() -> int:
    results = []
    for number, title, script in TASKS:
        print("=" * 72)
        print(f"{number}: {title}")
        print("=" * 72)

        started = time.time()
        outcome = subprocess.run([sys.executable, str(SRC / script)], cwd=SRC.parent)
        elapsed = time.time() - started

        results.append((number, title, outcome.returncode == 0, elapsed))
        print()

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    for number, title, ok, elapsed in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {number:<8} {title:<42} {elapsed:6.1f}s")

    failed = [r for r in results if not r[2]]
    print(f"\n{len(results) - len(failed)} of {len(results)} tasks completed successfully.")
    print("Figures: outputs/figures/   Reports: outputs/reports/")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
